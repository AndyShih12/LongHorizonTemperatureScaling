import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def chunk_cumsum(x, chunk=10):
    b, l = x.shape[:-1], x.shape[-1]

    pad_amt = chunk
    dummy = torch.zeros(*b, l+pad_amt, device=x.device)
    dummy[..., :l] = x
    x = dummy

    x = torch.sum(x, dim=1, keepdims=True) - torch.cumsum(x, dim=1) + x
    x = x[..., :l] - x[..., -l:]
    x = x.reshape(*b, l)
    return x

def debug_clamp(x, mx):
    total_entries = x.numel()
    clamp_entries = (x > mx).sum()
    x = torch.clamp(x, max=mx)

    return x, clamp_entries, total_entries

class GPT2Head(nn.Module):
    def __init__(self, _len, model, scorer, device):
        super().__init__()
        self.len = _len
        self.model = model
        self.scorer = scorer

        self.loss_fct = torch.nn.CrossEntropyLoss(reduce=False)
        self.kl_loss_fct = torch.nn.KLDivLoss(reduce=False, log_target=True)

        self._baseline_sum, self._baseline_cnt = 0, 1e-3
        self._baseline_sum_all, self._baseline_cnt_all = 0, 1e-3

    def sum_except_batch(self, x):
        return x.reshape(x.shape[0], -1).sum(-1)

    def _prepare_inputs(self, mod, input_ids, attn_mask):
        temp_attn = torch.ones( attn_mask.shape[0], 1, dtype=int, device=attn_mask.device )
        attn_mask = torch.cat( (temp_attn, attn_mask), dim=1 )[:, :-1]
        
        hidden_state = mod.transformer(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        return hidden_state

    def _model_forward(self, input_ids, attn_mask):
        x_hidden_state = self._prepare_inputs(self.model, input_ids, attn_mask)
        y = input_ids
        lm_logits = self.model.lm_head(x_hidden_state)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = y[..., 1:].contiguous()
        # Flatten the tokens
        selected_logits = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        selected_logits = selected_logits.reshape(shift_labels.shape)
        all_logits = shift_logits

        return selected_logits, all_logits

    # assume scorer_model is Autoregressive
    def _scorer_forward(self, input_ids, attn_mask):
        with torch.no_grad():
            x_hidden_state = self._prepare_inputs(self.scorer, input_ids, attn_mask)
            y = input_ids
            lm_logits = self.scorer.lm_head(x_hidden_state)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = y[..., 1:].contiguous()
            # Flatten the tokens
            selected_logits = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            selected_logits = selected_logits.reshape(shift_labels.shape)
            all_logits = shift_logits

            return selected_logits, all_logits

    def temperature_horizon_loss(self, input_ids, attn_mask, loss_cfg, debug=False):
        info = {}

        ctemp = torch.ones( (input_ids.shape[0], 1), device=input_ids.device ) * loss_cfg.T

        temp_attn = torch.ones( attn_mask.shape[0], 1, dtype=int, device=attn_mask.device )
        attn_mask_shifted = attn_mask[..., 1:]
        attn_mask_shifted_wend = torch.cat( (temp_attn, attn_mask_shifted), dim=1 )[:, :-1]

        with torch.no_grad():
            _logits, all_logits_scorer = self._scorer_forward(input_ids, attn_mask)
            score = - _logits
            score = score * attn_mask_shifted_wend

            self._baseline_sum += score.sum(dim=0)
            self._baseline_cnt += attn_mask_shifted_wend.sum(dim=0)
            score_baseline = self._baseline_sum / self._baseline_cnt
            score = score - (score_baseline[None,:] * attn_mask_shifted_wend)

        # reverse cumsum
        horizon_score = chunk_cumsum(score, chunk=loss_cfg.chunk)

        wt = (horizon_score / ctemp) - horizon_score
        wt, clamp_entries, total_entries = debug_clamp(wt, loss_cfg.clamp)
        info['clamp_entries'] = clamp_entries
        info['total_entries'] = total_entries

        selected_logits, all_logits_model = self._model_forward(input_ids, attn_mask)
        selected_logits = selected_logits * torch.exp(wt)

        ll = self.sum_except_batch( selected_logits * attn_mask_shifted_wend ) / self.sum_except_batch( attn_mask_shifted_wend )
        info['loss'] = ll.mean()

        # KL loss
        all_logits_scorer = F.log_softmax(all_logits_scorer, dim=-1)
        all_logits_model = F.log_softmax(all_logits_model, dim=-1)
        kll = self.kl_loss_fct(all_logits_scorer, all_logits_model).sum(dim=-1) * loss_cfg.kl_beta
        kll = self.sum_except_batch( kll * attn_mask_shifted_wend ) / self.sum_except_batch( attn_mask_shifted_wend )
        info['kll'] = kll.mean()

        ll = ll + kll

        return ll.mean(), info