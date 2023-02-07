import torch
from torch.utils.data import DataLoader

import os, wandb

from transformers import GPT2Tokenizer, GPT2LMHeadModel, logging, AdamW
from transformers.optimization import get_constant_schedule_with_warmup
logging.set_verbosity_error()
from datasets import load_dataset

from model_gpt2 import GPT2Head

def process_openwebtext(example, tokenizer, max_length):
    txts = example['text']

    encodings_dict = tokenizer(txts, truncation=True, max_length=max_length, padding="max_length")
    input_ids = torch.tensor(encodings_dict['input_ids'])
    attn_masks = torch.tensor(encodings_dict['attention_mask'])

    return input_ids, attn_masks

class Runner():
    def __init__(self, cfg):

        print(cfg)

        self.cfg = cfg

        self.device_id = 'cuda:{}'.format(cfg.local_rank)
        self.master_node = (self.cfg.local_rank == 0)
        self.distributed = (self.cfg.world_size > 1)

        self.model = GPT2LMHeadModel.from_pretrained(self.cfg.gpt_name)

        with torch.no_grad():
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(self.cfg.gpt_name, pad_token='<|endoftext|>')
            self.scorer = GPT2LMHeadModel.from_pretrained(self.cfg.gpt_name)

            for param in self.scorer.parameters():
                param.requires_grad = False
            self.scorer.eval()

        self.model_wrap = GPT2Head(_len=self.cfg.context, model=self.model, scorer=self.scorer, device=self.device_id)
        self.model_wrap.to(self.device_id)
        self.scorer.to(self.device_id)

        dataset = load_dataset('openwebtext', split='train')
        self.train_loader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True)

        if self.distributed:
            self.model_wrap = torch.nn.parallel.DistributedDataParallel(self.model_wrap, device_ids=[cfg.local_rank], output_device=cfg.local_rank)
            self.model_wrap = self.model_wrap.module
        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=cfg.world_size, rank=cfg.local_rank, shuffle=True, drop_last=False)
            self.train_loader = DataLoader(dataset, batch_size=cfg.batch, sampler=self.train_sampler)

        self.clip_grad = 1.0
        self.optimizer, self.scheduler = self._get_optimizer()

        self.epoch = 0 # do this before loading

        if self.cfg.loadpath is not None:
            self.load(self.cfg.loadpath)

        self.save_every = 4
        self.eval_every = 1

    def _get_optimizer(self):

        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model_wrap.named_parameters() if not any(nd in n for nd in no_decay) and not n.startswith('scorer')],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in self.model_wrap.named_parameters() if any(nd in n for nd in no_decay) and not n.startswith('scorer')],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.cfg.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=500)
        
        return optimizer, scheduler

    def getpath(self):
        return os.path.join(self.cfg.model_dir, 'checkpoint_{}.pth'.format(self.epoch))

    def load(self, path):
        map_location = {"cuda:0": self.device_id}
        checkpoint = torch.load(path, map_location=map_location)
        self.model_wrap.load_state_dict(checkpoint['model'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']

        print("loaded", flush=True)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.cfg.lr
            print(param_group['lr'])

    def train(self):

        model = self.model_wrap
        dataloader = self.train_loader

        while self.epoch < self.cfg.n_epochs:
            epoch_metrics = {
                'lossN': 0,
                'lossT': 0,
                'lossK': 0,
                'count': 0,
            }

            bsz = 0
            accum, accumN, accumT = 0, 0.0, 0.0

            print("epoch ", self.epoch, flush=True)
            if self.distributed: self.train_sampler.set_epoch(self.epoch + 1000) # ! important for shuffling
            model.train()
            
            if self.cfg.data_source == 'data':
                data_source_fn = lambda ex, _: process_openwebtext(ex, self.gpt2_tokenizer, self.cfg.context)
            elif self.cfg.data_source == 'sample':
                data_source_fn = lambda _: self.sample_model(batch=self.cfg.batch, length=self.cfg.context)
            elif self.cfg.data_source == 'mix':
                def data_source_fn(ex, it): 
                    if it % 5 == 0: # data-pool: 20% self-generation, 80% corpus
                        return self.sample_model(batch=self.cfg.batch, length=self.cfg.context)
                    else:
                        return process_openwebtext(ex, self.gpt2_tokenizer, self.cfg.context)
            else:
                raise NotImplementedError

            steps = 0

            for it, ex in enumerate(dataloader):
                input_ids, attn_mask = data_source_fn(ex, it)
                input_ids = input_ids.cuda(device=self.device_id, non_blocking=True)
                attn_mask = attn_mask.cuda(device=self.device_id, non_blocking=True)

                debug = False #(it == 0) and self.epoch % 10 == 0

                lossT, info = model.temperature_horizon_loss(input_ids, attn_mask, self.cfg.horizon_loss, debug )
                lossN = info['loss']
                lossK = info['kll']
                lossT.backward()

                count = input_ids.shape[0]
                epoch_metrics['lossN'] += lossN * count
                epoch_metrics['lossT'] += lossT * count
                epoch_metrics['lossK'] += lossK * count
                epoch_metrics['count'] += count

                bsz += input_ids.shape[0]
                accum += input_ids.shape[0]
                accumN += lossN * count
                accumT += lossT * count

                if bsz >= self.cfg.batch_step // self.cfg.world_size:
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    # print(total_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    bsz = 0
                    steps += 1

                if accum >= self.cfg.batch_step // self.cfg.world_size:
                    if self.master_node:
                        last_lr = self.scheduler.get_last_lr()
                        print("Iter %u out of %u, lossN: %.2f, lossT: %.2f, lr: %f" % (it, len(dataloader), lossN, lossT, last_lr[0]))
                        wandb.log({
                            "iter": (it + 1 + len(dataloader)*self.epoch) * self.cfg.batch,
                            "batch lossN": lossN,
                            "batch lossT": lossT,
                            "clamp ratio": info['clamp_entries'] / info['total_entries'],
                        })
                        accum = 0
                        accumN = 0.0
                        accumT = 0.0

                # each epoch is 100 steps
                if steps >= 100:
                    break

            if self.master_node:
                model_state_dict = self.model_wrap.state_dict()
                model_state_dict = {k : model_state_dict[k] for k in model_state_dict.keys() if not k.startswith('scorer')}
                states = {
                    'model': model_state_dict,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'epoch': self.epoch + 1,
                }

                torch.save(states, os.path.join(self.cfg.model_dir, 'checkpoint.pth'))
                if self.epoch > 0 and self.epoch % self.save_every == 0:
                    torch.save(states, os.path.join(self.cfg.model_dir, 'checkpoint_{}.pth'.format(self.epoch)))

            if self.epoch % self.eval_every == 0:
                with torch.no_grad():
                    metric_tensor = torch.tensor( [epoch_metrics['lossN'], epoch_metrics['lossT'],epoch_metrics['lossK'], epoch_metrics['count'] ] )
                    if self.distributed:
                        torch.distributed.reduce(metric_tensor, dst=0)

                self.test()

                if self.master_node:
                    metric_tensor[0] /= metric_tensor[3]
                    metric_tensor[1] /= metric_tensor[3]
                    metric_tensor[2] /= metric_tensor[3]
                    wandb.log({
                        "epoch": self.epoch,
                        "train lossN": metric_tensor[0],
                        "train lossT": metric_tensor[1],
                        "train lossK": metric_tensor[2],
                    })

                    print("Epoch %u out of %u, train lossN: %.2f, train lossT: %.2f, train lossK: %.2f" % (self.epoch, self.cfg.n_epochs, metric_tensor[0], metric_tensor[1], metric_tensor[2]))

            self.epoch += 1

    def sample_model(self, batch, length):
        with torch.no_grad():
            self.model_wrap.eval()
            
            sample_outputs = self.model.generate(
                do_sample=True,
                max_length=length,
                num_return_sequences=batch,
                temperature = 1.0,
                pad_token_id = self.gpt2_tokenizer.pad_token_id,
                bos_token_id = self.gpt2_tokenizer.bos_token_id,
                eos_token_id = self.gpt2_tokenizer.eos_token_id,
            )

            input_ids = []
            attn_mask = []

            for i, sample_output in enumerate(sample_outputs):
                txt = self.gpt2_tokenizer.decode(sample_output, skip_special_tokens=True)
                encodings_dict = self.gpt2_tokenizer(txt + '<|endoftext|>', truncation=True, max_length=length, padding="max_length")
                _input_ids = torch.tensor(encodings_dict['input_ids'])
                _attn_mask = torch.tensor(encodings_dict['attention_mask'])

                input_ids.append(_input_ids)
                attn_mask.append(_attn_mask)

            input_ids = torch.stack(input_ids, dim=0)
            attn_mask = torch.stack(attn_mask, dim=0)
        
            self.model_wrap.train()

        return input_ids, attn_mask


    def _score_samples(self, batch, c_myopic, length, debug=False):
        loss_fct = torch.nn.CrossEntropyLoss(reduce=False)
        
        with torch.no_grad():
            self.model_wrap.eval()

            samp_score_total = 0

            sample_outputs = self.model.generate(
                                    do_sample=True,
                                    max_length=length,
                                    num_return_sequences=batch,
                                    temperature = c_myopic,
                                    pad_token_id = self.gpt2_tokenizer.pad_token_id,
                                    bos_token_id = self.gpt2_tokenizer.bos_token_id,
                                    eos_token_id = self.gpt2_tokenizer.eos_token_id,
                                )

            sample_scores = []
            sample_txts = []
            sample_lengths = []

            for i, sample_output in enumerate(sample_outputs):
                sample_output = sample_output[1:] # get rid of first token
                txt = self.gpt2_tokenizer.decode(sample_output, skip_special_tokens=True)
                encodings_dict = self.gpt2_tokenizer(txt + '<|endoftext|>', truncation=True, max_length=self.cfg.context, padding="max_length")
                input_ids = torch.tensor(encodings_dict['input_ids'])
                attn_mask = torch.tensor(encodings_dict['attention_mask'])
                _length = attn_mask.sum()

                input_ids = input_ids.cuda(device=self.device_id, non_blocking=True)
                attn_mask = attn_mask.cuda(device=self.device_id, non_blocking=True)

                lm_logits = self.scorer(input_ids, attention_mask=attn_mask).logits

                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                # Flatten the tokens
                selected_logits = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                selected_logits = selected_logits.reshape(shift_labels.shape)
                score = selected_logits

                samp_score = score[:_length].mean()
                samp_score_total = samp_score_total + samp_score

                sample_scores.append(samp_score)
                sample_txts.append(txt)
                sample_lengths.append(_length)
                #print(txt)

            if self.master_node:
                if debug:
                    self._text_tab = wandb.Table(columns=["epoch", "i", "text", "score"])

                    for i in range(len(sample_txts)):
                        self._text_tab.add_data(self.epoch, i, sample_txts[i], sample_scores[i])
                    wandb.log({
                        "text": self._text_tab,
                    })
            
            return samp_score_total / batch

    def test(self):
        print("testing")

        epoch_metrics = {
            'sample_score': 0,
        }

        with torch.no_grad():
            avgp = 0
            length = self.cfg.context

            outer_batch = 100
            tot_batch = outer_batch
            tot_score = 0
            
            while tot_batch > 0:
                debug = (tot_batch == outer_batch)
                cur_batch = min(self.cfg.batch, tot_batch)
                samp_score = self._score_samples(batch=cur_batch, c_myopic=1.0, length=length, debug=debug)
                tot_score += samp_score * cur_batch
                tot_batch -= cur_batch
            avg_score = tot_score / outer_batch

            print("sample_score: %.4f" % (avg_score))

            if self.master_node:
                wandb.log({
                    "epoch": self.epoch,
                    "score_h" : avg_score,
                })

            epoch_metrics['sample_score'] = avg_score


        with torch.no_grad():
            metric_tensor = torch.tensor( [ epoch_metrics['sample_score'] ] )
            if self.distributed:
                torch.distributed.reduce(metric_tensor, dst=0)
                metric_tensor[0] /= self.cfg.world_size

            if self.master_node:
                print("test count %u sample_score: %.4f" % (outer_batch, metric_tensor[0]))

        return metric_tensor
