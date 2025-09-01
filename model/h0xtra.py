#  adapted from https://github.com/karpathy/nanoGPT (copyright (c) 2022 Andrej Karpathy)

"""
This file contains the overall pytorch model, combining the hypernetwork and the transformer backbone
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from model.transformer import GPT
from model.hypernetwork import HyperNetwork
import inspect


class H0xtra(nn.Module):
    def __init__(self, HNconfig, GPTconfig, task='trajgen'):
        super().__init__()

        self.HNconfig = HNconfig
        self.GPTconfig = GPTconfig
        self.task = task

        self.gpt = GPT(self.GPTconfig)
        self.hypernetwork = HyperNetwork(self.HNconfig)

        # fixed embedding for <pad> location
        self.embedding_pad = nn.Parameter(torch.zeros((1, self.GPTconfig.n_embd), dtype=torch.float32), requires_grad=False)

        # learnable embedding for the <start> location (only for generation)
        if task == 'trajgen':
            self.start_token = nn.Parameter(torch.zeros((1, self.GPTconfig.n_embd), dtype=torch.float32))
            torch.nn.init.normal_(self.start_token, mean=0.0, std=0.02)

    def forward(self, idx, matrix=None, targets=None):

        day = idx[:,:,1]
        hour = idx[:,:,2]
        idx = idx[:,:,0]
        if targets is not None:
            targets = targets[:,:,0]

        b, l = idx.shape
        device = idx.device

        emb = self.hypernetwork(matrix)

        assert emb.shape == (b, 40000, self.GPTconfig.n_embd), 'Shape of embedding matrix does not match number of locations. Please check hypernetwork input and configuration.'

        if self.task == 'trajgen':
            # <start> is set to last index
            emb = torch.cat((self.embedding_pad.expand((b, 1, self.GPTconfig.n_embd)), emb, self.start_token.expand((b, 1, self.GPTconfig.n_embd))), dim=1).contiguous()
        else:
            emb = torch.cat((self.embedding_pad.expand((b, 1, self.GPTconfig.n_embd)), emb), dim=1).contiguous()
        
        # embed the location ids with the learned embedding matrix
        embedded = torch.gather(emb, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, self.GPTconfig.n_embd))
        assert embedded.shape == (b, l, self.GPTconfig.n_embd), 'Embedded trajectory has wrong shape. Location embedding went wrong.'

        day_emb = self.gpt.embed_day(day)
        hour_emb = self.gpt.embed_hour(hour)
        time_emb = self.gpt.ln_time(day_emb + hour_emb)
        embedded = torch.cat((embedded, time_emb), dim=-1)

        # pass through transformer
        pos = torch.arange(0, l, dtype=torch.long, device=device) # shape (l)
        pos_emb = self.gpt.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.gpt.transformer.drop(embedded + pos_emb)
        for block in self.gpt.transformer.h:
            x = block(x)
        out = self.gpt.transformer.ln_f(x)
        out = self.gpt.down_proj(out)

        # compute logits using the learned embedding matrix
        logits = torch.matmul(emb.unsqueeze(1), out.unsqueeze(-1)).squeeze(-1)
        
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.contiguous().size(-1)), targets.reshape(-1), ignore_index=0, reduction='mean')
            return logits, loss

        else:
            loss = None

        return logits, loss
    

    @torch.no_grad
    def generate(self, idx, matrix, targets=None, test_last=False, lengths=None):
        """
        test routine
        """

        if test_last:
            assert not lengths is None, 'If only testing for the last location prediction, lengths must not be None'

        day = idx[:,:,1]
        hour = idx[:,:,2]
        idx = idx[:,:,0]
        if targets is not None:
            targets = targets[:,:,0]

        b, l = idx.shape
        device = idx.device

        emb = self.hypernetwork(matrix)

        assert emb.shape == (b, 40000, self.GPTconfig.n_embd), 'Shape of embedding matrix does not match number of locations. Please check hypernetwork input and configuration.'

        if self.task == 'trajgen':
            emb = torch.cat((self.embedding_pad.expand((b, 1, self.GPTconfig.n_embd)), emb, self.start_token.expand((b, 1, self.GPTconfig.n_embd))), dim=1)
        else:
            emb = torch.cat((self.embedding_pad.expand((b, 1, self.GPTconfig.n_embd)), emb), dim=1)

        embedded = torch.gather(emb, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, self.GPTconfig.n_embd))
        assert embedded.shape == (b, l, self.GPTconfig.n_embd), 'Embedded trajectory has wrong shape. Location embedding went wrong.'

        day_emb = self.gpt.embed_day(day)
        hour_emb = self.gpt.embed_hour(hour)
        time_emb = self.gpt.ln_time(day_emb + hour_emb)
        embedded = torch.cat((embedded, time_emb), dim=-1)

        pos = torch.arange(0, l, dtype=torch.long, device=device)
        pos_emb = self.gpt.transformer.wpe(pos)
        x = self.gpt.transformer.drop(embedded + pos_emb)
        for block in self.gpt.transformer.h:
            x = block(x)
        out = self.gpt.transformer.ln_f(x)
        out = self.gpt.down_proj(out)
    
        if test_last:
            final_out_index = lengths - 1
            targets = torch.gather(targets, 1, final_out_index.unsqueeze(-1))
            final_out_index = final_out_index.reshape(final_out_index.shape[0], 1, -1)
            final_out_index = final_out_index.repeat(1, 1, self.GPTconfig.n_embd)
            out = torch.gather(out, 1, final_out_index).squeeze(1)  # batch_size * hidden_size
            logits = torch.matmul(emb, out.unsqueeze(-1)).reshape(b, 1, -1)
        else:
            logits = torch.matmul(emb.unsqueeze(1), out.unsqueeze(-1)).squeeze(-1) # emb must be of shape (batch size, ouput dim, input dim)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(logits.reshape(-1, logits.contiguous().size(-1)), targets.reshape(-1), ignore_index=0, reduction='mean')
            return logits, targets, loss

        return logits
    
    @torch.no_grad
    def generate_fixed_emb(self, idx, targets=None, test_last=False, lengths=None):
        """
        Test routine with fixed location embeddings. 
        """

        assert isinstance(self.emb, nn.Embedding), 'For testing with fixed location embeddings please first run fix_embeddings!'

        if test_last:
            assert not lengths is None, 'If only testing for the last location prediction, lengths must not be None'

        day = idx[:,:,1]
        hour = idx[:,:,2]
        idx = idx[:,:,0]
        if targets is not None:
            targets = targets[:,:,0]

        b, l = idx.shape
        device = idx.device

        embedded = self.emb(idx)
        assert embedded.shape == (b, l, self.GPTconfig.n_embd), 'Embedded trajectory has wrong shape. Location embedding went wrong.'

        day_emb = self.gpt.embed_day(day)
        hour_emb = self.gpt.embed_hour(hour)
        time_emb = self.gpt.ln_time(day_emb + hour_emb)
        embedded = torch.cat((embedded, time_emb), dim=-1)

        pos = torch.arange(0, l, dtype=torch.long, device=device)
        pos_emb = self.gpt.transformer.wpe(pos)
        x = self.gpt.transformer.drop(embedded + pos_emb)
        for block in self.gpt.transformer.h:
            x = block(x)
        out = self.gpt.transformer.ln_f(x)
        out = self.gpt.down_proj(out)
    
        if test_last:
            final_out_index = lengths - 1
            targets = torch.gather(targets, 1, final_out_index.unsqueeze(-1))
            final_out_index = final_out_index.reshape(final_out_index.shape[0], 1, -1)
            final_out_index = final_out_index.repeat(1, 1, self.GPTconfig.n_embd)
            out = torch.gather(out, 1, final_out_index).squeeze(1)
            logits = self.head(out).reshape(b, 1, -1)
        else:
            logits = self.head(out)

        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.contiguous().size(-1)), targets.reshape(-1), ignore_index=0, reduction='mean')
            return logits, targets, loss

        return logits
    
    @torch.no_grad()
    def fix_embeddings(self, matrix):
        """
        fix location embeddings with a single hypernetwork pass
        """

        emb = self.hypernetwork(matrix.unsqueeze(0)).squeeze(0)

        assert emb.shape == (40000, self.GPTconfig.n_embd), 'Shape of embedding matrix does not match number of locations. Please check hypernetwork input and configuration.'

        if self.task == 'trajgen':
            emb = torch.cat((self.embedding_pad, emb, self.start_token), dim=0)
            assert emb.shape == (40002, self.GPTconfig.n_embd), '<start> or <pad> locations were not properly embedded.'
        else:
            emb = torch.cat((self.embedding_pad, emb), dim=0)
            assert emb.shape == (40001, self.GPTconfig.n_embd), '<pad> location was not properly embedded.'

        self.emb = nn.Embedding.from_pretrained(emb, freeze=True)
        self.head = nn.Linear(self.GPTconfig.n_embd, emb.shape[0], bias=False)
        self.head.weight = self.emb.weight

        print(f'fixed embeddings with embedding matrix of shape {self.emb.weight.shape}')

    def get_num_params(self):
        """
        Return the number of parameters in the hypernetwork.
        """
        n_params = sum(p.numel() for p in self.hypernetwork.parameters())
        return n_params

    def configure_optimizer(self, weight_decay, learning_rate, betas, device_type):
        """
        adapted from https://github.com/karpathy/nanoGPT
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]  # basically all biases and laynorm weights
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer