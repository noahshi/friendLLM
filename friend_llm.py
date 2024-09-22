import torch
import torch.nn.functional as F
import math
import random
from gpt import GPT, GPTConfig
from tokenizer import RegexTokenizer
from data_loader import DataLoaderLite

class FriendLLM:
    def __init__(self) -> None:
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        print('using device:', self.device)
        
        # get logits
        self.model = GPT(GPTConfig(vocab_size=10112))
        self.model.to(self.device)
        
        self.enc = RegexTokenizer("./data/merges.json", "./data/vocab.json")
        
        # print(self.enc.banned_tokens)
        
    def train_model(self, steps):
        import time

        train_loader = DataLoaderLite(B=16, T=512, file='bytes2.json')
        torch.set_float32_matmul_precision('high')

        # torch.compile slows down training for some reason???
        # model = torch.compile(model, backend="aot_eager")

        max_lr = 6e-4
        min_lr = max_lr * 0.1
        warmup_steps = int(steps * 0.025)
        max_steps = steps

        def get_lr(it):
            # warmup
            if it < warmup_steps:
                return max_lr * (it + 1) / warmup_steps
            # min learning rate
            if it > max_steps:
                return min_lr
            # decay
            decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (max_lr - min_lr)

        optimizer = self.model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=self.device)

        for step in range(max_steps):
            t0 = time.time()
            x, y = train_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                logits, loss = self.model(x, y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            
            torch.cuda.synchronize()
            t1 = time.time()
            print(f"step {step:4d}, loss: {loss.item()}, lr: {lr:.4e}, norm: {norm:.4f} dt: {((t1-t0)*1000):.4f}ms")

        torch.save(self.model.state_dict(), f'model_{max_steps}.pth')

    def load_model(self, file):
        self.model.load_state_dict(torch.load(file, weights_only=True))
        
    def prompt(self, prompt):
        message_count = 0
        message_count_hard_cap = 60

        tokens = self.enc.encode(prompt)
        tokens = torch.tensor(tokens).unsqueeze(0)
        x = tokens.to(self.device)
        
        def get_stop_prob(it):
            if it > message_count_hard_cap:
                return 0

            decay_ratio = it / message_count_hard_cap
            assert 0 <= decay_ratio <= 1
            return 0.5 * (1 + math.cos(math.pi * decay_ratio))

        # torch.manual_seed(42)
        # torch.cuda.manual_seed(42)
        while message_count < message_count_hard_cap:
            with torch.no_grad():
                logits, _ = self.model(x)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1)
                xcol = torch.gather(topk_indices, -1, ix)
                if xcol.item() in self.enc.banned_tokens:
                    print('bad token:', xcol.item())
                    continue
                
                x = torch.cat((x, xcol), dim=1)
                
                if xcol.item() in self.enc.eom_tokens:
                    if(random.random() > get_stop_prob(message_count)):
                        break
                    message_count += 1

        tokens = x[0, :].tolist()
        decoded = self.enc.decode(tokens)
        print(decoded)
