import torch
from tokenizer import RegexTokenizer

class DataLoaderLite:
    def __init__(self, B, T, file=None):
        self.B = B
        self.T = T
        self.enc = RegexTokenizer("./data/merges.json", "vocab.json")
        
        if file is not None:
            import json
            with open(file, 'r', encoding='utf-8') as f:
                tokens = json.load(f)
        else:
            with open('output-merged.txt', 'r', encoding='utf-8') as f:
                text = f.read()
            tokens = self.enc.encode(text)
        
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(tokens)} tokens')
        print(f'1 epoch = {len(tokens) // (B * T)} batches')
        
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets
        
        # move position forwards
        self.current_position += B*T
        # reset if out of bounds
        if self.current_position + B*T + 1 >= len(self.tokens):
            self.current_position = 0
        return x, y
