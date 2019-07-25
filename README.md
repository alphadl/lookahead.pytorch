## lookahead optimizer for pytorch

PyTorch implement of <a href="https://arxiv.org/abs/1907.08610" target="_blank">Lookahead Optimizer: k steps forward, 1 step back</a>

Usage:
```
base_opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999)) # Any optimizer
lookahead = Lookahead(base_opt, k=5, alpha=0.5) # Initialize Lookahead
lookahead.zero_grad()
loss_function(model(input), target).backward() # Self-defined loss function
lookahead.step()
```
