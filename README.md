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

## lookahead优化器的PyTorch实现

论文<a href="https://arxiv.org/abs/1907.08610" target="_blank">《Lookahead Optimizer: k steps forward, 1 step back》</a>的Keras实现。

用法：
```
base_opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999)) # 用你想用的优化器
lookahead = Lookahead(base_opt, k=5, alpha=0.5) # 初始化Lookahead
lookahead.zero_grad()
loss_function(model(input), target).backward() # 自定义的损失函数
lookahead.step()
```

中文介绍：https://mp.weixin.qq.com/s/3J-28xd0pyToSy8zzKs1RA
