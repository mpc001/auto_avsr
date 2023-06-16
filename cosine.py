import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, num_epochs, iter_per_epoch):
        self.base_lrs = {
            param_group["name"]: param_group["lr"]
            for param_group in optimizer.param_groups
        }
        self.warmup_iter = warmup_epochs * iter_per_epoch
        self.total_iter = num_epochs * iter_per_epoch
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

        self.init_lr()  # so that at first step we have the correct step size

    def get_lr(self, base_lr):
        if self.iter < self.warmup_iter:
            return base_lr * self.iter / self.warmup_iter
        else:
            decay_iter = self.total_iter - self.warmup_iter
            return (
                0.5
                * base_lr
                * (1 + np.cos(np.pi * (self.iter - self.warmup_iter) / decay_iter))
            )

    def update_param_groups(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr(self.base_lrs[param_group["name"]])

    def step(self):
        self.update_param_groups()
        self.iter += 1

    def init_lr(self):
        self.update_param_groups()
