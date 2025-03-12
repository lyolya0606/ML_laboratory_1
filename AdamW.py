import torch
from torch.optim.optimizer import Optimizer

class AdamWCustom(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamWCustom, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param.data)
                    state["exp_avg_sq"] = torch.zeros_like(param.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # Update step
                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Compute denominator
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group["eps"])

                # Compute step size
                step_size = group["lr"] / bias_correction1

                # Weight decay (decoupled)
                if group["weight_decay"] != 0:
                    param.data.add_(param.data, alpha=-group["weight_decay"] * group["lr"])

                # Parameter update
                param.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss