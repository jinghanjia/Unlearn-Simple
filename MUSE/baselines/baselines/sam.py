import torch
from torch.optim import AdamW
class SAMAdamW(AdamW):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, rho=0.05):
        if rho <= 0:
            raise ValueError(f"Invalid neighborhood size: {rho}")
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            group['rho'] = rho

    @torch.no_grad()
    def step(self, closure) -> torch.Tensor:
        closure = torch.enable_grad()(closure)
        loss = closure().detach()

        for group in self.param_groups:
            grads = []
            params_with_grads = []

            rho = group['rho']

            for p in group['params']:
                if p.grad is not None:
                    grads.append(p.grad.clone().detach())
                    params_with_grads.append(p)
            device = grads[0].device

            grad_norm = torch.stack([g.detach().norm(2).to(device) for g in grads]).norm(2).item()
            epsilon = grads

            torch._foreach_mul_(epsilon, rho / grad_norm)

            torch._foreach_add_(params_with_grads, epsilon)
            closure()
            torch._foreach_sub_(params_with_grads, epsilon)

        super().step()
        return loss
    
