"""
Optimizers Module

Contains optimizer implementations and factory functions:
- CPUOffloadOptimizer: Memory-efficient optimizer with CPU offloading
- create_optimizer: Factory function for creating optimizers
"""

import torch
import torch.optim as optim
import math
from config import training_config


class CPUOffloadOptimizer:
    """CPU-Offloading Optimizer für Memory-Effizienz."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.param_groups = [{'params': list(params), 'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}]
        self.state = {}
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # CPU-Offloading Setup
        self.cpu_states = {}
        self._setup_cpu_offload()

    def _setup_cpu_offload(self):
        """Initialisiere CPU-Offloading für Optimizer States."""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    param_id = id(p)
                    # Erstelle CPU-Kopien der Optimizer States
                    self.cpu_states[param_id] = {
                        'exp_avg': torch.zeros_like(p.data, device='cpu'),
                        'exp_avg_sq': torch.zeros_like(p.data, device='cpu'),
                        'step': 0
                    }

    def step(self):
        """Optimizer Step mit CPU-Offloading."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                param_id = id(p)
                state = self.cpu_states[param_id]

                # Lade States von CPU zu GPU
                exp_avg = state['exp_avg'].to(p.device)
                exp_avg_sq = state['exp_avg_sq'].to(p.device)

                state['step'] += 1

                # AdamW Update
                grad = p.grad.data
                if self.weight_decay != 0:
                    grad = grad.add(p.data, alpha=self.weight_decay)

                # Exponential moving average of gradient values
                exp_avg.mul_(self.betas[0]).add_(grad, alpha=1 - self.betas[0])

                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(self.betas[1]).addcmul_(grad, grad, value=1 - self.betas[1])

                # Bias correction
                bias_correction1 = 1 - self.betas[0] ** state['step']
                bias_correction2 = 1 - self.betas[1] ** state['step']

                step_size = self.lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                # Update parameters
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(self.eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Speichere States zurück auf CPU
                state['exp_avg'].copy_(exp_avg.cpu())
                state['exp_avg_sq'].copy_(exp_avg_sq.cpu())

                # Cleanup GPU Memory
                del exp_avg, exp_avg_sq

    def zero_grad(self, set_to_none=False):
        """Setze Gradienten zurück."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_()
                        p.grad.zero_()


def create_optimizer(model):
    """Erstellt Optimizer für das Modell mit GPT-5 Weight Decay Parameter Groups."""

    # GPT-5: Separate Parameter Groups für Weight Decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            # Keine Weight Decay für: LayerNorms, Biases, Embeddings
            if any(nd in name.lower() for nd in ['bias', 'norm', 'embedding']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': training_config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    # Optimizer Selection
    if training_config.optimizer_type == "adamw_fused":
        try:
            optimizer = optim.AdamW(
                param_groups,
                lr=training_config.learning_rate,
                betas=(training_config.adam_beta1, training_config.adam_beta2),
                eps=training_config.adam_eps,
                fused=True  # Fused AdamW für RTX 3090/4090
            )
        except:
            # Fallback: Standard AdamW
            optimizer = optim.AdamW(
                param_groups,
                lr=training_config.learning_rate,
                betas=(training_config.adam_beta1, training_config.adam_beta2),
                eps=training_config.adam_eps
            )
    
    elif training_config.optimizer_type == "cpu_offload":
        optimizer = CPUOffloadOptimizer(
            model.parameters(),
            lr=training_config.learning_rate,
            betas=(training_config.adam_beta1, training_config.adam_beta2),
            eps=training_config.adam_eps,
            weight_decay=training_config.weight_decay
        )
    
    else:  # Standard AdamW
        optimizer = optim.AdamW(
            param_groups,
            lr=training_config.learning_rate,
            betas=(training_config.adam_beta1, training_config.adam_beta2),
            eps=training_config.adam_eps
        )

    return optimizer
