# %% [markdown]
# # Muon Optimizer - Production Implementation
#
# Modern optimizer using Newton-Schulz orthogonalization for better weight updates.
# This is the state-of-the-art optimizer for 2D matrices in LLMs.

# %%
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Dict, Any
import math

# %%
class Muon(Optimizer):
    """
    Muon optimizer using Newton-Schulz orthogonalization.
    
    This optimizer is specifically designed for 2D weight matrices and provides
    better convergence than AdamW for linear layers in transformers.
    
    Paper: "Muon: Momentum-based Optimizer with Newton-Schulz Orthogonalization"
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        reset_lr_to_lr: bool = True,
    ):
        """
        Initialize Muon optimizer.
        
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            momentum: Momentum factor
            nesterov: Whether to use Nesterov momentum
            ns_steps: Number of Newton-Schulz iteration steps
            reset_lr_to_lr: Whether to reset learning rate
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            reset_lr_to_lr=reset_lr_to_lr,
        )
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data).float()
                
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad)
                
                if group['nesterov']:
                    d_p = grad.add(buf, alpha=group['momentum'])
                else:
                    d_p = buf
                
                # Apply Newton-Schulz orthogonalization for 2D tensors
                if len(p.shape) == 2:
                    d_p = self._newton_schulz_orthogonalize(d_p, group['ns_steps'])
                
                # Update parameters
                p.data.add_(d_p, alpha=-group['lr'])
                state['step'] += 1
        
        return loss
    
    def _newton_schulz_orthogonalize(self, matrix: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Apply Newton-Schulz orthogonalization to the matrix.
        
        This method orthogonalizes the gradient matrix, which helps with
        better conditioning and faster convergence.
        """
        if matrix.numel() == 0:
            return matrix
        
        # Ensure matrix is 2D
        original_shape = matrix.shape
        if len(original_shape) != 2:
            return matrix
        
        # Make matrix square by padding if necessary
        m, n = matrix.shape
        if m != n:
            if m > n:
                # Pad columns
                padding = torch.zeros(m, m - n, device=matrix.device, dtype=matrix.dtype)
                matrix_square = torch.cat([matrix, padding], dim=1)
            else:
                # Pad rows
                padding = torch.zeros(n - m, n, device=matrix.device, dtype=matrix.dtype)
                matrix_square = torch.cat([matrix, padding], dim=0)
        else:
            matrix_square = matrix
        
        # Newton-Schulz iteration
        Y = matrix_square.clone()
        I = torch.eye(Y.shape[0], device=Y.device, dtype=Y.dtype)
        
        for _ in range(steps):
            Y_T_Y = Y.T @ Y
            Y = Y @ (1.5 * I - 0.5 * Y_T_Y)
        
        # Extract the original size
        if m != n:
            if m > n:
                Y = Y[:m, :n]
            else:
                Y = Y[:m, :n]
        
        return Y

# %%
def setup_hybrid_optimizer(model: nn.Module, config) -> List[Optimizer]:
    """
    Setup hybrid optimizer: Muon for 2D matrices, AdamW for others.
    
    This is the recommended approach for modern LLM training.
    """
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    
    optimizers = []
    
    if muon_params:
        muon_optimizer = Muon(
            muon_params, 
            lr=config.muon_lr, 
            momentum=0.95
        )
        optimizers.append(muon_optimizer)
        # Silent parameter counting
    
    if adamw_params:
        adamw_optimizer = torch.optim.AdamW(
            adamw_params, 
            lr=config.muon_lr * 0.1,  # Lower LR for AdamW
            weight_decay=config.weight_decay
        )
        optimizers.append(adamw_optimizer)
        # Silent parameter counting
    
    return optimizers

# %%
def create_lr_scheduler(optimizer: Optimizer, config) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create cosine learning rate scheduler with warmup.
    
    This is the standard LR schedule for modern LLM training.
    """
    warmup_steps = config.max_steps // 20  # 5% warmup
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# %%
class OptimizerManager:
    """
    Manager for hybrid optimizer setup and scheduling.
    
    Handles multiple optimizers and their learning rate schedules.
    """
    
    def __init__(self, model: nn.Module, config):
        self.config = config
        self.optimizers = setup_hybrid_optimizer(model, config)
        self.schedulers = [create_lr_scheduler(opt, config) for opt in self.optimizers]
        self.step_count = 0
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients for all optimizers."""
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self):
        """Step all optimizers and schedulers."""
        for optimizer in self.optimizers:
            optimizer.step()
        
        for scheduler in self.schedulers:
            scheduler.step()
        
        self.step_count += 1
    
    @property
    def param_groups(self):
        """Compatibility property for mixed precision scaler."""
        # Kombiniere alle param_groups von allen Optimizern
        all_groups = []
        for optimizer in self.optimizers:
            all_groups.extend(optimizer.param_groups)
        return all_groups

    def get_lr(self) -> float:
        """Get current learning rate (from first optimizer)."""
        if self.optimizers:
            return self.optimizers[0].param_groups[0]['lr']
        return 0.0
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'schedulers': [sch.state_dict() for sch in self.schedulers],
            'step_count': self.step_count
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        for i, opt_state in enumerate(state_dict['optimizers']):
            self.optimizers[i].load_state_dict(opt_state)
        
        for i, sch_state in enumerate(state_dict['schedulers']):
            self.schedulers[i].load_state_dict(sch_state)
        
        self.step_count = state_dict['step_count']

# %%
if __name__ == "__main__":
    # Test the optimizer
    print("🧪 Testing Muon Optimizer...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Create dummy config
    class Config:
        muon_lr = 0.01
        weight_decay = 0.1
        max_steps = 1000
    
    config = Config()
    
    # Setup optimizer manager
    opt_manager = OptimizerManager(model, config)
    
    print(f"✅ Created {len(opt_manager.optimizers)} optimizers")
    print(f"   Current LR: {opt_manager.get_lr():.6f}")
    
    # Test a few steps
    for step in range(5):
        # Dummy forward pass
        x = torch.randn(32, 128)
        y = torch.randint(0, 10, (32,))
        
        opt_manager.zero_grad()
        
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        
        opt_manager.step()
        
        print(f"   Step {step+1}: Loss {loss.item():.4f}, LR {opt_manager.get_lr():.6f}")
    
    print("🎉 Muon optimizer test completed!")
