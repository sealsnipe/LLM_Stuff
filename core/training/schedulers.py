"""
Learning Rate Schedulers Module

Contains learning rate scheduler implementations and factory functions.
Supports cosine, linear, and constant schedules with warmup.
"""

import torch
import math
from torch.optim.lr_scheduler import LambdaLR
from config import training_config


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr_ratio*initial_lr, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.
    """
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup phase: linear increase from 0 to 1
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to
    min_lr_ratio*initial_lr, after a warmup period during which it increases linearly between 0 and the initial lr.
    """
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup phase: linear increase from 0 to 1
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Linear decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        linear_decay = 1.0 - progress
        
        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * linear_decay
    
    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps):
    """
    Create a schedule with a learning rate that increases linearly from 0 to the initial lr set in the optimizer
    during a warmup period, then remains constant.
    """
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup phase: linear increase from 0 to 1
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Constant phase
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


def create_lr_scheduler(optimizer, num_training_steps):
    """
    Factory function to create learning rate scheduler based on training config.
    
    Args:
        optimizer: The optimizer to schedule
        num_training_steps: Total number of training steps
        
    Returns:
        Learning rate scheduler
    """
    
    # Use dynamic warmup steps if epoch-based training is enabled
    if training_config.use_epoch_based_training:
        warmup_steps = training_config.dynamic_warmup_steps
    else:
        warmup_steps = training_config.warmup_steps

    min_lr_ratio = training_config.min_lr_ratio
    scheduler_type = training_config.lr_scheduler
    
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio
        )
    
    elif scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio
        )
    
    elif scheduler_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_current_lr(scheduler):
    """Get current learning rate from scheduler."""
    return scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else scheduler.get_lr()[0]
