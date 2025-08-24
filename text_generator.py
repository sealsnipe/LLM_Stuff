# %% [markdown]
# # Modern Text Generator - GPU Optimized
#
# Production-ready text generation system with:
# - GPU optimization for both training and inference
# - Efficient sampling strategies
# - Batched generation
# - Memory optimization

# %%
import torch
import torch.nn.functional as F
import math
import random
import numpy as np
from transformers import AutoTokenizer
from typing import List, Optional, Tuple, Dict, Any
import warnings
import os
from tqdm import tqdm
warnings.filterwarnings('ignore')

# %%
def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")

# %%
class GPUTextGenerator:
    """
    GPU-optimized text generation system.
    
    Features:
    - Automatic GPU detection and usage
    - Mixed precision inference
    - Efficient sampling strategies
    - Batched generation support
    - Memory optimization
    """
    
    def __init__(
        self, 
        model_path: str = "final_model.pt", 
        tokenizer_path: str = "HuggingFaceTB/SmolLM-135M",
        device: str = "auto",
        use_amp: bool = True
    ):
        """
        Initialize the GPU text generator.
        
        Args:
            model_path: Path to the saved model checkpoint
            tokenizer_path: Path to the tokenizer
            device: Device to use ("auto", "cpu", "cuda")
            use_amp: Whether to use automatic mixed precision
        """
        self.device = self._get_device(device)
        self.use_amp = use_amp and self.device.type == "cuda"
        
        print(f"🔧 Using device: {self.device}")
        if self.use_amp:
            print("⚡ Mixed precision enabled")
        
        # Load tokenizer
        print("📚 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print("🤖 Loading model...")
        self.model, self.config = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Enable optimizations
        if hasattr(torch, 'compile') and self.device.type == "cuda":
            print("🚀 Compiling model for faster inference...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Vocabulary size: {self.config.vocab_size}")
        print(f"   Max sequence length: {self.config.max_seq_len}")
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                # Use the GPU with most memory
                gpu_count = torch.cuda.device_count()
                if gpu_count > 1:
                    # Find GPU with most free memory
                    max_memory = 0
                    best_gpu = 0
                    for i in range(gpu_count):
                        memory = torch.cuda.get_device_properties(i).total_memory
                        if memory > max_memory:
                            max_memory = memory
                            best_gpu = i
                    return torch.device(f"cuda:{best_gpu}")
                else:
                    return torch.device("cuda:0")
            else:
                print("⚠️  CUDA not available, using CPU")
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_model(self, model_path: str) -> Tuple[torch.nn.Module, Any]:
        """Load the trained model from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"⚠️  weights_only=True failed, trying weights_only=False: {e}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract config and create model
        config = checkpoint['config']
        
        # Import model class (assuming it's available)
        try:
            from ..models.modern_llm import ModernLLM
            model = ModernLLM(config)
        except ImportError:
            # Fallback to a simple model class
            print("⚠️  Using fallback model class")
            model = self._create_fallback_model(config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, config
    
    def _create_fallback_model(self, config):
        """Create a fallback model if imports fail."""
        import torch.nn as nn
        
        class FallbackModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
                self.layers = nn.ModuleList([
                    nn.TransformerDecoderLayer(
                        d_model=config.d_model,
                        nhead=config.n_heads,
                        dim_feedforward=config.d_ff,
                        dropout=config.dropout,
                        batch_first=True
                    ) for _ in range(config.n_layers)
                ])
                self.norm = nn.LayerNorm(config.d_model)
                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            
            def forward(self, x):
                x = self.token_embedding(x)
                for layer in self.layers:
                    x = layer(x, x)
                x = self.norm(x)
                return self.lm_head(x)
        
        return FallbackModel(config)
    
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        stop_tokens: Optional[List[str]] = None,
        batch_size: int = 1
    ) -> List[str]:
        """
        Generate text from a prompt with GPU optimization.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text (including prompt)
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (False for greedy decoding)
            num_return_sequences: Number of sequences to generate
            stop_tokens: List of tokens to stop generation at
            batch_size: Batch size for generation
            
        Returns:
            List of generated text sequences
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        
        # Convert stop tokens to IDs
        stop_token_ids = set()
        if stop_tokens:
            for token in stop_tokens:
                token_id = self.tokenizer.encode(token, add_special_tokens=False)
                if token_id:
                    stop_token_ids.update(token_id)
        
        # Add EOS token to stop tokens
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.add(self.tokenizer.eos_token_id)
        
        generated_sequences = []
        
        # Generate in batches
        for batch_start in range(0, num_return_sequences, batch_size):
            batch_end = min(batch_start + batch_size, num_return_sequences)
            current_batch_size = batch_end - batch_start
            
            # Replicate input for batch
            batch_input_ids = input_ids.repeat(current_batch_size, 1)
            
            # Generate batch
            batch_sequences = self._generate_batch(
                input_ids=batch_input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                stop_token_ids=stop_token_ids
            )
            
            # Decode to text
            for sequence in batch_sequences:
                generated_text = self.tokenizer.decode(sequence, skip_special_tokens=True)
                generated_sequences.append(generated_text)
        
        return generated_sequences
    
    def _generate_batch(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        stop_token_ids: set
    ) -> List[torch.Tensor]:
        """Generate a batch of sequences."""
        batch_size = input_ids.shape[0]
        current_ids = input_ids.clone()
        generated_length = current_ids.shape[1]
        
        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            while generated_length < max_length and not finished.all():
                # Get model predictions
                logits = self.model(current_ids)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    # Scatter back to original indices
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample or take argmax
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check for stop tokens
                for i in range(batch_size):
                    if not finished[i] and next_tokens[i].item() in stop_token_ids:
                        finished[i] = True
                
                # Don't append tokens for finished sequences
                next_tokens[finished] = self.tokenizer.pad_token_id or 0
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_tokens], dim=1)
                generated_length += 1
        
        return [current_ids[i] for i in range(batch_size)]
    
    def get_perplexity(self, text: str) -> float:
        """Calculate perplexity of the given text."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) < 2:
            return float('inf')
        
        input_ids = torch.tensor([tokens[:-1]], device=self.device)
        target_ids = torch.tensor([tokens[1:]], device=self.device)
        
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            with torch.no_grad():
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size), 
                    target_ids.view(-1)
                )
        
        return math.exp(loss.item())

# %%
def interactive_mode(generator: GPUTextGenerator):
    """Run interactive text generation mode."""
    print("\n🎭 Interactive GPU Text Generation Mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\n💭 Enter your prompt: ").strip()
            
            if prompt.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif prompt.lower() == 'help':
                print("\n📖 Available commands:")
                print("  help - Show this help message")
                print("  quit - Exit the program")
                print("  settings - Show current generation settings")
                print("  sample - Generate with sampling")
                print("  greedy - Generate with greedy decoding")
                print("  batch - Generate multiple sequences")
                print("\n💡 Tips:")
                print("  - Use 'sample' or 'greedy' prefix to change generation mode")
                print("  - Use 'batch' prefix to generate multiple sequences")
                print("  - Example: 'sample The quick brown fox'")
                continue
            elif prompt.lower() == 'settings':
                print("\n⚙️ Current settings:")
                print(f"  Device: {generator.device}")
                print(f"  Mixed Precision: {generator.use_amp}")
                print("  Temperature: 0.8")
                print("  Top-p: 0.9")
                print("  Top-k: 50")
                print("  Max length: 100")
                continue
            
            # Check for mode prefixes
            do_sample = True
            num_sequences = 1
            
            if prompt.startswith('sample '):
                prompt = prompt[7:]
                do_sample = True
            elif prompt.startswith('greedy '):
                prompt = prompt[7:]
                do_sample = False
            elif prompt.startswith('batch '):
                prompt = prompt[6:]
                do_sample = True
                num_sequences = 3
            
            if not prompt:
                continue
            
            print(f"\n🚀 Generating text on {generator.device}...")
            generated_texts = generator.generate(
                prompt=prompt,
                max_length=100,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                do_sample=do_sample,
                num_return_sequences=num_sequences
            )
            
            print(f"\n✨ Generated text:")
            print("-" * 40)
            for i, text in enumerate(generated_texts, 1):
                print(f"{i}. {text}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# %%
if __name__ == "__main__":
    # Set seed
    set_seed(42)
    
    # Initialize generator
    try:
        generator = GPUTextGenerator("final_model.pt")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Make sure you have a trained model checkpoint!")
        exit(1)
    
    # Run interactive mode
    interactive_mode(generator)
