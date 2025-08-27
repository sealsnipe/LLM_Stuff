#!/usr/bin/env python3
"""
Simple Conversation Test for our trained 926M LLM
Tests if the model has learned to speak and respond to basic prompts
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from modern_llm import ModernLLM
from config import ModelConfig
from transformers import AutoTokenizer
import json

class SimpleConversationTester:
    def __init__(self, model_path="trained_models/modern_llm_926m_20250827_080157"):
        """Initialize the conversation tester with our trained model"""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.config = None
        
        print(f"🤖 Simple Conversation Tester")
        print(f"📁 Model Path: {model_path}")
        print(f"🔧 Device: {self.device}")
        print("-" * 50)
    
    def load_model(self):
        """Load the trained model and configuration"""
        try:
            # Load config
            config_path = os.path.join(self.model_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            self.config = ModelConfig(**config_dict)
            print(f"✅ Config loaded: {self.config.vocab_size} vocab, {self.config.num_layers} layers")
            
            # Initialize model
            self.model = ModernLLM(self.config)
            
            # Load trained weights
            model_path = os.path.join(self.model_path, "model.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weights not found: {model_path}")

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Try to load state dict with compatibility mapping
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                print(f"⚠️  Direct loading failed, trying compatibility mapping...")
                # Detect actual model dimensions from checkpoint
                old_state = checkpoint['model_state_dict']
                self._detect_and_fix_config(old_state)

                # Recreate model with correct config
                self.model = ModernLLM(self.config)

                # Map old architecture to new architecture
                new_state = self._map_old_to_new_state_dict(old_state)
                self.model.load_state_dict(new_state, strict=False)

            self.model.to(self.device)
            self.model.eval()

            # Load the CORRECT tokenizer (same as training!)
            print(f"🔧 Loading SmolLM tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"✅ Model loaded successfully!")
            print(f"📊 Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"🎯 Tokenizer: SmolLM-135M (vocab_size: {self.tokenizer.vocab_size})")

            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def _detect_and_fix_config(self, old_state):
        """Detect actual model dimensions from checkpoint and fix config"""
        # Detect hidden size from embeddings
        if 'token_embeddings.weight' in old_state:
            actual_hidden_size = old_state['token_embeddings.weight'].shape[1]
            if actual_hidden_size != self.config.hidden_size:
                print(f"🔧 Fixing hidden_size: {self.config.hidden_size} → {actual_hidden_size}")
                self.config.hidden_size = actual_hidden_size

        # Detect intermediate size from FFN
        if 'layers.0.gate_proj.weight' in old_state:
            actual_intermediate_size = old_state['layers.0.gate_proj.weight'].shape[0]
            if actual_intermediate_size != self.config.intermediate_size:
                print(f"🔧 Fixing intermediate_size: {self.config.intermediate_size} → {actual_intermediate_size}")
                self.config.intermediate_size = actual_intermediate_size

        # Detect number of attention heads from attention projections
        if 'layers.0.attention.q_proj.weight' in old_state:
            q_proj_out = old_state['layers.0.attention.q_proj.weight'].shape[0]
            actual_n_heads = q_proj_out // (self.config.hidden_size // self.config.num_attention_heads)
            if actual_n_heads != self.config.num_attention_heads:
                print(f"🔧 Fixing num_attention_heads: {self.config.num_attention_heads} → {actual_n_heads}")
                self.config.num_attention_heads = actual_n_heads

        # Detect KV heads
        if 'layers.0.attention.k_proj.weight' in old_state:
            k_proj_out = old_state['layers.0.attention.k_proj.weight'].shape[0]
            actual_kv_heads = k_proj_out // (self.config.hidden_size // self.config.num_attention_heads)
            if actual_kv_heads != self.config.num_key_value_heads:
                print(f"🔧 Fixing num_key_value_heads: {self.config.num_key_value_heads} → {actual_kv_heads}")
                self.config.num_key_value_heads = actual_kv_heads

    def _map_old_to_new_state_dict(self, old_state):
        """Map old model architecture to new architecture"""
        new_state = {}

        # Map embeddings
        if 'token_embeddings.weight' in old_state:
            new_state['token_embedding.weight'] = old_state['token_embeddings.weight']

        # Map layers to transformer_blocks
        for i in range(self.config.num_layers):
            old_prefix = f'layers.{i}'
            new_prefix = f'transformer_blocks.{i}'

            # Attention mappings
            if f'{old_prefix}.attention.q_proj.weight' in old_state:
                new_state[f'{new_prefix}.attention.q_proj.weight'] = old_state[f'{old_prefix}.attention.q_proj.weight']
            if f'{old_prefix}.attention.k_proj.weight' in old_state:
                new_state[f'{new_prefix}.attention.k_proj.weight'] = old_state[f'{old_prefix}.attention.k_proj.weight']
            if f'{old_prefix}.attention.v_proj.weight' in old_state:
                new_state[f'{new_prefix}.attention.v_proj.weight'] = old_state[f'{old_prefix}.attention.v_proj.weight']
            if f'{old_prefix}.attention.o_proj.weight' in old_state:
                new_state[f'{new_prefix}.attention.w_o.weight'] = old_state[f'{old_prefix}.attention.o_proj.weight']

            # Feed forward mappings
            if f'{old_prefix}.gate_proj.weight' in old_state:
                new_state[f'{new_prefix}.feed_forward.gate_proj.weight'] = old_state[f'{old_prefix}.gate_proj.weight']
            if f'{old_prefix}.up_proj.weight' in old_state:
                new_state[f'{new_prefix}.feed_forward.up_proj.weight'] = old_state[f'{old_prefix}.up_proj.weight']
            if f'{old_prefix}.down_proj.weight' in old_state:
                new_state[f'{new_prefix}.feed_forward.down_proj.weight'] = old_state[f'{old_prefix}.down_proj.weight']

            # Norm mappings (ignore bias for RMSNorm)
            if f'{old_prefix}.attention_norm.weight' in old_state:
                new_state[f'{new_prefix}.norm1.weight'] = old_state[f'{old_prefix}.attention_norm.weight']
            if f'{old_prefix}.ffn_norm.weight' in old_state:
                new_state[f'{new_prefix}.norm2.weight'] = old_state[f'{old_prefix}.ffn_norm.weight']

        # Map final norm
        if 'norm.weight' in old_state:
            new_state['norm.weight'] = old_state['norm.weight']

        # Map LM head (usually tied to embeddings)
        if 'lm_head.weight' in old_state:
            new_state['lm_head.weight'] = old_state['lm_head.weight']
        elif 'token_embeddings.weight' in old_state:
            # Use tied embeddings
            new_state['lm_head.weight'] = old_state['token_embeddings.weight']

        print(f"✅ Mapped {len(old_state)} → {len(new_state)} parameters")
        return new_state

    def tokenize_text(self, text):
        """Proper tokenization using SmolLM tokenizer (same as training!)"""
        tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        return tokens

    def detokenize_text(self, tokens):
        """Convert tokens back to text using proper tokenizer"""
        if tokens.dim() > 1:
            tokens = tokens.squeeze(0)

        # Convert to list if it's a tensor
        if hasattr(tokens, 'tolist'):
            token_ids = tokens.tolist()
        else:
            token_ids = tokens

        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text
    
    def generate_response(self, prompt, max_length=50, temperature=0.8, top_k=50):
        """Generate a response to the given prompt using proper tokenization"""
        print(f"🎯 Prompt: '{prompt}'")

        # Tokenize input with CORRECT tokenizer
        input_tokens = self.tokenize_text(prompt)
        generated_tokens = input_tokens.clone()

        print(f"🔄 Generating response...")
        print(f"🔧 Input tokens: {input_tokens.shape} - {input_tokens.tolist()}")

        with torch.no_grad():
            for i in range(max_length):
                # Get model predictions - handle both output formats
                outputs = self.model(generated_tokens)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs

                # Get logits for next token (last position)
                next_token_logits = logits[0, -1, :] / temperature

                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                # Add to generated sequence
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)

                # Check for EOS token or natural stopping
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Prevent infinite loops
                if generated_tokens.size(1) > 200:
                    break

        # Convert back to text with CORRECT tokenizer
        full_response = self.detokenize_text(generated_tokens)

        # Extract only the generated part
        response_only = full_response[len(prompt):].strip()

        print(f"🎯 Generated tokens: {generated_tokens.shape}")
        print(f"📝 Full response: '{full_response}'")

        return response_only
    
    def run_conversation_tests(self):
        """Run a series of simple conversation tests"""
        if not self.load_model():
            return False
        
        print("\n" + "="*60)
        print("🗣️  CONVERSATION TESTS")
        print("="*60)
        
        # Test prompts - simple and basic
        test_prompts = [
            "Hello",
            "How are you",
            "What is your name",
            "Tell me about",
            "The weather is",
            "I like to",
            "Can you help",
            "Today is a good day",
            "The cat is",
            "Programming is"
        ]
        
        results = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}/{len(test_prompts)} ---")
            
            try:
                response = self.generate_response(
                    prompt, 
                    max_length=50,
                    temperature=0.7,
                    top_k=40
                )
                
                print(f"✅ Response: '{response}'")
                
                # Simple quality check
                is_reasonable = (
                    len(response) > 0 and 
                    len(response) < 200 and
                    not response.isspace()
                )
                
                results.append({
                    'prompt': prompt,
                    'response': response,
                    'reasonable': is_reasonable
                })
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    'prompt': prompt,
                    'response': f"ERROR: {e}",
                    'reasonable': False
                })
        
        # Summary
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        reasonable_count = sum(1 for r in results if r['reasonable'])
        total_count = len(results)
        
        print(f"✅ Reasonable responses: {reasonable_count}/{total_count}")
        print(f"📈 Success rate: {reasonable_count/total_count*100:.1f}%")
        
        if reasonable_count > total_count * 0.5:
            print("🎉 Model seems to have learned basic language patterns!")
        else:
            print("🤔 Model might need more training or different approach")
        
        return results

def main():
    """Main function to run the conversation test"""
    print("🚀 Starting Simple Conversation Test")
    print("=" * 50)
    
    # Check if model exists
    model_path = "trained_models/modern_llm_926m_20250827_080157"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please make sure you have trained the model first!")
        return
    
    # Run tests
    tester = SimpleConversationTester(model_path)
    results = tester.run_conversation_tests()
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    main()
