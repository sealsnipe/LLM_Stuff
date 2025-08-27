#!/usr/bin/env python3
"""
Interactive Chat with our trained 926M LLM
Simple chat interface to have conversations with the model
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
import json

class InteractiveChat:
    def __init__(self, model_path="trained_models/modern_llm_926m_20250827_080157"):
        """Initialize the interactive chat with our trained model"""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.config = None
        
        print("🤖 Interactive Chat with 926M LLM")
        print(f"📁 Model: {model_path}")
        print(f"🔧 Device: {self.device}")
        print("-" * 50)
    
    def load_model(self):
        """Load the trained model"""
        try:
            # Load config
            config_path = os.path.join(self.model_path, "config.json")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            self.config = ModelConfig(**config_dict)
            print(f"✅ Config loaded")
            
            # Initialize and load model
            self.model = ModernLLM(self.config)
            model_file = os.path.join(self.model_path, "model.pt")
            checkpoint = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def simple_tokenize(self, text):
        """Simple character-level tokenization"""
        tokens = [ord(c) % self.config.vocab_size for c in text]
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
    
    def simple_detokenize(self, tokens):
        """Convert tokens back to text"""
        if tokens.dim() > 1:
            tokens = tokens.squeeze(0)
        
        text = ""
        for token in tokens:
            try:
                char = chr(token.item())
                if char.isprintable():
                    text += char
                else:
                    text += "?"
            except:
                text += "?"
        return text
    
    def generate_response(self, prompt, max_length=80, temperature=0.8, top_k=50):
        """Generate a response to the prompt"""
        input_tokens = self.simple_tokenize(prompt)
        generated_tokens = input_tokens.clone()
        
        with torch.no_grad():
            for i in range(max_length):
                logits = self.model(generated_tokens)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
                
                # Stop on sentence endings
                if next_token.item() in [ord('.'), ord('!'), ord('?')] and i > 5:
                    break
                
                # Prevent runaway generation
                if generated_tokens.size(1) > 300:
                    break
        
        full_response = self.simple_detokenize(generated_tokens)
        response_only = full_response[len(prompt):].strip()
        
        return response_only
    
    def chat_loop(self):
        """Main interactive chat loop"""
        if not self.load_model():
            return
        
        print("\n" + "="*60)
        print("💬 INTERACTIVE CHAT MODE")
        print("="*60)
        print("Type your messages and press Enter.")
        print("Commands:")
        print("  'quit' or 'exit' - Exit chat")
        print("  'clear' - Clear conversation")
        print("  'help' - Show this help")
        print("-" * 60)
        
        conversation_history = ""
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    conversation_history = ""
                    print("🧹 Conversation cleared!")
                    continue
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  'quit' or 'exit' - Exit chat")
                    print("  'clear' - Clear conversation")
                    print("  'help' - Show this help")
                    continue
                
                # Generate response
                print("🤖 AI: ", end="", flush=True)
                
                # Use conversation context (simple approach)
                context = conversation_history + f"Human: {user_input}\nAI: "
                
                try:
                    response = self.generate_response(
                        context,
                        max_length=60,
                        temperature=0.7,
                        top_k=40
                    )
                    
                    if response:
                        print(response)
                        # Update conversation history (keep it short)
                        conversation_history += f"Human: {user_input}\nAI: {response}\n"
                        
                        # Keep conversation history manageable
                        if len(conversation_history) > 500:
                            # Keep only last part
                            lines = conversation_history.split('\n')
                            conversation_history = '\n'.join(lines[-10:])
                    else:
                        print("(no response generated)")
                        
                except Exception as e:
                    print(f"(error: {e})")
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def main():
    """Main function"""
    print("🚀 Starting Interactive Chat")
    
    # Check if model exists
    model_path = "trained_models/modern_llm_926m_20250827_080157"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please train the model first!")
        return
    
    # Start chat
    chat = InteractiveChat(model_path)
    chat.chat_loop()

if __name__ == "__main__":
    main()
