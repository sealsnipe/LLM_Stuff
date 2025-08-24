"""
OpenAI tiktoken vs Hugging Face Tokenizers Comparison
Comparing performance and behavior of different tokenizers
"""

import tiktoken
import time
from tokenizers import Tokenizer

# %%
# === Load different tokenizers ===
print("=== Loading Tokenizers ===")

# OpenAI tiktoken tokenizers
tiktoken_encodings = {
    "gpt-4": tiktoken.encoding_for_model("gpt-4"),
    "gpt-3.5-turbo": tiktoken.encoding_for_model("gpt-3.5-turbo"), 
    "text-davinci-003": tiktoken.encoding_for_model("text-davinci-003"),
    "cl100k_base": tiktoken.get_encoding("cl100k_base"),  # GPT-4 base
    "p50k_base": tiktoken.get_encoding("p50k_base"),      # GPT-3 base
}

# Hugging Face tokenizers
hf_tokenizers = {}
try:
    hf_tokenizers["gpt2"] = Tokenizer.from_pretrained("gpt2")
    hf_tokenizers["bert"] = Tokenizer.from_pretrained("bert-base-uncased")
    print("✓ All tokenizers loaded successfully")
except Exception as e:
    print(f"Error loading HF tokenizers: {e}")

# %%
# === Compare vocabulary sizes ===
print("\n=== Vocabulary Sizes ===")

print("OpenAI tiktoken:")
for name, enc in tiktoken_encodings.items():
    print(f"  {name}: {enc.n_vocab:,} tokens")

print("\nHugging Face:")
for name, tok in hf_tokenizers.items():
    print(f"  {name}: {tok.get_vocab_size():,} tokens")

# %%
# === Test texts for comparison ===
test_texts = [
    "Hello, world!",
    "This is a test of tokenization efficiency.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning and artificial intelligence are revolutionizing technology.",
    "🤖 Python programming with emojis and special characters: @#$%^&*()",
    "Supercalifragilisticexpialidocious is a very long word from Mary Poppins.",
    "Tokenization is the process of breaking down text into smaller units called tokens.",
    """This is a longer text that spans multiple sentences. It contains various punctuation marks, numbers like 123 and 456, and different types of content. We want to see how different tokenizers handle longer passages of text and whether there are significant differences in their tokenization strategies."""
]

# %%
# === Compare tokenization results ===
print("\n=== Tokenization Comparison ===")

def compare_tokenization(text, max_length=100):
    """Compare how different tokenizers handle the same text"""
    print(f"\nText: '{text[:max_length]}{'...' if len(text) > max_length else ''}'")
    print("-" * 80)
    
    # OpenAI tiktoken
    for name, enc in tiktoken_encodings.items():
        tokens = enc.encode(text)
        decoded = enc.decode(tokens)
        print(f"{name:15} | {len(tokens):3} tokens | Perfect roundtrip: {text == decoded}")
    
    # Hugging Face
    for name, tok in hf_tokenizers.items():
        encoding = tok.encode(text)
        tokens = encoding.tokens
        decoded = tok.decode(encoding.ids)
        print(f"{name:15} | {len(tokens):3} tokens | Perfect roundtrip: {text == decoded}")

# Test first few texts
for text in test_texts[:4]:
    compare_tokenization(text)

# %%
# === Performance Comparison ===
print("\n=== Performance Comparison ===")

# Use a longer text for performance testing
long_text = " ".join(test_texts) * 100  # Repeat to make it longer
print(f"Testing with text of {len(long_text):,} characters")

def time_tokenizer(name, tokenizer, text, is_tiktoken=True):
    """Time how long tokenization takes"""
    start_time = time.time()
    
    if is_tiktoken:
        for _ in range(10):  # Run multiple times for better measurement
            tokens = tokenizer.encode(text)
    else:
        for _ in range(10):
            encoding = tokenizer.encode(text)
            tokens = encoding.ids
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 10
    
    return avg_time, len(tokens)

print("\nTokenization speed (average over 10 runs):")
print("Tokenizer       | Time (ms) | Tokens | Tokens/sec")
print("-" * 50)

# Test tiktoken
for name, enc in tiktoken_encodings.items():
    try:
        avg_time, token_count = time_tokenizer(name, enc, long_text, True)
        tokens_per_sec = token_count / avg_time if avg_time > 0 else 0
        print(f"{name:15} | {avg_time*1000:8.2f} | {token_count:6} | {tokens_per_sec:10.0f}")
    except Exception as e:
        print(f"{name:15} | Error: {e}")

# Test Hugging Face
for name, tok in hf_tokenizers.items():
    try:
        avg_time, token_count = time_tokenizer(name, tok, long_text, False)
        tokens_per_sec = token_count / avg_time if avg_time > 0 else 0
        print(f"{name:15} | {avg_time*1000:8.2f} | {token_count:6} | {tokens_per_sec:10.0f}")
    except Exception as e:
        print(f"{name:15} | Error: {e}")

# %%
# === Detailed token analysis ===
print("\n=== Detailed Token Analysis ===")

sample_text = "The tokenization process breaks text into subword units."

print(f"Analyzing: '{sample_text}'")
print("\nDetailed tokenization:")

# Show actual tokens for comparison
if "gpt-4" in tiktoken_encodings and "gpt2" in hf_tokenizers:
    # GPT-4 (tiktoken)
    gpt4_tokens = tiktoken_encodings["gpt-4"].encode(sample_text)
    gpt4_token_strings = [tiktoken_encodings["gpt-4"].decode([token]) for token in gpt4_tokens]
    print(f"\nGPT-4 (tiktoken): {len(gpt4_tokens)} tokens")
    print(f"Tokens: {gpt4_token_strings}")
    
    # GPT-2 (Hugging Face)
    gpt2_encoding = hf_tokenizers["gpt2"].encode(sample_text)
    gpt2_tokens = gpt2_encoding.tokens
    print(f"\nGPT-2 (HF): {len(gpt2_tokens)} tokens")
    print(f"Tokens: {gpt2_tokens}")
    
    # BERT (Hugging Face)
    bert_encoding = hf_tokenizers["bert"].encode(sample_text)
    bert_tokens = bert_encoding.tokens
    print(f"\nBERT (HF): {len(bert_tokens)} tokens")
    print(f"Tokens: {bert_tokens}")

# %%
# === Special characters and edge cases ===
print("\n=== Special Characters & Edge Cases ===")

edge_cases = [
    "🤖🚀🎉",  # Multiple emojis
    "café naïve résumé",  # Accented characters
    "Hello\n\nWorld\t\tTest",  # Whitespace variations
    "123,456.789",  # Numbers with punctuation
    "user@example.com",  # Email
    "https://www.example.com/path?param=value",  # URL
    "C++, JavaScript, and Python",  # Programming languages
    "",  # Empty string
    " ",  # Single space
    "a",  # Single character
]

for text in edge_cases:
    if text.strip():  # Skip empty strings for display
        print(f"\nText: '{text}'")
        
        # Compare GPT-4 vs GPT-2
        if "gpt-4" in tiktoken_encodings and "gpt2" in hf_tokenizers:
            gpt4_count = len(tiktoken_encodings["gpt-4"].encode(text))
            gpt2_count = len(hf_tokenizers["gpt2"].encode(text).tokens)
            print(f"  GPT-4: {gpt4_count} tokens | GPT-2: {gpt2_count} tokens")

print("\n=== Comparison Complete ===")
