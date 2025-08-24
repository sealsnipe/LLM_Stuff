"""
Hugging Face Tokenizers Examples
Demonstrating how to load pretrained tokenizers and tokenize/detokenize text
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# %%
# === Example 1: Load a pretrained GPT-2 tokenizer ===
print("=== Loading GPT-2 Tokenizer ===")

try:
    # Load GPT-2 tokenizer from Hugging Face Hub
    tokenizer_gpt2 = Tokenizer.from_pretrained("gpt2")
    print("✓ GPT-2 tokenizer loaded successfully")
    
    # Get some basic info
    vocab_size = tokenizer_gpt2.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
except Exception as e:
    print(f"Error loading GPT-2 tokenizer: {e}")
    print("Note: This requires internet connection to download from Hugging Face Hub")

# %%
# === Example 2: Tokenize and Detokenize Text ===
print("\n=== Tokenization Examples ===")

# Sample texts to tokenize
texts = [
    "Hello, world!",
    "This is a simple example of tokenization.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning and artificial intelligence are fascinating!",
    "🤖 Emojis and special characters: @#$%^&*()",
]

if 'tokenizer_gpt2' in locals():
    for i, text in enumerate(texts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Original text: '{text}'")
        
        # Tokenize (encode)
        encoding = tokenizer_gpt2.encode(text)
        tokens = encoding.tokens
        token_ids = encoding.ids
        
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Number of tokens: {len(tokens)}")
        
        # Detokenize (decode)
        decoded_text = tokenizer_gpt2.decode(token_ids)
        print(f"Decoded text: '{decoded_text}'")
        
        # Check if roundtrip is perfect
        if text == decoded_text:
            print("✓ Perfect roundtrip!")
        else:
            print("⚠ Roundtrip differs (this can be normal for some tokenizers)")

# %%
# === Example 3: Analyze tokenization patterns ===
print("\n=== Tokenization Analysis ===")

if 'tokenizer_gpt2' in locals():
    # Analyze how different words are tokenized
    test_words = [
        "hello",
        "Hello", 
        "HELLO",
        "tokenization",
        "subword",
        "artificial",
        "intelligence",
        "programming",
        "python",
        "machine",
        "learning"
    ]
    
    print("Word -> Tokens analysis:")
    for word in test_words:
        encoding = tokenizer_gpt2.encode(word)
        tokens = encoding.tokens
        print(f"'{word}' -> {tokens} ({len(tokens)} tokens)")

# %%
# === Example 4: Try different pretrained tokenizers ===
print("\n=== Trying Different Tokenizers ===")

tokenizer_names = [
    "bert-base-uncased",
    "roberta-base", 
    "distilbert-base-uncased"
]

sample_text = "Hello, how are you doing today?"

for name in tokenizer_names:
    try:
        print(f"\n--- {name} ---")
        tokenizer = Tokenizer.from_pretrained(name)
        encoding = tokenizer.encode(sample_text)
        
        print(f"Text: '{sample_text}'")
        print(f"Tokens: {encoding.tokens}")
        print(f"Token count: {len(encoding.tokens)}")
        print(f"Vocab size: {tokenizer.get_vocab_size()}")
        
    except Exception as e:
        print(f"Could not load {name}: {e}")

# %%
# === Example 5: Batch tokenization ===
print("\n=== Batch Tokenization ===")

if 'tokenizer_gpt2' in locals():
    batch_texts = [
        "First sentence.",
        "Second sentence is longer.",
        "The third sentence contains more words and is even longer than the previous ones."
    ]
    
    print("Batch tokenization:")
    encodings = tokenizer_gpt2.encode_batch(batch_texts)
    
    for i, (text, encoding) in enumerate(zip(batch_texts, encodings)):
        print(f"\nText {i+1}: '{text}'")
        print(f"Tokens: {encoding.tokens}")
        print(f"Length: {len(encoding.tokens)} tokens")

# %%
# === Example 6: Special tokens and vocabulary inspection ===
print("\n=== Vocabulary Inspection ===")

if 'tokenizer_gpt2' in locals():
    # Get vocabulary as dictionary
    vocab = tokenizer_gpt2.get_vocab()
    
    print(f"Total vocabulary size: {len(vocab)}")
    
    # Show some example tokens
    print("\nFirst 20 tokens in vocabulary:")
    for i, (token, token_id) in enumerate(list(vocab.items())[:20]):
        print(f"  {token_id}: '{token}'")
    
    # Look for special tokens
    special_tokens = []
    for token, token_id in vocab.items():
        if token.startswith('<') and token.endswith('>'):
            special_tokens.append((token, token_id))
    
    if special_tokens:
        print(f"\nSpecial tokens found: {len(special_tokens)}")
        for token, token_id in special_tokens[:10]:  # Show first 10
            print(f"  {token_id}: '{token}'")
    else:
        print("\nNo obvious special tokens found (format: <token>)")

print("\n=== Examples Complete ===")
