# Our sample training data
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

print("Training Corpus:")
for doc in corpus:
    print(doc)

class SimpleTokenizer:
    """
    Ein einfacher Tokenizer, der Text in Tokens aufteilt und ein Vokabular erstellt.
    """
    
    def __init__(self):
        self.vocab = {}  # Token -> ID mapping
        self.id_to_token = {}  # ID -> Token mapping
        self.next_id = 0
        
        # Spezielle Tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"  # Beginning of sequence
        self.eos_token = "<EOS>"  # End of sequence
        
        # Spezielle Tokens zum Vokabular hinzufügen
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Fügt spezielle Tokens zum Vokabular hinzu."""
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token in special_tokens:
            self._add_token(token)
    
    def _add_token(self, token):
        """Fügt ein Token zum Vokabular hinzu, falls es noch nicht existiert."""
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
    
    def tokenize(self, text):
        """
        Tokenisiert einen Text in einzelne Wörter.
        Einfache Implementierung: Split bei Leerzeichen und entferne Interpunktion.
        """
        # Einfache Tokenisierung: Kleinbuchstaben und Split bei Leerzeichen
        text = text.lower()
        # Entferne Interpunktion (einfache Version)
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        return tokens
    
    def build_vocab(self, corpus):
        """Erstellt das Vokabular aus dem Trainingskorpus."""
        print("\nBuilding vocabulary...")
        
        for document in corpus:
            tokens = self.tokenize(document)
            for token in tokens:
                self._add_token(token)
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Vocabulary: {list(self.vocab.keys())}")
    
    def encode(self, text):
        """Konvertiert Text in Token-IDs."""
        tokens = self.tokenize(text)
        token_ids = []
        
        # BOS Token hinzufügen
        token_ids.append(self.vocab[self.bos_token])
        
        # Text-Tokens konvertieren
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab[self.unk_token])
        
        # EOS Token hinzufügen
        token_ids.append(self.vocab[self.eos_token])
        
        return token_ids
    
    def decode(self, token_ids):
        """Konvertiert Token-IDs zurück in Text."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                # Spezielle Tokens überspringen (außer für Debug-Zwecke)
                if token not in [self.bos_token, self.eos_token, self.pad_token]:
                    tokens.append(token)
        
        return " ".join(tokens)
    
    def get_vocab_size(self):
        """Gibt die Größe des Vokabulars zurück."""
        return len(self.vocab)
    
    def save_vocab(self, filepath):
        """Speichert das Vokabular in eine Datei."""
        import json
        vocab_data = {
            'vocab': self.vocab,
            'id_to_token': self.id_to_token,
            'next_id': self.next_id
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocab(self, filepath):
        """Lädt das Vokabular aus einer Datei."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab = vocab_data['vocab']
        self.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
        self.next_id = vocab_data['next_id']
        print(f"Vocabulary loaded from {filepath}")


# Demo des Tokenizers
if __name__ == "__main__":
    print("=" * 50)
    print("SIMPLE TOKENIZER DEMO")
    print("=" * 50)
    
    # Tokenizer erstellen
    tokenizer = SimpleTokenizer()
    
    # Vokabular aus Korpus erstellen
    tokenizer.build_vocab(corpus)
    
    print("\n" + "=" * 50)
    print("TOKENIZATION EXAMPLES")
    print("=" * 50)
    
    # Beispiele für Tokenisierung
    test_texts = [
        "This is the first document.",
        "This is a new sentence.",
        "Unknown words will be handled."
    ]
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        tokens = tokenizer.tokenize(text)
        print(f"Tokens: {tokens}")
        
        encoded = tokenizer.encode(text)
        print(f"Encoded: {encoded}")
        
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")
    
    # Vokabular speichern
    tokenizer.save_vocab("vocab.json")
