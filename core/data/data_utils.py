"""
Data Utilities Module

Contains utility functions for data processing:
- Sequence packing functions
- Attention mask creation
- Data preprocessing utilities
"""

import torch
from typing import List, Dict, Tuple
from config import training_config


def create_packed_sequences(sequences: List[torch.Tensor], max_length: int, eos_token_id: int = 2) -> Dict:
    """
    Packt mehrere Sequenzen in eine einzige Sequenz für effizienteres Training.
    Verwendet intelligente Packing-Strategie für maximale Effizienz.

    Args:
        sequences: Liste von Token-Sequenzen
        max_length: Maximale Länge der gepackten Sequenz
        eos_token_id: EOS Token ID für Trennung zwischen Dokumenten

    Returns:
        Dict mit 'input_ids', 'attention_mask', 'position_ids', 'packing_efficiency'
    """

    # Sortiere Sequenzen nach Länge für besseres Packing
    sequences_with_length = [(seq, len(seq)) for seq in sequences]
    sequences_with_length.sort(key=lambda x: x[1], reverse=True)

    packed_sequences = []
    current_length = 0
    current_sequence = []
    total_original_tokens = sum(len(seq) for seq in sequences)

    for seq, seq_len in sequences_with_length:
        # Prüfe ob Sequenz in aktuelle Packung passt
        if current_length + seq_len + 1 <= max_length:  # +1 für EOS
            current_sequence.extend(seq.tolist())
            current_sequence.append(eos_token_id)
            current_length += seq_len + 1
        else:
            # Aktuelle Packung abschließen
            if current_sequence:
                # Padding hinzufügen
                while len(current_sequence) < max_length:
                    current_sequence.append(eos_token_id)  # Pad with EOS

                packed_sequences.append(torch.tensor(current_sequence, dtype=torch.long))

            # Neue Packung starten
            if seq_len + 1 <= max_length:
                current_sequence = seq.tolist() + [eos_token_id]
                current_length = seq_len + 1
            else:
                # Sequenz ist zu lang, truncate
                truncated = seq[:max_length-1].tolist() + [eos_token_id]
                packed_sequences.append(torch.tensor(truncated, dtype=torch.long))
                current_sequence = []
                current_length = 0

    # Letzte Packung abschließen
    if current_sequence:
        while len(current_sequence) < max_length:
            current_sequence.append(eos_token_id)
        packed_sequences.append(torch.tensor(current_sequence, dtype=torch.long))

    if not packed_sequences:
        return None
    
    # Stack zu Batch
    input_ids = torch.stack(packed_sequences)

    # Berechne Packing-Effizienz
    total_original_tokens = sum(len(seq) for seq in sequences)
    total_packed_tokens = input_ids.numel()
    efficiency = (total_original_tokens / total_packed_tokens) * 100 if total_packed_tokens > 0 else 0

    return {
        'input_ids': input_ids,
        'attention_mask': None,  # Flash-kompatibel: nur kausale Maske
        'position_ids': None,
        'packing_efficiency': efficiency
    }


def create_packed_attention_masks(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """
    Erstellt Attention Masks für Sequence Packing.
    Verhindert Attention zwischen verschiedenen Dokumenten.
    """
    B, T = input_ids.shape
    
    # Finde EOS Token Positionen
    eos_positions = (input_ids == eos_token_id).nonzero(as_tuple=True)
    
    # Erstelle Standard Causal Mask
    causal_mask = torch.ones(T, T, dtype=torch.bool).tril().expand(B, -1, -1)
    
    # Modifiziere Mask für Document Boundaries
    for batch_idx in range(B):
        batch_eos_positions = eos_positions[1][eos_positions[0] == batch_idx]
        
        if len(batch_eos_positions) > 1:
            # Verhindere Attention zwischen Dokumenten
            for i, eos_pos in enumerate(batch_eos_positions[:-1]):
                next_eos_pos = batch_eos_positions[i + 1]
                
                # Tokens nach diesem EOS können nicht auf Tokens vor diesem EOS schauen
                causal_mask[batch_idx, eos_pos+1:next_eos_pos+1, :eos_pos+1] = False
    
    return causal_mask


def calculate_padding_efficiency(batch: torch.Tensor, pad_token_id: int = -100) -> float:
    """
    Berechnet die Padding-Effizienz eines Batches.
    
    Returns:
        Effizienz als Prozentsatz (0-100)
    """
    total_tokens = batch.numel()
    pad_tokens = (batch == pad_token_id).sum().item()
    
    efficiency = ((total_tokens - pad_tokens) / total_tokens) * 100
    return efficiency


def length_bucket_batch(sequences: List[torch.Tensor], bucket_size: int = 8) -> List[List[torch.Tensor]]:
    """
    Gruppiert Sequenzen nach Länge für effizienteres Batching.
    
    Args:
        sequences: Liste von Sequenzen
        bucket_size: Größe der Buckets
    
    Returns:
        Liste von Buckets (Listen von Sequenzen)
    """
    
    # Sortiere nach Länge
    sorted_sequences = sorted(sequences, key=len)
    
    # Gruppiere in Buckets
    buckets = []
    current_bucket = []
    
    for seq in sorted_sequences:
        current_bucket.append(seq)
        
        if len(current_bucket) >= bucket_size:
            buckets.append(current_bucket)
            current_bucket = []
    
    # Letzten Bucket hinzufügen falls nicht leer
    if current_bucket:
        buckets.append(current_bucket)
    
    return buckets


def pad_sequences_to_length(sequences: List[torch.Tensor], target_length: int, pad_token_id: int = 0) -> torch.Tensor:
    """
    Padded Sequenzen auf eine Ziellänge.
    
    Args:
        sequences: Liste von Sequenzen
        target_length: Ziellänge
        pad_token_id: Padding Token ID
    
    Returns:
        Gepadded Tensor [batch_size, target_length]
    """
    
    batch_size = len(sequences)
    padded = torch.full((batch_size, target_length), pad_token_id, dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), target_length)
        padded[i, :seq_len] = seq[:seq_len]
    
    return padded


def create_position_ids(input_ids: torch.Tensor, eos_token_id: int = 2) -> torch.Tensor:
    """
    Erstellt Position IDs für Sequence Packing.
    Position IDs werden bei jedem EOS Token zurückgesetzt.
    """
    B, T = input_ids.shape
    position_ids = torch.zeros_like(input_ids)
    
    for batch_idx in range(B):
        current_pos = 0
        for token_idx in range(T):
            position_ids[batch_idx, token_idx] = current_pos
            current_pos += 1
            
            # Reset position bei EOS Token
            if input_ids[batch_idx, token_idx] == eos_token_id:
                current_pos = 0
    
    return position_ids


def validate_batch_format(batch) -> bool:
    """
    Validiert das Format eines Batches.
    
    Returns:
        True wenn Format korrekt, False sonst
    """
    
    if isinstance(batch, dict):
        # FineWeb format
        required_keys = ['input_ids']
        return all(key in batch for key in required_keys)
    
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        # Synthetic format: [input_ids, labels]
        return all(isinstance(x, torch.Tensor) for x in batch)
    
    else:
        return False
