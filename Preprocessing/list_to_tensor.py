import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def nested_list_to_tensor(nested_list, dtype=torch.float32):
    """
    Convert a nested list of sequences to a padded PyTorch tensor for LSTM input.
    Also returns the sequence lengths for each sequence in the batch. Needed fpr masking afterwards.
    """
    tensor_sequences = [torch.tensor(seq, dtype=dtype) for seq in nested_list]
    padded_tensor = pad_sequence(tensor_sequences, batch_first=True, padding_value=0.0)
    sequence_lengths = torch.tensor([len(seq) for seq in nested_list], dtype=torch.long)  # Sequence lengths

    return padded_tensor, sequence_lengths

def list_to_tensor(lst):
    tensor = torch.from_numpy(np.array(lst))
    return tensor.float()


