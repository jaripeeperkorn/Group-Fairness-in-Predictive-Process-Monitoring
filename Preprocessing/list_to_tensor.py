import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np





def nested_list_to_tensor(nested_list, dtype=torch.float32):
    """
    Convert a nested list of sequences to a padded PyTorch tensor for LSTM input.
    Also returns the sequence lengths for each sequence in the batch. Needed for masking afterwards.
    """
    tensor_sequences = [torch.tensor(seq, dtype=dtype) for seq in nested_list]
    padded_tensor = pad_sequence(tensor_sequences, batch_first=True, padding_value=0.0)
    sequence_lengths = torch.tensor([len(seq) for seq in nested_list], dtype=torch.long)  # Sequence lengths

    return padded_tensor, sequence_lengths

def list_to_tensor(lst):
    tensor = torch.from_numpy(np.array(lst))
    return tensor.float()


def nested_list_to_tensor_left_padding(nested_list, dtype=torch.float32):
    """
    Convert a nested list of sequences to a left-padded PyTorch tensor for LSTM input.
    Also returns the sequence lengths for each sequence in the batch. Needed for masking afterwards.
    
    Parameters:
    - nested_list (list of lists): The nested list containing sequences of feature vectors.
    - dtype (torch.dtype): The data type for the tensor (e.g., torch.float32).
    
    Returns:
    - tensor (torch.Tensor): A padded tensor of shape (batch_size, seq_length, feature_dim).
    - seq_lengths (torch.Tensor): A tensor containing the sequence lengths for each sequence in the batch.
    """
    max_len = max(len(seq) for seq in nested_list)  # Find the length of the longest sequence
    padded_sequences = []
    seq_lengths = []

    # Assuming each sequence is a list of feature vectors, we pad the sequence's length dimension (axis 0)
    for seq in nested_list:
        left_padding = torch.zeros((max_len - len(seq), len(seq[0])), dtype=dtype)  # Left padding with zeros
        padded_seq = torch.cat([left_padding, torch.tensor(seq, dtype=dtype)], dim=0)  # Pad along the sequence dimension
        padded_sequences.append(padded_seq)
        seq_lengths.append(len(seq))  # Store the original sequence length

    padded_tensor = torch.stack(padded_sequences)  # Shape (batch_size, max_len, feature_dim)

    # Convert sequence lengths to a tensor
    seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long)

    return padded_tensor, seq_lengths_tensor