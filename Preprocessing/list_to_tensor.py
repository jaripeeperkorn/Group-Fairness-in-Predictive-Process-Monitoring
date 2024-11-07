import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

def nested_list_to_tensor(nested_list, dtype=torch.float32):
    """
    Convert a nested list of sequences to a padded PyTorch tensor for LSTM input.
    
    Parameters:
    - nested_list (list of lists): The nested list containing sequences.
    - dtype (torch.dtype): The data type for the tensor (e.g., torch.float32).
    
    Returns:
    - tensor (torch.Tensor): A padded tensor of shape (batch_size, seq_length, feature_dim).
    """
    # Convert each inner list to a tensor
    tensor_sequences = [torch.tensor(seq, dtype=dtype) for seq in nested_list]
    
    # Pad sequences to the length of the longest sequence
    padded_tensor = pad_sequence(tensor_sequences, batch_first=True, padding_value=0.0)
    
    return padded_tensor

def list_to_tensor(lst):
    tensor = torch.from_numpy(np.array(lst))
    return tensor.float()


