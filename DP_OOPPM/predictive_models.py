import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM_Model(nn.Module):
    def __init__(self, 
                 vocab_sizes, 
                 embed_sizes,
                 num_numerical_features,
                 max_length, 
                 dropout = 0.2, 
                 lstm_size = 32, 
                 num_lstm = 1,
                 bidirectional = False
                 ):
        """
    LSTM_Model is a neural network model for processing sequences with both categorical and numerical features.
    It uses an LSTM layer to capture temporal dependencies and a dense layer for final output.

    Attributes:
        vocab_sizes (list): Sizes of the vocabularies for each categorical feature.
        embed_sizes (list): Embedding dimensions for each categorical feature.
        num_numerical_features (int): Number of numerical features.
        max_length (int): Maximum sequence length.
        dropout (float): Dropout rate for regularization.
        lstm_size (int): Number of features in the hidden state of the LSTM.
        num_lstm (int): Number of LSTM layers.
        bidirectional (bool): If True, use a bidirectional LSTM.

    Methods:
        forward(inputs, sequence_lengths):
            Forward pass through the model. Takes inputs and sequence lengths, processes them through embedding,
            LSTM, and dense layers, and returns the final output.
    """
        
        super(LSTM_Model, self).__init__()
        
        self.max_length = max_length
        self.lstm_dropout = dropout
        self.lstm_size= lstm_size
        self.num_lstm = num_lstm
        #we assume the input is a list of vocabulary sizes and a list of embedding sizes for each of the categorical embeddings
         #! we assume the input and the lists are in the same order
        self.vocab_sizes = vocab_sizes 
        self.embed_sizes = embed_sizes
        self.dropout = dropout

        #! binary features are counted as numerical here
        self.num_numerical_features = num_numerical_features
        self.bidirectional = bidirectional

        #we do need the total input size etc.
        self.total_num_features_after_emb = sum(self.embed_sizes) + self.num_numerical_features

        self.emb_layers = nn.ModuleList(
            [nn.Embedding(self.vocab_sizes[i], self.embed_sizes[i]) for i in range(len(vocab_sizes))]
            )
        self.dropout_layer = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(input_size=self.total_num_features_after_emb, hidden_size=self.lstm_size, num_layers=self.num_lstm, batch_first=True, bidirectional=bidirectional)
        #self.bn = nn.BatchNorm1d(self.lstm_size)

        # Dense layer
        # Adjust the input size based on the bidirectionality of the LSTM
        lstm_output_size = self.lstm_size * (2 if self.bidirectional else 1)
        self.dense = nn.Linear(lstm_output_size, 1)

    def forward(self, inputs, sequence_lengths):

        device = inputs.device  # Use the device of `inputs`
 
        #! we assume integer ecoding
        # Assume `inputs` has shape (batch_size, seq_len, total_num_features)
        batch_size, seq_len, _ = inputs.size()

        # Separate categorical and numerical features at each timestep
        categorical_inputs = inputs[:, :, :len(self.vocab_sizes)].long().to(device)
        numerical_inputs = inputs[:, :, len(self.vocab_sizes):].float().to(device)  # Remaining part for numerical features

        # Embed each categorical feature separately for each timestep and concatenate embeddings
        embeddings = [emb_layer(categorical_inputs[:, :, i]) for i, emb_layer in enumerate(self.emb_layers)]
        embedded_categorical = torch.cat(embeddings, dim=-1)  # Concatenate all embeddings for each timestep

        # Concatenate embedded categorical features with numerical inputs
        combined_inputs = torch.cat([embedded_categorical, numerical_inputs], dim=-1)  # Shape (batch_size, seq_len, total_num_features_after_emb)

        #Apply dropout
        x = self.dropout_layer(combined_inputs)

        # Pack the sequence for masking
        # sequence_lengths needs to be on the CPU is due to a specific requirement of PyTorch's pack_padded_sequence function, 
        # which currently only accepts the lengths argument on the CPU
        packed_input = pack_padded_sequence(x, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Pass the combined packed inputs through the LSTM
        packed_output, _ = self.lstm(packed_input)

        # Unpack sequence
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)

        # Take the last output for classification or regression
        #last_outputs = lstm_out[:, -1, :]  # (batch_size, lstm_size) - last timestep's output for each sequence
        #now we use the last non-padded output!
        last_outputs = lstm_out[torch.arange(batch_size), sequence_lengths - 1, :]

        # Apply batch normalization
        #!if we do also need to permute and re permute afterwards .permute(0, 2, 1)
        #lstm_out = self.bn(lstm_out)

        # Apply final dense layer and sigmoid activation
        output = self.dense(last_outputs)
        output = torch.sigmoid(output)

        return output

        
