import torch.nn as nn
import torch

#todo do we need batchnorm?

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
        self.dense = nn.Linear(self.lstm_size, 1)

    def forward(self, inputs):
 
        #! we assume integer ecoding
        # Assume `inputs` has shape (batch_size, seq_len, total_num_features)
        batch_size, seq_len, _ = inputs.size()

        # Separate categorical and numerical features at each timestep
        categorical_inputs = inputs[:, :, :len(self.vocab_sizes)].long()
        numerical_inputs = inputs[:, :, len(self.vocab_sizes):].long()  # Remaining part for numerical features

        # Embed each categorical feature separately for each timestep and concatenate embeddings
        embeddings = [emb_layer(categorical_inputs[:, :, i]) for i, emb_layer in enumerate(self.emb_layers)]
        embedded_categorical = torch.cat(embeddings, dim=-1)  # Concatenate all embeddings for each timestep

        # Concatenate embedded categorical features with numerical inputs
        combined_inputs = torch.cat([embedded_categorical, numerical_inputs], dim=-1)  # Shape (batch_size, seq_len, total_num_features_after_emb)

        #Apply dropout
        x = self.dropout_layer(combined_inputs)

        # Pass the combined inputs through the LSTM
        lstm_out, _ = self.lstm(x)  # Output shape (batch_size, seq_len, lstm_size)

        # Take the last output for classification or regression
        lstm_out = lstm_out[:, -1, :]  # (batch_size, lstm_size) - last timestep's output for each sequence

        # Apply batch normalization
        #!if we do also need to permute and re permute afterwards .permute(0, 2, 1)
        #lstm_out = self.bn(lstm_out)

        # Final dense layer to get output
        output = self.dense(lstm_out)
        output = torch.sigmoid(output)

        return output

        
