import torch
import torch.nn as nn

class CharacterCNN(nn.Module):
    def __init__(self, vocab_size=261, embedding_dim=50, num_filters=2048, projection_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters//7, w) for w in range(1,8)])
        self.highway = nn.Linear(num_filters, projection_dim)
    
    def forward(self, char_ids):
        x = self.embed(char_ids)
        conv_out = torch.cat([torch.max(c(x.transpose(1,2)), 2)[0] for c in self.convs], 1)
        return torch.relu(self.highway(conv_out))

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, projection_size, num_layers=2):
        super().__init__()
        self.forward_lstms = nn.ModuleList([nn.LSTMCell(input_size if i==0 else projection_size, hidden_size) for i in range(num_layers)])
        self.backward_lstms = nn.ModuleList([nn.LSTMCell(input_size if i==0 else projection_size, hidden_size) for i in range(num_layers)])
        self.proj = nn.Linear(hidden_size, projection_size)
    
    def forward(self, x):
        return x

class ELMo(nn.Module):
    def __init__(self, vocab_size, char_embed_dim, num_filters, projection_dim, hidden_size, num_layers):
        super().__init__()
        self.char_cnn = CharacterCNN(vocab_size, char_embed_dim, num_filters, projection_dim)
        self.bilstm = BiLSTM(projection_dim, hidden_size, projection_dim, num_layers)
    
    def forward(self, char_ids):
        x = self.char_cnn(char_ids)
        return self.bilstm(x)
