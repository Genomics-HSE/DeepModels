import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        
        input_size = config["input_size"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        n_heads = config["n_heads"]
        pf_size = config["pf_size"]
        length = config["length"]
        dropout_p = config["dropout"]
        self.device = device
        
        self.pos_embedding = nn.Embedding(length, hidden_size)
        
        self.layers = nn.ModuleList([EncoderLayer(hidden_size,
                                                  n_heads,
                                                  pf_size,
                                                  dropout_p,
                                                  device)
                                     for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout_p)
        
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)
    
    def forward(self, input_tensor):
        # input_tensor = [batch size, seq_len, 1]
        
        batch_size = input_tensor.shape[0]
        src_len = input_tensor.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        # pos = [batch size, src len]
    
        src = self.dropout((input_tensor * self.scale) + self.pos_embedding(pos))
        
        # src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src)
        
        # src = [batch size, src len, hid dim]
        return src


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]
        
        # self attention
        _src, _ = self.self_attention(src, src, src)
        
        # dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))
        
        # src = [batch size, src len, hid dim]
        
        # positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        # dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(_src))
        
        # src = [batch size, src len, hid dim]
        
        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        # energy = [batch size, n heads, seq len, seq len]
        
        attention = torch.softmax(energy, dim=-1)
        
        # attention = [batch size, n heads, query len, key len]
        
        x = torch.matmul(self.dropout(attention), V)
        
        # x = [batch size, n heads, seq len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        # x = [batch size, seq len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        # x = [batch size, seq len, hid dim]
        
        x = self.fc_o(x)
        
        # x = [batch size, seq len, hid dim]
        
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        # x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        # x = [batch size, seq len, hid dim]
        
        return x


class Decoder(nn.Module):
    def __init__(self, config,
                 device
                 ):
        super().__init__()
        
        output_size = config["output_size"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        n_heads = config["n_heads"]
        pf_size = config["pf_size"]
        dropout_p = config["dropout"]
        length = config["length"]
        self.device = device
        
        self.pos_embedding = nn.Embedding(length, hidden_size)
        
        self.layers = nn.ModuleList([DecoderLayer(hidden_size,
                                                  n_heads,
                                                  pf_size,
                                                  dropout_p,
                                                  device)
                                     for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout_p)
        
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)
    
    def forward(self, trg, enc_src):
        # trg = [batch size, trg len, 1]
        # enc_src = [batch size, src len, hid dim]
        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        # pos = [batch size, trg len]
        
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        
        # trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src)
        
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        # output = [batch size, trg len, output dim]
        
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, trg, enc_src):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg)

        # dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src)

        # dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        
        return trg, attention


class Model(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        
        self.encoder = Encoder(config["encoder"], device)
        self.decoder = Decoder(config["decoder"], device)
        self.device = device
    
    def forward(self, input_tensor, target_tensor):
        # src = [seq_len, batch size, src len]
        # trg = [seq_len, batch size, trg len]
        input_tensor = input_tensor.permute(1, 0, 2)
        target_tensor = target_tensor.permute(1, 0, 2)
        
        enc_src = self.encoder(input_tensor)

        # enc_src = [batch size, src len, hid dim]
        
        output, attention = self.decoder(target_tensor, enc_src)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]
        
        return output
