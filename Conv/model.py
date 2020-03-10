import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        kernel_size = config["kernel_size"]
        emb_dim = config["emb_dim"]
        hid_dim = config["hid_dim"]
        length = config["length"]
        scale = config["scale"]
        num_layers = config["num_layers"]
        dropout = config["dropout"]
        
        self.device = device
        assert kernel_size % 2 == 1, "Kernel must be odd!"
        self.scale = torch.sqrt(torch.FloatTensor([scale])).to(device)
        
        self.pos_embedding = nn.Embedding(length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2)
                                    for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_tensor):
        # input_tensor shape [batch_size, seq_len, 1]
        
        batch_size = input_tensor.shape[0]
        src_len = input_tensor.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        pos_embedded = self.pos_embedding(pos)
        
        embedded = self.dropout(input_tensor + pos_embedded)
        # embedded = [batch size, src len, emb dim]
        
        # pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(embedded)
        
        # conv_input = [batch size, src len, hid dim]
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        
        # conv_input = [batch size, hid dim, src len]
        
        # begin convolutional blocks
        
        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            
            # conved = [batch size, 2 * hid dim, src len]
            
            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            
            # conved = [batch size, hid dim, src len]
            
            # apply residual connection
            conved = (conved + conv_input) * self.scale
            
            # conved = [batch size, hid dim, src len]
            
            # set conv_input to conved for next loop iteration
            conv_input = conved
        
        # end convolutional blocks
        
        # permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))
        
        # conved = [batch size, src len, emb dim]
        
        # elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale
        
        # combined = [batch size, src len, emb dim]
        
        return conved, combined


class Decoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        
        emb_dim = config["emb_dim"]
        hid_dim = config["hid_dim"]
        output_dim = config["output_dim"]
        length = config["length"]
        kernel_size = config["kernel_size"]
        num_layers = config["num_layers"]
        dropout = config["dropout"]
        scale = config["scale"]
        self.trg_pad_idx = config["trg_pad_idx"]
        self.device = device
        
        self.scale = torch.sqrt(torch.FloatTensor([scale])).to(device)
        self.kernel_size = kernel_size
        self.pos_embedding = nn.Embedding(length, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)
        
        self.fc_out = nn.Linear(emb_dim, output_dim)
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2*hid_dim,
                                              kernel_size=kernel_size)
                                    for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)
    
    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        # embedded = [batch size, trg len, emb dim]
        # conved = [batch size, hid dim, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]
        
        # permute and convert back to emb dim
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
        
        # conved_emb = [batch size, trg len, emb dim]
        
        combined = (conved_emb + embedded) * self.scale
        
        # combined = [batch size, trg len, emb dim]
        
        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        
        # energy = [batch size, trg len, src len]
        
        attention = F.softmax(energy, dim=2)
        
        # attention = [batch size, trg len, src len]
        
        attended_encoding = torch.matmul(attention, encoder_combined)
        
        # attended_encoding = [batch size, trg len, emd dim]
        
        # convert from emb dim -> hid dim
        attended_encoding = self.attn_emb2hid(attended_encoding)
        
        # attended_encoding = [batch size, trg len, hid dim]
        
        # apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale
        
        # attended_combined = [batch size, hid dim, trg len]
        
        return attention, attended_combined
    
    def forward(self, target_tensor, encoder_conved, encoder_combined):
        # target_tensor = [seq_len, batch size, 1]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]
        
        batch_size = target_tensor.shape[0]
        trg_len = target_tensor.shape[1]
        
        # create position tensor
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        # pos = [batch size, trg len]
        
        # embed tokens and positions
        pos_embedded = self.pos_embedding(pos)
        # pos_embedded = [batch size, trg len, emb dim]
        
        # combine embeddings by elementwise summing
        embedded = self.dropout(target_tensor + pos_embedded)
        
        # embedded = [batch size, trg len, emb dim]
        
        # pass embedded through linear layer to go through emb dim -> hid dim
        conv_input = self.emb2hid(embedded)
        
        # conv_input = [batch size, trg len, hid dim]
        
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        
        # conv_input = [batch size, hid dim, trg len]
        
        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]
        
        for i, conv in enumerate(self.convs):
            # apply dropout
            conv_input = self.dropout(conv_input)
            
            # need to pad so decoder can't "cheat"
            padding = torch.zeros(batch_size,
                                  hid_dim,
                                  self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)
            
            padded_conv_input = torch.cat((padding, conv_input), dim=2)
            
            # padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]
            
            # pass through convolutional layer
            conved = conv(padded_conv_input)
            
            # conved = [batch size, 2 * hid dim, trg len]
            
            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            
            # conved = [batch size, hid dim, trg len]
            
            # calculate attention
            attention, conved = self.calculate_attention(embedded,
                                                         conved,
                                                         encoder_conved,
                                                         encoder_combined)
            
            # attention = [batch size, trg len, src len]
            
            # apply residual connection
            conved = (conved + conv_input) * self.scale
            
            # conved = [batch size, hid dim, trg len]
            
            # set conv_input to conved for next loop iteration
            conv_input = conved
        
        conved = self.hid2emb(conved.permute(0, 2, 1))
        
        # conved = [batch size, trg len, emb dim]
        
        output = self.fc_out(self.dropout(conved))
        
        # output = [batch size, trg len, output dim]
        
        return output, attention


class Model(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        
        self.encoder = Encoder(config["encoder"], device)
        self.decoder = Decoder(config["decoder"], device)
        self.device = device
    
    def forward(self, input_tensor, target_tensor):
        # input_tensor = [seq_len, batch size, 1]
        # target_tensor = [seq_len, batch size, 1]
        input_tensor = input_tensor.permute(1, 0, 2)
        target_tensor = target_tensor.permute(1, 0, 2)
        # calculate z^u (encoder_conved) and (z^u + e) (encoder_combined)
        # encoder_conved is output from final encoder conv. block
        # encoder_combined is encoder_conved plus (elementwise) src embedding plus
        #  positional embeddings
        encoder_conved, encoder_combined = self.encoder(input_tensor)

        # encoder_conved = [batch size, src len, emb dim]
        # encoder_combined = [batch size, src len, emb dim]

        # calculate predictions of next words
        # output is a batch of predictions for each word in the trg sentence
        # attention a batch of attention scores across the src sentence for
        #  each word in the trg sentence
        output, attention = self.decoder(target_tensor, encoder_conved, encoder_combined)

        # output = [batch size, trg len - 1, output dim] CHECK HERE
        # attention = [batch size, trg len - 1, src len] CHECK HERE
        
        return output.permute(1, 0, 2)
