import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBiGRU(nn.Module):
    def __init__(self, config, dec_hidden_size):
        super().__init__()
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]  # 1
        self.dropout_p = config["dropout"]
        
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, dec_hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
    
    def forward(self, input):
        outputs, hidden = self.rnn(input)
        
        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]
        
        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer
        
        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN
        
        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        
        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]
        
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        
        self.attn = nn.Linear((self.enc_hidden_size * 2) + dec_hidden_size, self.dec_hidden_size)
        self.v = nn.Linear(self.dec_hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        src_len = encoder_outputs.shape[0]
        
        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # energy = [batch size, src len, dec hid dim]
        
        attention = self.v(energy).squeeze(2)
        
        # attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, config, attention, e_hidden_size):
        super().__init__()
        
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.output_size = config["output_size"]
        self.num_layers = config["num_layers"]
        self.dropout_p = config["dropout"]
        
        self.attention = attention
        self.rnn = nn.GRU((e_hidden_size * 2) + self.input_size, self.hidden_size, self.num_layers)
        
        self.fc_out = nn.Linear((e_hidden_size * 2) + self.hidden_size + self.input_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_p)
    
    def forward(self, input, hidden, encoder_outputs):
        # input = [1, batch size, input_size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        a = self.attention(hidden, encoder_outputs)
        
        # a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        # a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        # weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        # weighted = [1, batch size, enc hid dim * 2]
        rnn_input = torch.cat((input, weighted), dim=2)
        # rnn_input = [1, batch size, (enc hid dim * 2) + input_size]
        
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()
        
        prediction = self.fc_out(torch.cat((output, weighted, input), dim=2))
        
        # prediction = [1, batch size, output size]
        return prediction, hidden.squeeze(0)


class Model(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        enc_hidden_size = config["encoder"]["hidden_size"]
        dec_hidden_size = config["decoder"]["hidden_size"]
        self.device = device
    
        self.encoder = EncoderBiGRU(config["encoder"], dec_hidden_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.decoder = Decoder(config["decoder"], self.attention, enc_hidden_size)
        
        assert self.encoder.hidden_size == self.decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.num_layers == self.decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"
            
    def forward(self, input_tensor, target_tensor, teacher_forcing_ratio=0.5):
        # input_tensor = [seq len, batch size, 1]
        # target_tensor = [trg seq len, batch size, 1]
        batch_size = target_tensor.shape[1]
        trg_len = target_tensor.shape[0]
        output_size = self.decoder.output_size
        
        outputs = torch.zeros(trg_len, batch_size, output_size).to(self.device)
        
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        
        decoder_output = torch.zeros(1, batch_size, 1).to(self.device)
        decoder_hidden = encoder_hidden
        
        for t in range(trg_len):
            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = target_tensor[t].unsqueeze(0) if teacher_force else decoder_output
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            outputs[t] = decoder_output
        
        return outputs
