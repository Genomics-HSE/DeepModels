import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderGRU(nn.Module):
    def __init__(self, config):
        super(EncoderGRU, self).__init__()
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout_p = config["dropout"]
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers,
                          dropout=self.dropout_p)
    
    def forward(self, input):
        output, hidden = self.gru(input)
        return hidden


class DecoderGRU(nn.Module):
    def __init__(self, config):
        super(DecoderGRU, self).__init__()
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.output_size = config["output_size"]
        self.num_layers = config["num_layers"]
        self.length = config["length"]
        self.dropout_p = config["dropout"]
        
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers,
                          dropout=self.dropout_p)
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.dropout(output)
        output = self.out(output)
        return output, hidden


class Model(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.encoder = EncoderGRU(config["encoder"]).to(device)
        self.decoder = DecoderGRU(config["decoder"]).to(device)
        self.device = device
        
        assert self.encoder.hidden_size == self.decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.num_layers == self.decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, input_tensor, target_tensor, teacher_forcing_ratio=0.5):
    
        batch_size = target_tensor.shape[1]
        trg_len = target_tensor.shape[0]
        output_size = self.decoder.output_size
    
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, output_size).to(self.device)
    
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_hidden = self.encoder(input_tensor)
    
        # decoder_input = target_tensor[0].unsqueeze(0)
        decoder_output = torch.zeros(1, batch_size, 1)
        decoder_hidden = encoder_hidden
    
        for t in range(trg_len):
            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = target_tensor[t].unsqueeze(0) if teacher_force else decoder_output
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
    
        return outputs
