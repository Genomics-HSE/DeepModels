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
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers)
    
    def forward(self, input):
        output, hidden = self.gru(input)
        return output, hidden


class AttnDecoderGRU(nn.Module):
    def __init__(self, config):
        super(AttnDecoderGRU, self).__init__()
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.output_size = config["output_size"]
        self.num_layers = config["num_layers"]
        self.dropout_p = config["dropout"]
        self.length = config["length"]
        
        self.attn = nn.Linear(self.input_size + self.hidden_size, self.length)
        self.attn_combine = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, input, hidden, encoder_outputs):
        
        repeat_vals = [input.shape[0] // hidden.shape[0]] + [-1] * (len(hidden.shape) - 1)
        print("repeat_vals", repeat_vals)
        print("input", input.size())
        print("hidden", hidden.size())
        
        concatenated = torch.cat((input, hidden.expand(*repeat_vals)), dim=-1)
        attn = self.attn(concatenated)
        attn_weights = F.softmax(attn, dim=-1)
        # print("attn_weights", attn_weights.shape)
        # print("encoder_outputs", encoder_outputs.shape)
        attn_applied = torch.bmm(attn_weights.permute(1, 0, 2), encoder_outputs.permute(1, 0, 2))
        attn_applied = attn_applied.permute(1, 0, 2)
        
        # print("attn_applied", attn_applied.shape)
        # print("input", input.shape)
        output = torch.cat((input, attn_applied), dim=-1)
        
        output = self.attn_combine(output)
        
        # print("output", output.shape)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        
        # print("output", output.shape)
        # print("hidden", hidden.shape)
        output = self.out(output)
        # print("output", output.shape)
        return output, hidden, attn_weights


class Model(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        
        self.encoder = EncoderGRU(config["encoder"])
        self.decoder = AttnDecoderGRU(config["decoder"])
        self.device = device

        assert self.encoder.hidden_size == self.decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.num_layers == self.decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, input_tensor, target_tensor, teacher_forcing_ratio=0.5):
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
