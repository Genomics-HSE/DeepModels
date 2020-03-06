import torch
import torch.nn as nn
import random


class EncoderLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
    
    def forward(self, input):
        outputs, (hidden, cell) = self.rnn(input)
        return hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.fc_out = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell


class Model(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        
        self.encoder = EncoderLSTM(config["encoder"]).to(device)
        self.decoder = DecoderLSTM(config["decoder"]).to(device)
        self.device = device
        
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    
    def forward(self, input_tensor, target_tensor, teacher_forcing_ratio=0.5):
        batch_size = target_tensor.shape[1]
        trg_len = target_tensor.shape[0]
        output_size = self.decoder.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, output_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_hidden, encoder_cell = self.encoder(input_tensor)

        decoder_cell = encoder_cell
        decoder_hidden = encoder_hidden
        decoder_output = torch.zeros(1, batch_size, 1)

        for t in range(trg_len):
            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = target_tensor[t].unsqueeze(0) if teacher_force else decoder_output
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            outputs[t] = decoder_output
        
        return outputs

