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


class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    
    def forward(self, input_tensor, target_tensor, teacher_forcing_ratio=0.5):
        batch_size = target_tensor.shape[1]
        trg_len = target_tensor.shape[0]
        output_size = self.decoder.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, output_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_hidden, encoder_cell = self.encoder(input_tensor)

        # first input to the decoder is the <sos> tokens
        decoder_input = target_tensor[0].unsqueeze(0)
        decoder_cell = encoder_cell
        decoder_hidden = encoder_hidden
        
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = decoder_output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            decoder_input = target_tensor[t].unsqueeze(0) if teacher_force else decoder_output
        
        return outputs


def train(model, input, target, optimizer, criterion, clip):
    
    model.train()
    
    optimizer.zero_grad()
    output = model(input, target)
    loss = criterion(output, target)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
        
    return loss.item()
