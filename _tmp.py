import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, config):
        super(EncoderRNN, self).__init__()
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        
        self.gru = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
    
    def forward(self, input, hidden):
        # input = input.unsqueeze(0).unsqueeze(0)
        output, hidden = self.gru(input, hidden)
        return output, hidden
    
    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, config):
        super(AttnDecoderRNN, self).__init__()
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.output_size = config["output_size"]
        self.length = config["length"]
        self.dropout_p = config["dropout"]
        
        self.attn = nn.Linear(self.input_size + self.hidden_size, self.length)
        self.attn_combine = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, input, hidden, encoder_outputs):
        
        hidden = hidden.permute(1, 0, 2)
        repeat_vals = [-1] + [input.shape[1] // hidden.shape[1]] + [-1]
        concatenated = torch.cat((input, hidden.expand(*repeat_vals)), dim=-1)
        attn = self.attn(concatenated)
        attn_weights = F.softmax(attn, dim=-1)
        # print("attn_weights", attn_weights.shape)
        # print("encoder_outputs", encoder_outputs.shape)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        # attn_applied = attn_applied.permute(1, 0, 2)
        
        # print("attn_applied", attn_applied.shape)
        # print("input", input.shape)
        output = torch.cat((input, attn_applied), dim=-1)
        
        output = self.attn_combine(output)
        
        # print("output", output.shape)
        output = F.relu(output)
        hidden = hidden.permute(1, 0, 2)
        output, hidden = self.gru(output, hidden)
        
        # print("output", output.shape)
        # print("hidden", hidden.shape)
        output = self.out(output)
        # print("output", output.shape)
        return output, hidden, attn_weights
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def train(input_tensor, target_tensor, encoder, decoder, encoder_opt, decoder_opt, criterion):
    encoder.train()
    decoder.train()
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()
    
    device = input_tensor.get_device()
    encoder_hidden = encoder.initHidden(input_tensor.size(0)).to(device)
    
    target_length = target_tensor.size(1)
    loss = 0
    
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    
    decoder_input = target_tensor[:, 0, :].unsqueeze(1)
    decoder_hidden = encoder_hidden
    print(decoder_hidden.size())
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        decoder_output, decoder_hidden, decoder_attention = decoder(
            target_tensor, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_tensor)
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[:, di, :].unsqueeze(1))
            decoder_input = decoder_output
    
    loss.backward()
    
    encoder_opt.step()
    decoder_opt.step()
    
    return loss.item()


def validate(input_tensor, target_tensor, encoder, decoder, criterion):
    encoder.eval()
    decoder.eval()
    
    device = input_tensor.get_device()
    encoder_hidden = encoder.initHidden(input_tensor.size(0)).to(device)
    target_length = target_tensor.size(1)
    
    loss = 0
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
        decoder_hidden = encoder_hidden
        decoder_input = target_tensor[:, 0, :].unsqueeze(1)
        
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[:, di, :].unsqueeze(1))
            decoder_input = decoder_output
    
    return loss.item()