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

