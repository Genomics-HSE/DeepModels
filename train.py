import random
import torch

teacher_forcing_ratio = 0.5


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

