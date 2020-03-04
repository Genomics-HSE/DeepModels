import random
import torch

teacher_forcing_ratio = 0

def train_kotok(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    device = input_tensor.get_device()
    encoder_hidden = encoder.initHidden(input_tensor.size(1)).to(device)
    
    encoder.train()
    decoder.train()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    target_length = target_tensor.size(0)
    
    loss = 0
    
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    
    decoder_input = target_tensor[0].unsqueeze(0)
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
            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
            decoder_input = decoder_output
    
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item()


def train_lstm(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    
    encoder.train()
    decoder.train()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    target_length = target_tensor.size(0)
    
    loss = 0
    
    encoder_hidden, cell = encoder(input_tensor)
    
    decoder_input = target_tensor[0].unsqueeze(0)
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        decoder_output, decoder_hidden, decoder_attention = decoder(
            target_tensor, decoder_hidden, cell)
        loss += criterion(decoder_output, target_tensor)
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, cell)
            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
            decoder_input = decoder_output
    
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item()


def validate_lstm(input_tensor, target_tensor, encoder, decoder, criterion):
    
    encoder.eval()
    decoder.eval()
    
    target_length = target_tensor.size(0)
    
    loss = 0
    with torch.no_grad():
        encoder_hidden, cell = encoder(input_tensor)
        decoder_input = target_tensor[0].unsqueeze(0)
        decoder_hidden = encoder_hidden
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        if use_teacher_forcing:
            decoder_output, decoder_hidden, decoder_attention = decoder(
                target_tensor, decoder_hidden, cell)
            loss += criterion(decoder_output, target_tensor)
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, cell)
                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                decoder_input = decoder_output
        
    return loss.item()