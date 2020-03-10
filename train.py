import torch


def train(input, target, model, optimizer, criterion, clip):
    model.train()
    
    optimizer.zero_grad()
    output = model(input, target)

    loss = criterion(output, target)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    
    return loss.item()


def validate(input, target, model, criterion):
    model.eval()
    with torch.no_grad():
        output = model(input, target)
    loss = criterion(output, target)
    return loss.item()
