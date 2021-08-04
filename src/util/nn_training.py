import torch

def optimizer_factory(config, parameters):
    optimizer_type = config['optimizer']
    learning_rate = config['lr']
    weight_decay = config['weight_decay']
    if optimizer_type == 'adam':  
        optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer
