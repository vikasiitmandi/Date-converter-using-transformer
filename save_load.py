import os
import torch

def find_last_checkpoint(checkpoint_dir):
    """
    Finds the latest checkpoint file in the directory.
    
    Args:
        checkpoint_dir (str): The directory where checkpoint files are stored.
    
    Returns:
        int: The epoch number of the latest checkpoint.
    
    Raises:
        IOError: If no checkpoint files are found in the directory.
    """
    epochs = []
    for name in os.listdir(checkpoint_dir):
        if os.path.splitext(name)[-1] == '.pth':
            epochs.append(int(name.strip('ckpt_epoch_.pth')))
    if len(epochs) == 0:
        raise IOError('No checkpoint found in {}'.format(checkpoint_dir))
    return max(epochs)

def save_checkpoint(checkpoint_dir, epoch, model, optimizer=None):
    """
    Saves the current state of the model and optimizer as a checkpoint.
    
    Args:
        checkpoint_dir (str): The directory where the checkpoint will be saved.
        epoch (int): The current epoch number.
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer, optional): The optimizer to be saved. Defaults to None.
    """
    checkpoint = {'epoch': epoch}

    if isinstance(model, torch.nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    checkpoint['model'] = model_state_dict

    if optimizer is not None:
        optimizer_state_dict = optimizer.state_dict()
        checkpoint['optimizer'] = optimizer_state_dict
    else:
        checkpoint['optimizer'] = None

    torch.save(checkpoint, os.path.join(checkpoint_dir, 'ckpt_epoch_%02d.pth' % epoch))

def load_checkpoint(checkpoint_dir, epoch=-1):
    """
    Loads a checkpoint from the directory.
    
    Args:
        checkpoint_dir (str): The directory where checkpoint files are stored.
        epoch (int, optional): The epoch number of the checkpoint to load. 
                               Defaults to -1, which loads the latest checkpoint.
    
    Returns:
        dict: The checkpoint dictionary containing the model and optimizer states.
    """
    if epoch == -1:
        epoch = find_last_checkpoint(checkpoint_dir)
    checkpoint_name = 'ckpt_epoch_%02d.pth' % epoch
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return ckpt

def save_model(checkpoint_dir, epoch, model):
    """
    Saves only the model state as a checkpoint.
    
    Args:
        checkpoint_dir (str): The directory where the checkpoint will be saved.
        epoch (int): The current epoch number.
        model (torch.nn.Module): The model to be saved.
    """
    save_checkpoint(checkpoint_dir, epoch, model, optimizer=None)

def load_model(checkpoint_dir, epoch, model):
    """
    Loads the model state from a checkpoint.
    
    Args:
        checkpoint_dir (str): The directory where checkpoint files are stored.
        epoch (int): The epoch number of the checkpoint to load.
        model (torch.nn.Module): The model to load the state into.
    
    Returns:
        torch.nn.Module: The model with the loaded state.
    """
    try:
        ckpt = load_checkpoint(checkpoint_dir, epoch)
        model_state_dict = ckpt['model']

        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
    except Exception as e:
        print('Failed to load model, {}'.format(e))
    return model

def load_optimizer(checkpoint_dir, epoch, optimizer):
    """
    Loads the optimizer state from a checkpoint.
    
    Args:
        checkpoint_dir (str): The directory where checkpoint files are stored.
        epoch (int): The epoch number of the checkpoint to load.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
    
    Returns:
        torch.optim.Optimizer: The optimizer with the loaded state.
    """
    try:
        ckpt = load_checkpoint(checkpoint_dir, epoch)
        optimizer_state_dict = ckpt['optimizer']
        optimizer.load_state_dict(optimizer_state_dict)
    except Exception as e:
        print('Failed to load optimizer, {}'.format(e))
    return optimizer
