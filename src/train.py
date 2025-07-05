import os
import torch
from torch.utils.data import DataLoader
from datetime import datetime

import losses as losses
from solver import Solver
from dataset import HSTNDataSet
from hstasnet import HSTasNet
from config import config, get_model_args, get_file_paths

def define_args():
    """Define the training parameters from config.

    Returns:
        args (dict): A dictionary containing the parameters for the training routine and the model.
    """
    file_paths = get_file_paths()
    
    args = {
        # Training parameters.
        'solver_path': file_paths['solver_path'],
        'batch_size': config.training['batch_size'],
        'num_epochs': config.training['num_epochs'],
        'num_workers': config.training['num_workers'],
        'device': torch.device(config.training['device']),

        # Optimizer parameters.
        'learning_rate': config.training['learning_rate'],
        'weight_decay': config.training['weight_decay'],

        # Model parameters.
        'model_name': config.model['name'],
        'model_path': file_paths['model_path'],
        'model_srcs': config.data['sources'],
        'model_args': get_model_args(),

        # Other parameters.
        'continue_from': config.other['continue_from'],
        'log_path': file_paths['log_path'],
        'data_root': config.data['root'],
        'sample_rate': config.data['sample_rate'],
    }
    return args

def define_loaders(args):
    """Define DataLoaders for the training, validation, and test sets.

    Args:
        args (dict): Dictionary containing the training parameters.

    Returns:
        loaders (dict): Dictionary containing the DataLoaders.
    """
        
    root = args['data_root']
    sources = args['model_srcs']

    trn_dataset = HSTNDataSet(root, 'train', sources)
    val_dataset = HSTNDataSet(root, 'valid', sources)
    tst_dataset = HSTNDataSet(root, 'test', sources)

    # Define DataLoaders.
    trn_loader = DataLoader(trn_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)
    tst_loader = DataLoader(tst_dataset, batch_size=args['batch_size'], shuffle=False)

    # Store DataLoaders in a dictionary.
    loaders = {
        'trn_loader': trn_loader,
        'val_loader': val_loader,
        'tst_loader': tst_loader,
        }
    
    return loaders


def main(args, train=True):
    """Define model, loaders, optimizer, criterion, scheduler, solver and launch training.

    Args:
        args (dict): Dictionary containing the training parameters.

    Returns:
        solver (Solver): Solver instance containing the training information, e.g. model and loss history.
    """

    # Define loaders.
    loaders = define_loaders(args)

    # Define model.
    model = HSTasNet(**args['model_args'])
    os.makedirs(os.path.dirname(args['model_path']), exist_ok=True)

    # Define criterion.
    criterion = losses.l1_loss

    # Define optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    # Define scheduler.
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=4)

    # Define solver.
    solver = Solver(model, criterion, optimizer, scheduler, loaders, args, device=args['device'])
    os.makedirs(os.path.dirname(args['solver_path']), exist_ok=True)

    print(f"Model: {model.__class__.__name__}")
    print(f"Device: {args['device']}")
    print(f"Batch size: {args['batch_size']}")
    print(f"Learning rate: {args['learning_rate']}")
    print(f"Weight decay: {args['weight_decay']}")
    print(f"Number of epochs: {args['num_epochs']}")
    print(f"Training samples: {len(loaders['trn_loader'])}")
    print(f"Validation samples: {len(loaders['val_loader'])}")
    print(f"Test samples: {len(loaders['tst_loader'])}")
    print(f"Model path: {args['model_path']}")
    print(f"Solver path: {args['solver_path']}")
    print(f"Log path: {args['log_path']}")
    print("-" * 50)

    # Train model.
    solver = solver.train() if train else solver

    # Save log file.
    os.makedirs(os.path.dirname(args['log_path']), exist_ok=True)
    os.rename(src='logs/train.log', dst=args['log_path'])

    return solver


if __name__ == '__main__':

    # Empty the GPU cache.
    torch.cuda.empty_cache()

    print("*** START TRAINING ***\n")

    # Read parameters for the training routine and the model.
    args = define_args()

    # Train the model.
    solver = main(args)

    print("\n*** FINISHED TRAINING ***")
