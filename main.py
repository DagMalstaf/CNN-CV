
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import numpy as np
from time import time
import model
import train
from data_functions import load_and_prepare_data, load_config, load_test_data

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def calculate_accuracy(predicted_labels, true_labels):
    correct = (predicted_labels == true_labels).sum()
    total = len(true_labels)
    accuracy = correct / total
    return accuracy

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def create_data_loaders(train_imgs, train_labels, val_imgs, val_labels, batch_size):
    train_loader = DataLoader(list(zip(train_imgs, train_labels)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(val_imgs, val_labels)), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def main():
    start_time = time()
    config = load_config('config.yaml')
    net = model.resnet50(num_classes=config['model']['num_classes'])
    criterion = nn.CrossEntropyLoss()

    optimizer = getattr(optim, config['training']['optimizer'])(
        net.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        **config['training'].get('optimizer_params', {})
    )

    train_imgs, train_labels, val_imgs, val_labels = load_and_prepare_data(config['dataset']['csv_path_train'], transform)

    test_loader = load_test_data(
        csv_file=config['dataset']['csv_path_test'],
        root_dir='data/images/',  # Specify the correct root directory where images are stored
        transform=transform
    )

    train_loader, val_loader = create_data_loaders(
        train_imgs, train_labels, val_imgs, val_labels, config['training']['batch_size'])
    
    scheduler = None
    scheduler_type = config['training'].get('lr_scheduler')
    
    if scheduler_type == 'StepLR':
        scheduler_params = config['training'].get('step_lr_params', {})
        scheduler = lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler_params = config['training'].get('reduce_lr_on_plateau_params', {})
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)

    print('-' * 20)
    print('Started training the model...')
    trained_model = train.train_model(
            net, criterion, optimizer, train_loader, val_loader,
            num_epochs=config['training']['epochs'],
            scheduler=scheduler
        )
    
    print('Finished training the model.')
    print('-' * 20)

    print('-' * 20)
    print('Started testing the model...')
    test_preds, test_labels = train.evaluate_model(trained_model, test_loader)
    test_accuracy = calculate_accuracy(np.array(test_preds), np.array(test_labels))
    print('Finished training the model.')
    print('-' * 20)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    end_time = time()
    print('-' * 20)
    total_time = end_time - start_time
    print(f"Finished processing within: {total_time} seconds")
    print('-' * 20)

if __name__ == '__main__':
    main()

