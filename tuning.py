import optuna
import yaml
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import model
import train
import torch.nn as nn

def load_hyperparams(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def objective(trial):
    # Load configuration
    config = load_hyperparams('config.yaml')
    hyperparams = load_hyperparams('hyperparams.yaml')['hyperparameters']

    # Suggest values for the hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', hyperparams['learning_rate']['low'], hyperparams['learning_rate']['high'])
    momentum = trial.suggest_uniform('momentum', hyperparams['momentum']['low'], hyperparams['momentum']['high'])
    weight_decay = trial.suggest_loguniform('weight_decay', hyperparams['weight_decay']['low'], hyperparams['weight_decay']['high'])
    batch_size = trial.suggest_categorical('batch_size', hyperparams['batch_size']['values'])
    epochs = trial.suggest_categorical('epochs', hyperparams['epochs']['values'])

    # Define model, optimizer, loss function
    net = model.resnet50(num_classes=config['model']['num_classes'])
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Data loading and preprocessing
    train_loader, val_loader = train.prepare_data_loaders(batch_size)

    # Train and evaluate the model
    trained_model = train.train_model(net, criterion, optimizer, train_loader, val_loader, num_epochs=epochs)
    val_preds, val_labels = train.evaluate_model(trained_model, val_loader)
    val_accuracy = train.calculate_accuracy(val_preds, val_labels)

    return val_accuracy

def save_best_hyperparams(study, save_path):
    best_params = study.best_trial.params
    with open(save_path, 'w') as file:
        yaml.dump({'best_hyperparameters': best_params}, file)

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    save_best_hyperparams(study, 'best_hyperparams.yaml')
    print("Best hyperparameters saved to best_hyperparams.yaml")

if __name__ == '__main__':
    main()
