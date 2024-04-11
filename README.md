# CNN for Computer Vision (CNN-CV)

This repository contains the implementation of a Convolutional Neural Network (CNN) for image classification tasks. The project is structured to facilitate data processing, model definition, training, tuning, and evaluation.

### Folders and Files Description:

- `__pycache__`: Compiled Python files that are automatically generated.
- `data`: Directory containing the dataset and related files.
  - `images`: Folder where image data are stored.
  - `output_notaugmented*.csv`: CSV files with output data for non-augmented test sets.
- `.gitignore`: Lists files and directories that are to be ignored by git version control.
- `best_hyperparameters.yaml`: Contains the best hyperparameters found during tuning.
- `config.yaml`: General configuration settings for the model and training process.
- `data_functions.py`: Helper functions for data handling and preprocessing.
- `hyperparams.yaml`: Hyperparameter configurations used in the model.
- `main.py`: Entry point of the program to run training or testing processes.
- `model.py`: Defines the CNN architecture.
- `train.py`: Contains code to train the CNN.
- `tuning.py`: Script for hyperparameter tuning.

## Getting Started

To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/CNN-CV.git
cd CNN-CV
