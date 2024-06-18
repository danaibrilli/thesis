import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from GraphQADataset import GraphQADataset
import pickle
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from .train import prepare_datasets, train, initialize_models, evaluate
import os
import shutil


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def main():
    """
    Parses command-line arguments, loads configuration file, prepares data,
    initializes models, trains the model, and evaluates performance on the test set.
    """

    # Parse command-line arguments for configuration file path
    parser = argparse.ArgumentParser(description='Train HCRN model for Video Question Answering')
    parser.add_argument('-conf', '--config', required=True,
                        help='Path to the configuration YAML file.')

    args = parser.parse_args()
    config_path = args.config

    # Load configuration from YAML file
    print(f"Loading configuration from: {config_path}")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load preprocessed data
    print("Loading preprocessed data...")
    with open('/home/dbrilli/GNN_MLP/preprocess/data_small_preprocessed_all.pickle', 'rb') as f:
        data_small_train_all = pickle.load(f)

    # Prepare training, validation, and test datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(data_small_train_all, config)

    # Initialize GNN model, HCRN model, data loaders, device, and vocabulary
    conv, hcrn, train_loader, val_loader, device, vocab = initialize_models(config, train_dataset, val_dataset, config_path)

    # Train the HCRN model with the GNN for video question answering
    train(config, conv, hcrn, train_loader, val_loader, device)

    # Evaluate the model's performance on the test set
    evaluate(hcrn, conv, test_dataset, config, device)


if __name__ == "__main__":
    main()
