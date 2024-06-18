import glob 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from torch.nn.functional import softmax
from torch.optim import AdamW
import joblib
from sklearn.metrics import classification_report
import time
import wandb
from GraphQADataset_EGAT_1file import GraphQADataset, custom_collate_fn
import pickle
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
import wandb
from GINE import GINE_GNN
from EdgeGAT import EdgeGATGNN
from GINE_EGAT import GINE_EGAT_GNN
from models.scene_model import SCENE
from dgl.nn.pytorch.conv import GINEConv, EdgeGATConv
import numpy as np
import warnings
import random
import argparse
import os
from models.HCRN import HCRNNetwork
import shutil
import json
from utils import shuffle
from utils import load_vocab



def save_checkpoint(epoch, model, optimizer, filename):
    """
    Saves the model and optimizer state dictionaries to a checkpoint file.

    Args:
        epoch: [int] Current training epoch.
        model: [nn.Module] The model to be saved.
        optimizer: [optim.Optimizer] The optimizer used for training.
        filename: [str] Path to the checkpoint file.
    """

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)


def batch_accuracy(predicted, true):
    """
    Calculates the accuracy for a batch of predictions and ground truth labels.

    Args:
        predicted: [Tensor] Predicted labels.
        true: [Tensor] Ground truth labels.

    Returns:
        float: Batch accuracy.
    """

    predicted = predicted.detach().argmax(1)  
    agreeing = (predicted == true) 
    return torch.sum(agreeing).float() / len(true)


def prepare_datasets(data_small_train_all, config):
    """
    Prepares train, validation, and test datasets for the HCRN model.

    Args:
        data_small_train_all: [dict] Dictionary containing all data for training.
        config: [dict] Configuration dictionary containing hyperparameters and paths.

    Returns:
        tuple: A tuple of three GraphQADataset objects representing the train, validation, and test datasets.
    """

    # Extract data from the provided dictionary
    all_video_ids = np.array(data_small_train_all['all_video_ids'])
    all_question_ids = np.array(data_small_train_all['all_question_ids'])
    all_questions = np.array(data_small_train_all['all_questions'])
    all_answers = np.array(data_small_train_all['all_answers'])

    # Shuffle data for train/val/test split with a fixed seed for reproducibility
    seed = config['seed']
    np.random.seed(seed)  # Set random seed for shuffling
    all_video_ids = shuffle(all_video_ids)
    all_question_ids = shuffle(all_question_ids)
    all_questions = shuffle(all_questions)
    all_answers = shuffle(all_answers)

    # Extract data split sizes from configuration
    train_num = config['train']['train_num']
    val_num = config['val']['val_num']
    test_num = config['test']['test_num']

    # Calculate indices for train, validation, and test sets
    train_idx = test_num + train_num
    val_idx = train_idx + val_num
    test_idx = test_num

    # Path to scene graphs (used for visual features)
    scene_graphs_path = config['sg_path']

    # Prepare dictionaries containing answer counts for each dataset (for potential filtering)
    train_answers = {k: v for k, v in sorted(dict(zip(*np.unique(train_answers, return_counts=True))).items(), key=lambda item: item[1])}
    val_answers = {k: v for k, v in sorted(dict(zip(*np.unique(val_answers, return_counts=True))).items(), key=lambda item: item[1])}
    test_answers = {k: v for k, v in sorted(dict(zip(*np.unique(test_answers, return_counts=True))).items(), key=lambda item: item[1])}

    # Edge feature type (used for constructing the graph)
    edge_feat_type = config['train']['edge_feat_type']

    # Create GraphQADataset objects for train, validation, and test sets
    train_dataset = GraphQADataset(all_video_ids[test_idx:train_idx], all_question_ids[test_idx:train_idx], all_questions[test_idx:train_idx],
                                   all_answers[test_idx:train_idx], blind=True, use_sg=True,
                                   scene_graphs_path=scene_graphs_path, mode='train', edge_feat=edge_feat_type)

    val_dataset = GraphQADataset(all_video_ids[train_idx:val_idx], all_question_ids[train_idx:val_idx], all_questions[train_idx:val_idx],
                                 all_answers[train_idx:val_idx], blind=True, use_sg=True,
                                 scene_graphs_path=scene_graphs_path, mode='train', edge_feat=edge_feat_type)

    test_dataset = GraphQADataset(all_video_ids[:test_idx], all_question_ids[:test_idx], all_questions[:test_idx],
                                  all_answers[:test_idx], blind=True, use_sg=True,
                                  scene_graphs_path=scene_graphs_path, mode='train', edge_feat=edge_feat_type)

    return train_dataset, val_dataset, test_dataset


def initialize_models(config, train_dataset, val_dataset, config_path):
    """
    Initializes the models, data loaders, and training environment based on the configuration.

    Args:
        config: [dict] Configuration dictionary containing hyperparameters and paths.
        train_dataset: [GraphQADataset] Training dataset object.
        val_dataset: [GraphQADataset] Validation dataset object.
        config_path: [str] Path to the configuration file.

    Returns:
        tuple: A tuple containing the initialized HCRN model, GNN convolutional layer, data loaders, device, and vocabulary.
    """

    # Extract hyperparameters from the configuration
    lr = float(config['train']['lr'])
    batch_size = config['train']['batch_size']
    epochs = config['train']['num_epochs']
    node_feat_shape = config['train']['node_feat_shape']
    edge_feat_shape = config['train']['edge_feat_shape']
    in_feats = config['train']['in_feats']
    out_feats = config['train']['out_feats']
    num_layers = config['train']['num_layers']
    dropout = config['train']['dropout']
    path = config['dataset']['save_dir']
    mlp_in = config['train']['mlp_in_feats']
    hidden_sizes = config['train']['hidden_size']
    attention_heads = config['train']['attention_heads']

    # Create save directory if it doesn't exist
    os.makedirs(path, exist_ok=True)  # Improved error handling

    # Copy configuration file for reference
    shutil.copy(config_path, path + 'config.yml')

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn,
                              num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn,
                            num_workers=config['num_workers'])

    # Determine device for computations (CPU or GPU)
    device = torch.device(f"cuda:{config['gpu_id']}") if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Extract additional hyperparameters for GNN and HCRN
    graph_feat_size = config['train']['in_feats']
    hidden_feats = 195

    # Initialize GNN convolutional layer based on architecture choice
    if config['train']['gnn_arch'] == 'EdgeGATConv':
        conv = EdgeGATConv(node_feat_shape, edge_feat_shape, in_feats, attention_heads, allow_zero_in_degree=True)
    elif config['train']['gnn_arch'] == 'GINE2L':
        conv = GINE_GNN(node_feat_shape, hidden_feats=hidden_feats, out_feats=graph_feat_size)
    elif config['train']['gnn_arch'] == 'EGAT2L':
        conv = EdgeGATGNN(node_feat_shape, edge_feat_shape, hidden_feats=hidden_feats, out_feats=graph_feat_size)
    elif config['train']['gnn_arch'] == 'GINE_EGAT':
        conv = GINE_EGAT_GNN(in_node_feats=node_feat_shape, in_edge_feats=node_feat_shape, hidden_feats=hidden_feats,
                             gine_out_feats=hidden_feats, egat_out_feats=graph_feat_size, num_heads=attention_heads)
    elif config['train']['gnn_arch'] == 'SCENE':
        conv = SCENE(node_feat_shape, edge_feat_shape, num_heads=attention_heads, hidden_size=hidden_feats,
                     out_feats=graph_feat_size)
    else:
        raise ValueError(f"Invalid GNN architecture: {config['train']['gnn_arch']}")

    conv.to(device)  # Move GNN to chosen device

    # Load vocabulary from JSON file
    vocab_json = '/ssd_data/agqa/questions/csvs/balanced/tgif-qa_frameqa_vocab-balanced.json'
    print('loading vocab from %s' % (vocab_json))
    vocab = load_vocab(vocab_json)  # Assuming load_vocab function is defined elsewhere

    # Hyperparameters for HCRN model
    module_dim = 512
    word_dim = 768
    k_max_graph_level = 3
    k_max_clip_level = 5
    spl_resolution = 1

    hcrn = HCRNNetwork(module_dim=module_dim, graph_embedding_dim=256, word_dim=word_dim,
                       k_max_graph_level=k_max_graph_level, k_max_clip_level=k_max_clip_level,
                       spl_resolution=spl_resolution, vocab=vocab, dropout_style=1, dropout_prob=0.1,
                       crn_dropout_prob=0.1)
    hcrn.to(device)  # Move HCRN to chosen device

    return conv, hcrn, train_loader, val_loader, device, vocab


def train(config, conv, hcrn, train_loader, val_loader, device):
    """
    Trains the HCRN model with the GNN for video question answering.

    Args:
        config: [dict] Configuration dictionary containing hyperparameters and paths.
        train_dataset: [GraphQADataset] Training dataset object.
        val_dataset: [GraphQADataset] Validation dataset object.
        test_dataset: [GraphQADataset] Test dataset object (optional).
        conv: [nn.Module] GNN convolutional layer.
        hcrn: [HCRNNetwork] HCRN model.
        train_loader: [DataLoader] Data loader for training set.
        val_loader: [DataLoader] Data loader for validation set.
        device: [torch.device] Device for computations (CPU or GPU).
        vocab: [dict] Vocabulary dictionary.

    Returns:
        None
    """

    # Use WANDB for logging if enabled in config
    WANDB = config['use_wandb']
    if WANDB:
        wandb.init(project="SG_GNN_HCRN", config=config)
    
    in_feats = config['train']['in_feats']

    train_loss = []
    val_loss = []
    val_micro = []
    val_macro = []
    val_recall = []
    val_accuracy = []

    
    epochs = config['train']['num_epochs']
    criterion = nn.CrossEntropyLoss()  # Assuming nn is imported
    optim = torch.optim.Adam(list(conv.parameters()) + list(hcrn.parameters()), lr=float(config['train']['lr']))

    for epoch in range(epochs):
        hcrn.train()
        conv.train()
        batch_losses = []
        batch_acc = []
        for batch in tqdm(train_loader):
            optim.zero_grad()
            cls = batch['cls'].to(device)
            labels = batch['encoded_answer'].to(device)
            gbatch = batch['graphs'].to(device)
        
            nfeat = batch['ndata'].to(device)
            efeat = batch['edata'].to(device)
            
            nodes = batch['num_n']
            edges = batch['num_e']

            res = conv(gbatch, nfeat, efeat)
            nodes = batch['num_n']
            r = res.cpu().detach().numpy()
            graph_mean = r

            graph_mean = np.array([np.mean(r[start:finish], axis=0) for start, finish in zip(nodes[:-1], nodes[1:])])
            

            try: #-1 is for batch size (not perfect division)
                graph_mean = torch.from_numpy(np.reshape(graph_mean, (-1, 5, 3, in_feats))).to(device)
            except:
                print('ERROR', graph_mean.shape)
                continue
            
            
            outputs = hcrn(graph_mean, cls)

            loss = criterion(outputs.type(torch.float), labels.type(torch.long))
            batch_losses.append(loss.item())
            loss.backward()
            optim.step()

            acc = batch_accuracy(outputs, labels)
            
            train_accuracy = acc.float().mean().cpu().numpy()
            batch_acc.append(train_accuracy)
            if WANDB:
                wandb.log({"ce_loss": loss.item(), "train_acc": acc.float().mean().cpu().numpy(), 
                            "avg_loss": sum(batch_losses)/len(batch_losses),
                            'avg_acc': sum(batch_acc)/len(batch_acc)})

        hcrn.eval()
        conv.eval()
        print('into validation')
        model_result = []
        targets = []
        val_batch_loss = []

        for batch in tqdm(val_loader):
            cls = batch['cls'].to(device)
            labels = batch['encoded_answer'].to(device)
            gbatch = batch['graphs'].to(device)
            
            nfeat = batch['ndata'].to(device)
            efeat = batch['edata'].to(device)

            
            with torch.no_grad():
                
                
                res = conv(gbatch, nfeat, efeat)
                nodes = batch['num_n']
                r = res.cpu().detach().numpy()

                graph_mean = r
                graph_mean = np.array([np.mean(r[start:finish], axis=0) for start, finish in zip(nodes[:-1], nodes[1:])])
                
                if config['train']['gnn_arch'] == 'EdgeGATConv':
                    graph_mean = np.mean(graph_mean, axis=1) #mean across attention heads
            
                try:
                    graph_mean = torch.from_numpy(np.reshape(graph_mean, (-1, 5, 3, in_feats))).to(device)
                except:
                    print('ERROR', graph_mean.shape)
                    continue

            
                outputs = hcrn(graph_mean, cls)
                
                vloss = criterion(outputs.type(torch.float), labels.type(torch.long))
                val_batch_loss.append(vloss.item()) 
                model_result.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        result = calculate_metrics(np.array(model_result), np.array(targets))
        print("epoch:{:2d} val: "
                "micro f1: {:.3f} "
                "macro f1: {:.3f} "
                "recall: {:.3f} "
                "accuracy: {:.3f}".format(epoch,
                                            result['micro/f1'],
                                            result['macro/f1'],
                                            result['recall'],
                                            result['accuracy']))
        val_micro.append(result['micro/f1'])
        val_macro.append(result['macro/f1'])
        val_recall.append(result['recall'])
        val_accuracy.append(result['accuracy'])

        if WANDB:
            wandb.log({"epoch": epoch, "val_micro": result['micro/f1'], "val_macro": result['macro/f1'], 
                "val_recall": result['recall'], "val_accuracy": result['accuracy'],
                "val_ce_loss": sum(val_batch_loss)/len(val_loader)})

        if epoch % 5 == 0:
            torch.save(hcrn.state_dict(), config['dataset']['save_dir']+'hcrn_{}.pth'.format(epoch))
            torch.save(conv.state_dict(), config['dataset']['save_dir']+'conv_{}.pth'.format(epoch))
            print('Model Saved: Epoch {}'.format(epoch))
        
        if val_accuracy[-1] == max(val_accuracy):
            torch.save(hcrn.state_dict(), config['dataset']['save_dir']+'hcrn_best.pth')
            torch.save(conv.state_dict(), config['dataset']['save_dir']+'conv_best.pth')
            print('Model Saved: Best')

        print("Epoch {}: train loss: {:.3f}, val loss: {:.3f}".format(epoch, np.mean(batch_losses), sum(val_batch_loss)/len(val_loader)))
        train_loss.append(np.mean(batch_losses))
        val_loss.append(np.mean(val_batch_loss))



def calculate_metrics(pred, target):
    """
    Calculates the micro F1, macro F1, and accuracy scores for a set of predictions and ground truth labels.

    Args:
        pred: [np.ndarray] Predicted labels.
        target: [np.ndarray] Ground truth labels.

    Returns:
        dict: A dictionary containing the micro F1, macro F1, and accuracy scores.
    """

    micro_f1 = f1_score(target, np.argmax(pred, axis=1), average='micro')
    macro_f1 = f1_score(target, np.argmax(pred, axis=1), average='macro')
    accuracy = accuracy_score(target, np.argmax(pred, axis=1))
    recall = recall_score(target, np.argmax(pred, axis=1), average='macro')

    return {'micro/f1': micro_f1, 'macro/f1': macro_f1, 'accuracy': accuracy, 'recall': recall}



def evaluate(hcrn, conv, test_dataset, config, device):
    """
    Evaluates the performance of the trained HCRN model on the test dataset.

    Args:
        hcrn: [HCRNNetwork] HCRN model instance.
        conv: [nn.Module] GNN convolutional layer.
        test_dataset: [GraphQADataset] Test dataset object.
        config: [dict] Configuration dictionary containing hyperparameters and paths.
        device: [torch.device] Device for computations (CPU or GPU).

    Returns:
        None
    """

    # Extract hyperparameters from configuration
    lr = float(config['train']['lr'])
    batch_size = config['train']['batch_size']
    epochs = config['train']['num_epochs']
    in_feats = config['train']['in_feats']

    # Create data loader for the test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn,
                             num_workers=config['num_workers'])

    title = "lr{}-bs{}-epochs{}-test{}".format(lr, batch_size, epochs, len(test_loader))

    # Load the best saved models for evaluation
    hcrn.load_state_dict(torch.load(config['dataset']['save_dir'] + 'hcrn_best.pth'))
    conv.load_state_dict(torch.load(config['dataset']['save_dir'] + 'conv_best.pth'))

    # Initialize empty lists to store model predictions and targets
    model_result = []
    targets = []

    # Set models to evaluation mode
    hcrn.eval()
    conv.eval()

    print('Evaluating on test set...')

    # Iterate through batches in the test data loader
    for batch in test_loader:
        # Transfer data to chosen device (CPU or GPU)
        cls = batch['cls'].to(device)
        labels = batch['encoded_answer'].to(device)
        gbatch = batch['graphs'].to(device)
        nfeat = batch['ndata'].to(device)
        efeat = batch['edata'].to(device)

        val_batch_loss = 0
        with torch.no_grad():

            # Perform GNN forward pass
            res = conv(gbatch, nfeat, efeat)
            nodes = batch['num_n']
            r = res.cpu().detach().numpy()

            # Calculate graph representation (mean pooling)
            graph_mean = r
            graph_mean = np.array([np.mean(r[start:finish], axis=0) for start, finish in zip(nodes[:-1], nodes[1:])])

            # Handle specific logic for EdgeGATConv architecture
            if config['train']['gnn_arch'] == 'EdgeGATConv':
                graph_mean = np.mean(graph_mean, axis=1)  # mean across attention heads

            # Handle potential batch size mismatch during reshaping
            try:
                graph_mean = torch.from_numpy(np.reshape(graph_mean, (-1, 5, 3, in_feats))).to(device)
            except:
                print('ERROR', graph_mean.shape)
                continue

            # Perform HCRN forward pass
            outputs = hcrn(graph_mean, cls)

            # Apply softmax for probability distribution
            result = torch.nn.Softmax(dim=1)(outputs)
            model_result.extend(result.detach().cpu().numpy())
            targets.extend(labels.cpu().numpy())

    # Calculate evaluation metrics
    result = calculate_metrics(np.array(model_result), np.array(targets))

    # Print evaluation results for micro F1, macro F1, accuracy, and recall
    print("Test: "
          "micro f1: {:.3f} "
          "macro f1: {:.3f} "
          "accuracy: {:.3f}"
          "recall: {:3f}"
          .format(result['micro/f1'],
                  result['macro/f1'],
                  result['accuracy'],
                  result['recall']))

    le = joblib.load('label_encoder.joblib')

    # Generate classification report
    cr = classification_report(targets, np.argmax(model_result, axis=1), target_names=le.classes_, labels=targets)

    # Write classification report to file with title
    with open(config['dataset']['save_dir'] + 'report_' + title + '.txt', 'w') as f:
        f.write('Title\n\nClassification Report\n\n{}'.format(cr))

    print(cr)

