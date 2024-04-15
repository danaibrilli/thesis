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
from GraphQADataset_EGAT_1file import GraphQADataset
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
from scene_model import SCENE
from dgl.nn.pytorch.conv import GINEConv, EdgeGATConv
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
import warnings
import random
import argparse
import os
from inputUnit import HCRNNetwork
import shutil
import json


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    return vocab

def get_key(data, key):
    feat = [x[key] for x in data]
    if torch.is_tensor(feat[0]):
        feat = torch.cat(feat, dim=0)
    else:
        feat = np.array(feat)
        feat = np.concatenate(feat, axis=0)
    return feat


def remove_key(data, del_key):
    # Recursive --> )O(n) time + memory
    # checking if data is a dictionary or list
    if isinstance(data, dict):
        # if key is present in dictionary then remove it
        data.pop(del_key, None)
        #iterating over all the keys in the dictionary
        for key in data:
            # calling function recursively for nested dictionaries
            remove_key(data[key], del_key)
    elif isinstance(data, list):
        #iterating over all the items of the list
        for item in data:
            # calling function recursively for all elements of the list
            remove_key(item, del_key)


def custom_collate_fn(batch):
    #call the default collate for some of the keys
    ndata= get_key(batch, 'ndata')
    edata= get_key(batch, 'edata')
    num_n = get_key(batch, 'num_n')
    num_e = get_key(batch, 'num_e')

    num_n = np.cumsum(num_n)
    num_e = np.cumsum(num_e)
    simple_batch = batch

    graphs = [x['graphs'] for x in simple_batch]
    remove_key(simple_batch, 'graphs')
    remove_key(simple_batch, 'ndata')
    remove_key(simple_batch, 'edata')
    remove_key(simple_batch, 'num_n')
    remove_key(simple_batch, 'num_e')
    collated_batch = torch.utils.data.default_collate(simple_batch)
    collated_batch['graphs'] = dgl.batch(graphs)
    collated_batch['ndata'] = ndata
    collated_batch['edata'] = edata
    collated_batch['num_n'] = np.insert(num_n, 0, 0)
    collated_batch['num_e'] = np.insert(num_e, 0, 0)
    return collated_batch

def save_checkpoint(epoch, model, optimizer, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    time.sleep(10)
    torch.save(state, filename)

def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    predicted = predicted.detach().argmax(1)
    agreeing = (predicted == true)
    return agreeing

def calculate_metrics(output, targets):
    result = {}
    result['micro/f1'] = f1_score(targets, np.argmax(output, axis=1), average='micro')
    result['macro/f1'] = f1_score(targets, np.argmax(output, axis=1), average='macro')
    result['accuracy'] = accuracy_score(targets, np.argmax(output, axis=1))
    result ['recall'] = recall_score(targets, np.argmax(output, axis=1), average='macro')
    return result

def shuffle(data, seed = 42):
    idx = list(range(len(data)))
    random.seed(seed)
    random.shuffle(idx)
    data = data[idx]
    return data


print('started')
#with argparse read -conf argument
parser = argparse.ArgumentParser()
parser.add_argument('-conf', '--config', help='config file')

args = parser.parse_args()
config_path = args.config

#read config yml file
import yaml
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


with open ('/home/dbrilli/GNN_MLP/preprocess/data_small_preprocessed_all.pickle', 'rb') as f:
    data_small_train_all = pickle.load(f)

all_video_ids = np.array(data_small_train_all['all_video_ids'])
all_question_ids = np.array(data_small_train_all['all_question_ids'])
all_questions = np.array(data_small_train_all['all_questions'])
all_answers = np.array(data_small_train_all['all_answers'])

seed = config['seed']
all_video_ids = shuffle(all_video_ids, seed = seed)
all_question_ids = shuffle(all_question_ids, seed = seed)
all_questions = shuffle(all_questions, seed = seed)
all_answers = shuffle(all_answers, seed = seed)


#---- Here change for smaller - bigger exps -----
train_num = config['train']['train_num']
val_num = config['val']['val_num']
test_num = config['test']['test_num']

train_idx = test_num + train_num
val_idx = train_idx + val_num
test_idx = test_num

scene_graphs_path = config['sg_path']

train_answers = all_answers[test_idx:train_idx]
train_answers = {k: v for k, v in sorted(dict(zip(*np.unique(train_answers, return_counts=True))).items(), key=lambda item: item[1])}

val_answers = all_answers[train_idx:val_idx]
val_answers = {k: v for k, v in sorted(dict(zip(*np.unique(val_answers, return_counts=True))).items(), key=lambda item: item[1])}

test_answers = all_answers[:test_idx]
test_answers = {k: v for k, v in sorted(dict(zip(*np.unique(test_answers, return_counts=True))).items(), key=lambda item: item[1])}
edge_feat_type = config['train']['edge_feat_type']

train_dataset = GraphQADataset(all_video_ids[test_idx:train_idx], all_question_ids[test_idx:train_idx], all_questions[test_idx:train_idx], 
                 all_answers[test_idx:train_idx], blind = True, use_sg = True, 
                 scene_graphs_path = scene_graphs_path, mode = 'train', edge_feat=edge_feat_type)

val_dataset = GraphQADataset(all_video_ids[train_idx:val_idx], all_question_ids[train_idx:val_idx], all_questions[train_idx:val_idx], 
                 all_answers[train_idx:val_idx], blind = True, use_sg = True, 
                 scene_graphs_path = scene_graphs_path, mode = 'train', edge_feat=edge_feat_type)

test_dataset = GraphQADataset(all_video_ids[:test_idx], all_question_ids[:test_idx], all_questions[:test_idx], 
                 all_answers[:test_idx], blind = True, use_sg = True, 
                 scene_graphs_path = scene_graphs_path, mode = 'train', edge_feat=edge_feat_type)

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))


WANDB = config['use_wandb']
lr, batch_size, epochs = float(config['train']['lr']), config['train']['batch_size'], config['train']['num_epochs']
node_feat_shape = config['train']['node_feat_shape']
edge_feat_shape = config['train']['edge_feat_shape']
in_feats =  config['train']['in_feats'] #was 256
out_feats = config['train']['out_feats'] #was 171
num_layers = config['train']['num_layers']
dropout = config['train']['dropout']
path = config['dataset']['save_dir']
mlp_in = config['train']['mlp_in_feats']
hidden_sizes = config['train']['hidden_size']
attention_heads = config['train']['attention_heads']

#if path does not exist create it
if not os.path.exists(path):
    os.makedirs(path)

shutil.copy(config_path, path+'config.yml')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=config['num_workers']) #was 3
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=config['num_workers'])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=config['num_workers'])

device = torch.device(f"cuda:{config['gpu_id']}") if torch.cuda.is_available() else torch.device('cpu')
print(device)

graph_feat_size = config['train']['in_feats']
hidden_feats = 195
attention_heads = config['train']['attention_heads']

#GNN Architecture
if config['train']['gnn_arch'] == 'EdgeGATConv':
    conv = EdgeGATConv(node_feat_shape, edge_feat_shape, in_feats, attention_heads, allow_zero_in_degree=True)
elif config['train']['gnn_arch'] == 'GINE2L':
    conv = GINE_GNN(node_feat_shape, hidden_feats=hidden_feats, out_feats=graph_feat_size)
elif config['train']['gnn_arch'] == 'EGAT2L':
    conv = EdgeGATGNN(node_feat_shape, edge_feat_shape, hidden_feats=hidden_feats, out_feats=graph_feat_size)
elif config['train']['gnn_arch'] == 'GINE_EGAT':
    conv = GINE_EGAT_GNN(in_node_feats=node_feat_shape, in_edge_feats=node_feat_shape, hidden_feats=hidden_feats, gine_out_feats=hidden_feats, egat_out_feats=graph_feat_size, num_heads=attention_heads)
elif config['train']['gnn_arch'] == 'SCENE':
    conv = SCENE(node_feat_shape, edge_feat_shape, num_heads=attention_heads, hidden_size=hidden_feats, out_feats=graph_feat_size)

#conv = GINEConv(nn.Linear(node_feat_shape,in_feats))
conv.to(device)

vocab_json = '/ssd_data/agqa/questions/csvs/balanced/tgif-qa_frameqa_vocab-balanced.json'
print('loading vocab from %s' % (vocab_json))
vocab = load_vocab(vocab_json)

module_dim = 512
word_dim = 768
k_max_graph_level = 3
k_max_clip_level = 5
spl_resolution = 1


hcrn = HCRNNetwork(module_dim = module_dim, graph_embedding_dim=256, word_dim = word_dim, k_max_graph_level = k_max_graph_level, 
                k_max_clip_level = k_max_clip_level, spl_resolution=1, vocab = vocab,
                dropout_style = 1, dropout_prob= 0.1, crn_dropout_prob=0.1)


hcrn.to(device)

optim = torch.optim.AdamW(list(hcrn.parameters()) + list(conv.parameters()), lr=lr)
criterion = nn.CrossEntropyLoss()

hcrn_params = sum(p.numel() for p in hcrn.parameters())
conv_params = sum(p.numel() for p in conv.parameters())
total_params = hcrn_params + conv_params


print('HCRN params: ', hcrn_params) 
print('Conv params: ', conv_params)
print('Total params: ', total_params)



train_loss = []
val_loss = []
val_micro = []
val_macro = []
val_recall = []
val_accuracy = []

if WANDB:
    wandb.init(
        # set the wandb project where this run will be logged
        project="SG_GNN_HCRN",
        
        # track hyperparameters and run metadata
        config=config
    )



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

        #R87CC.mp4 000157
        #GBZAK.mp4 000194
        #MQ4YR.mp4 000387
        #MQ4YR.mp4 000039

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
        
            try: #-1 is for batch size (not perfect division)
                graph_mean = torch.from_numpy(np.reshape(graph_mean, (-1, 5, 3, in_feats))).to(device)
            except:
                print('ERROR', graph_mean.shape)
                continue

          
            outputs = hcrn(graph_mean, cls)
            
            vloss = criterion(outputs.type(torch.float), labels.type(torch.long))
            val_batch_loss.append(vloss.item()) 
            model_result.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    #print(f'SHAPE: { len(model_result)}, {len(targets)}')
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



title = "lr{}-bs{}-epochs{}-train{}-val{}-test{}".format(lr, batch_size, epochs, len(train_loader), len(val_loader), len(test_loader))



#for testing load best model
hcrn.load_state_dict(torch.load(config['dataset']['save_dir']+'hcrn_best.pth'))
conv.load_state_dict(torch.load(config['dataset']['save_dir']+'conv_best.pth'))

model_result = []
targets = []

hcrn.eval()
conv.eval()
print('into test')
for batch in test_loader:
    cls = batch['cls'].to(device)
    labels = batch['encoded_answer'].to(device)
    gbatch = batch['graphs'].to(device)
    
    nfeat = batch['ndata'].to(device)
    efeat = batch['edata'].to(device)

    val_batch_loss = 0
    with torch.no_grad():
    
        res = conv(gbatch, nfeat, efeat)
        nodes = batch['num_n']
        r = res.cpu().detach().numpy()

        graph_mean = r
        graph_mean = np.array([np.mean(r[start:finish], axis=0) for start, finish in zip(nodes[:-1], nodes[1:])])
        
        if config['train']['gnn_arch'] == 'EdgeGATConv':
            graph_mean = np.mean(graph_mean, axis=1) #mean across attention heads
        
        try: #-1 is for batch size (not perfect division)
            graph_mean = torch.from_numpy(np.reshape(graph_mean, (-1, 5, 3, in_feats))).to(device)
        except:
            print('ERROR', graph_mean.shape)
            continue

        outputs = hcrn(graph_mean, cls)

        result = torch.nn.Softmax(dim=1)(outputs)
        model_result.extend(result.detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())

result = calculate_metrics(np.array(model_result), np.array(targets))
print("Test: "
        "micro f1: {:.3f} "
        "macro f1: {:.3f} "
        "accuracy: {:.3f}"
        "recall: {:3f}" .format(result['micro/f1'],
                                    result['macro/f1'],
                                    result['accuracy'],
                                    result['recall']))
le = joblib.load('label_encoder.joblib')
cr = classification_report(targets, np.argmax(model_result, axis=1), target_names=le.classes_, labels = targets)
f = open(config['dataset']['save_dir']+'report_'+title+'.txt', 'w')
f.write('Title\n\nClassification Report\n\n{}'.format(cr))
f.close()
print(cr)
