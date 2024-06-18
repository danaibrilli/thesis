import torch 
import numpy as np
import json
import random

def get_key(data, key):
    """
    Extract and concatenate values associated with a key from a list of dictionaries.

    Args:
        data (list): List of dictionaries.
        key (str): Key to extract values for.

    Returns:
        torch.Tensor or np.ndarray: Concatenated values.
    """
    feat = [x[key] for x in data]
    if torch.is_tensor(feat[0]):
        feat = torch.cat(feat, dim=0)
    else:
        feat = np.array(feat)
        feat = np.concatenate(feat, axis=0)
    return feat


def remove_key(data, del_key):
    """
    Recursively remove a key from dictionaries within a list or a dictionary.

    Args:
        data (list or dict): List or dictionary to process.
        del_key (str): Key to remove.
    """
    if isinstance(data, dict):
        data.pop(del_key, None)
        for key in data:
            # Recursively call remove_key for each value
            remove_key(data[key], del_key)
    elif isinstance(data, list):
        for item in data:
            remove_key(item, del_key)

def invert_dict(d):
    """
    Invert keys and values of a dictionary.

    Args:
        d (dict): Dictionary to invert.

    Returns:
        dict: Inverted dictionary.
    """
    return {v: k for k, v in d.items()}


def load_vocab(path):
    """
    Load vocabulary from a JSON file and create inverted dictionaries.

    Args:
        path (str): Path to the JSON file.

    Returns:
        dict: Loaded vocabulary with added inverted dictionaries.
    """
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    return vocab

def shuffle(data, seed = 42):
    """
    Shuffle data with a fixed seed for reproducibility.

    Args:
        data (list or np.ndarray): Data to shuffle.
        seed (int): Seed for the random number generator.

    Returns:
        list or np.ndarray: Shuffled data.
    """
    idx = list(range(len(data)))
    random.seed(seed)
    random.shuffle(idx)
    # Ensure data is a numpy array for indexing
    data = np.array(data)
    return data[idx]
