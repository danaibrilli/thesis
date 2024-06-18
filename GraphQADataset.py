from select import select
import torch
from dgl.data import DGLDataset
from utils import get_key
import json
import pickle
import torch.nn.functional as F
import dgl
import os
import numpy as np
import joblib

def custom_collate_fn(batch):
    """Custom collate function for DataLoader to handle graph-related batch data."""
    ndata= get_key(batch, 'ndata')
    edata= get_key(batch, 'edata')
    num_n = np.cumsum(get_key(batch, 'num_n'))
    num_e = np.cumsum(get_key(batch, 'num_e'))

    simple_batch = batch

    graphs = [x['graphs'] for x in simple_batch]
    
    simple_batch = [{k: v for k, v in item.items() if k not in ['graphs', 'ndata', 'edata', 'num_n', 'num_e']} for item in batch]
    collated_batch = torch.utils.data.default_collate(simple_batch)
    collated_batch.update({
        'graphs': dgl.batch(graphs),
        'ndata': ndata,
        'edata': edata,
        'num_n': np.insert(num_n, 0, 0),
        'num_e': np.insert(num_e, 0, 0)
    })

    return collated_batch


class GraphQADataset(DGLDataset):
    def __init__(self, all_video_ids, all_question_ids, all_questions, 
                 all_answers, blind = True, use_sg = True, edge_feat = 'original',
                 scene_graphs_path = None, mode = 'train', device = None):
        
        """
        Initialize the GraphQADataset.

        Args:
            all_video_ids (list): List of video IDs.
            all_question_ids (list): List of question IDs.
            all_questions (list): List of questions.
            all_answers (list): List of answers.
            blind (bool): Flag for blind mode.
            use_sg (bool): Flag for using scene graphs.
            edge_feat (str): Type of edge feature.
            scene_graphs_path (str): Path to scene graphs.
            mode (str): Mode of dataset (train/val/test).
            device (torch.device): Device to be used.
        """

        self.all_video_ids = all_video_ids
        self.all_video_names = [vid + '.mp4' for vid in all_video_ids]
        self.all_question_ids = all_question_ids
        self.all_questions = all_questions
        self.all_answers = all_answers
        self.mode = mode
        self.device = device
        self.edge_feat = edge_feat
        self.filenames = '/ssd_data/agqa/questions/AGQA_balanced/'+ self.mode + '/'
        self.blind = blind
        self.use_sg = use_sg
        self.scene_graphs_path = scene_graphs_path

        #load label encoder.joblib
        self.le = joblib.load('/home/dbrilli/GNN_MLP/train/predicted_sgs/label_encoder.joblib')
        
        # Load additional data if using scene graphs
        if self.use_sg:
            self._load_scene_graph_data()
           

    def __len__(self):
        return len(self.all_question_ids)
    
    

    def __getitem__(self,idx):
        """
        Get a dataset item by index.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: Dictionary containing data for the given index.
        """

        video_name = self.all_video_names[idx].split('-')[0]
        video_idx = self.all_video_ids[idx]
        question_idx = self.all_question_ids[idx]
        question = self.all_questions[idx]
        answer = self.all_answers[idx]
        encoded_answer = self.le.transform([answer])[0]
        cls_path = os.path.join(self.filenames, f'{question_idx}_cls.pt')
        cls = torch.load(cls_path)

        if self.use_sg:
            graphs, ndata_clip, edata_clip, num_n, num_e = self._process_scene_graphs(video_name)
        else:
            graphs, ndata_clip, edata_clip, num_n, num_e = [], [], [], [], []

        return {
            'video_idx': video_idx,
            'question_idx': question_idx,
            'video_name': video_name,
            'answer': answer,
            'encoded_answer': torch.tensor(encoded_answer),
            'question': question,
            'cls': cls,
            'graphs': graphs,
            'ndata': ndata_clip,
            'edata': edata_clip,
            'num_n': num_n,
            'num_e': num_e
        }
    
    def _load_scene_graph_data(self):
        with open ('/home/dbrilli/GNN_MLP/preprocess/video_data_small_preprocessed.pickle', 'rb') as f:
            self.video_data = pickle.load(f)
        with open('/home/dbrilli/GNN_MLP/train/predicted_sgs/sg_lbs.pkl', 'rb') as f:
            (self.obj_lb, self.rel_lb) = pickle.load(f)
        if self.scene_graphs_path is not None:
            with open(self.scene_graphs_path, 'rb') as f:
                self.sgs = pickle.load(f)
        
    def _process_scene_graphs(self, video_name):
        """
        Process scene graphs for a given video.

        Args:
            video_name (str): Name of the video.

        Returns:
            tuple: Batch of graphs, node data, edge data, number of nodes, number of edges.
        """
        graphs = []
        selected_frames = self.video_data[video_name]['sel_frames_per_clip']
        ndata_clip = torch.empty((0, 151), dtype=torch.float32)
        edata_clip = torch.empty((0, 151 if self.edge_feat != 'original' else 51), dtype=torch.float32)
        num_n, num_e = [], []

        for clip in selected_frames:
            for frame in clip:
                objs, preds = self._scene_graph_from_frame(video_name, frame)
                frame_g, ndata, edata = self._graph_format(objs, preds)
                graphs.append(frame_g)
                num_n.append(frame_g.number_of_nodes())
                num_e.append(frame_g.number_of_edges())
                ndata_clip = torch.cat((ndata_clip, ndata), 0)
                edata_clip = torch.cat((edata_clip, edata), 0)

        return dgl.batch(graphs), ndata_clip, edata_clip, num_n, num_e


    def _scene_graph_from_frame(self, video_name, frame_id):
        """
        Generate a scene graph from a frame.

        Args:
            video_name (str): Name of the video.
            frame_id (int): Frame ID.

        Returns:
            tuple: Objects and predicates for the frame.
        """
        frame_id = self._frame_format(frame_id)
        triplets = self.sgs[video_name][f'{frame_id}.png']
        if len(triplets) > 40:
            triplets = triplets[:40]

        objs = [[x[0],x[1]] for x in triplets]
        preds = [x[2] for x in triplets]
        
        if not objs or not preds:
            objs, preds = [[0, 0]], [0]  # background

        return objs, preds
    

    def _graph_format(self,objs,preds):
        """
        Format the graph from objects and predicates.

        Args:
            objs (list): List of objects.
            preds (list): List of predicates.

        Returns:
            tuple: Graph, node data, edge data.
        """


        obj_nodes = list(set([ob for obj in objs for ob in obj]))
        nodes_feat = self.obj_lb.transform(obj_nodes)

        edges_init, edges_end = [], []
        obj2node = {obj: node for node, obj in enumerate(obj_nodes)}
        edge_feat = [self.rel_lb.transform([rel])[0] for [obj1, obj2], rel in zip(objs, preds)]


        for [obj1, obj2] in objs:
            edges_init.append(obj2node[obj1])
            edges_end.append(obj2node[obj2])

        g = dgl.graph((edges_init, edges_end), num_nodes=len(obj_nodes))
        ndata = torch.from_numpy(nodes_feat).float()
        edata = torch.from_numpy(np.array(edge_feat)).float()

        if self.edge_feat is not 'original':
            edata = F.pad(edata,(0,100),'constant',0)

        return g, ndata, edata
    
    def _frame_format(self,number):
        """
        Format the frame number.

        Args:
            number (int): Frame number.

        Returns:
            str: Formatted frame number.
        """
        return "{number:06}".format(number=int(number))
