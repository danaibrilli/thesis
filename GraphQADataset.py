from select import select
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import json
import pickle
import torch.nn.functional as F
import dgl
import os
import numpy as np
import joblib



class GraphQADataset(DGLDataset):
    def __init__(self, all_video_ids, all_question_ids, all_questions, 
                 all_answers, blind = True, use_sg = True, edge_feat = 'original',
                 scene_graphs_path = None, mode = 'train', device = None):
        
        
        self.all_video_ids = all_video_ids
        self.all_video_names = [vid + '.mp4' for vid in all_video_ids]
        self.all_question_ids = all_question_ids
        self.all_questions = all_questions
        self.all_answers = all_answers
        self.mode = mode
        self.device = device
        self.edge_feat = edge_feat

        #load label encoder.joblib
        self.le = joblib.load('/home/dbrilli/GNN_MLP/train/predicted_sgs/label_encoder.joblib')
        
        self.filenames = '/ssd_data/agqa/questions/AGQA_balanced/'+ self.mode + '/'
        
        self.blind = blind
        self.use_sg = use_sg
        self.scene_graphs_path = scene_graphs_path

        pwd = os.getcwd()
        print(pwd.split('/')[1])
       


        if self.use_sg is not None:
            with open ('/home/dbrilli/GNN_MLP/preprocess/video_data_small_preprocessed.pickle', 'rb') as f:
                self.video_data = pickle.load(f)
            with open('/home/dbrilli/GNN_MLP/train/predicted_sgs/sg_lbs.pkl', 'rb') as f:
                (self.obj_lb, self.rel_lb) = pickle.load(f)
            if self.scene_graphs_path is not None:
                with open(self.scene_graphs_path, 'rb') as f:
                    self.sgs = pickle.load(f)

    def __len__(self):
        return len(self.all_question_ids)
    

    def __getitem__(self,idx):

        video_name = self.all_video_names[idx].split('-')[0]
        video_idx = self.all_video_ids[idx]
        question_idx = self.all_question_ids[idx]

        question = self.all_questions[idx]
        answer = self.all_answers[idx]
        encoded_answer = self.le.transform([answer])[0]

        path = self.filenames +  question_idx + '_cls.pt'
       
        cls = torch.load(path)


        if self.use_sg:
            graphs = []
            selected_frames = self.video_data[video_name]['sel_frames_per_clip']
            ndata_clip = torch.empty((0, 151), dtype=torch.float32)
            if self.edge_feat is not 'original':
                edata_clip = torch.empty((0, 151), dtype=torch.float32)
            else:
                edata_clip = torch.empty((0, 51), dtype=torch.float32)
            num_n = []
            num_e = []
            
            for i,clip in enumerate(selected_frames):
                for i,frame in enumerate(clip):
                    objs, preds = self.scene_graph_from_frame(video_name, frame)
                    frame_g, ndata, edata = self.graph_format(objs,preds) 
                    graphs.append(frame_g)
                    num_nodes = frame_g.number_of_nodes()
                    num_n.append(num_nodes)
                    num_edges = frame_g.number_of_edges()
                    num_e.append(num_edges)
                    ndata_clip =  torch.cat((ndata_clip, ndata), 0)
                    edata_clip =  torch.cat((edata_clip, edata), 0)

            graphs = dgl.batch(graphs)
        else:
            graphs = []
            ndata_clip = []
            edata_clip = []
            num_n = []
            num_e = []

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

    def scene_graph_from_frame(self, video_name, frame_id):
        """
        Generate a scene graph from a frame.
        """
        frame_id = self.frame_format(frame_id)

        triplets = self.sgs[video_name][frame_id + '.png']
        if len(triplets) > 40:
            triplets = triplets[:40]

        objs = [[x[0],x[1]] for x in triplets]
        preds = [x[2] for x in triplets]
        
        if len(objs) == 0 or len(preds) == 0:
            print('empty graph')
            print(video_name, frame_id)
            print(triplets)
            objs = [[0,0]] #background
            preds = [0] #background

        return objs, preds
    

    def graph_format(self,objs,preds):
      

        obj_nodes = list(set([ob for obj in objs for ob in obj ]))
        #print(obj_nodes)
        nodes_feat = self.obj_lb.transform(obj_nodes)

        if len(preds) == 0:
            preds = ['0'] #background
            objs = ['0'] #background
        edges_init = []
        edges_end = []

        obj2node = {obj:node for node, obj in enumerate(obj_nodes)}

        edge_feat = []

        for [obj1,obj2],rel in zip(objs,preds):
        
            edges_init.append(obj2node[obj1])
            edges_end.append(obj2node[obj2])
            edge_feat.extend(self.rel_lb.transform([rel]))
        

        g = dgl.graph((edges_init, edges_end), num_nodes=len(obj_nodes))
        
        ndata = torch.from_numpy(nodes_feat).type(torch.FloatTensor)
        
        edata = torch.from_numpy(np.array(edge_feat)).type(torch.FloatTensor)
        
        if self.edge_feat is not 'original':
            edata = F.pad(edata,(0,100),'constant',0)
            #TODO:  parameterize the padding size

        return g, ndata, edata
    
    def frame_format(self,number):
        return "{number:06}".format(number=int(number))
