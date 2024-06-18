import torch
import torch.nn as nn
import itertools
import numpy as np

class CRN(nn.Module):
    def __init__(self, module_dim, num_objects, max_subset_size, gating=False, spl_resolution=1,
                 dropout_style=0, crn_dropout_prob=0.10):
        """
        Initialize the Composition Relation Network (CRN) module.

        Args:
            module_dim (int): Dimensionality of input and output features.
            num_objects (int): Number of objects in the input.
            max_subset_size (int): Maximum subset size for relations.
            gating (bool): Whether to use gating mechanism.
            spl_resolution (int): Resolution for subsampling.
            dropout_style (int): Dropout style (0 or 1).
            crn_dropout_prob (float): Dropout probability for CRN.
        """
        super(CRN, self).__init__()
        self.module_dim = module_dim
        self.gating = gating

        self.k_objects_fusion = nn.ModuleList()
        if self.gating:
            self.gate_k_objects_fusion = nn.ModuleList()
        # Initialize fusion layers for different subset sizes
        for i in range(min(num_objects, max_subset_size + 1), 1, -1):
            if dropout_style == 0:
                self.k_objects_fusion.append(nn.Linear(2 * module_dim, module_dim))
                if self.gating:
                    self.gate_k_objects_fusion.append(nn.Linear(2 * module_dim, module_dim))
            elif dropout_style == 1:
                # NEW DROPOUT
                self.k_objects_fusion.append(nn.Sequential(nn.Dropout(crn_dropout_prob), 
                                                            nn.Linear(2 * module_dim, module_dim)))
                if self.gating:
                    self.gate_k_objects_fusion.append(nn.Sequential(nn.Dropout(crn_dropout_prob), 
                                                                     nn.Linear(2 * module_dim, module_dim)))
                                    
        self.spl_resolution = spl_resolution
        self.activation = nn.ELU()
        self.max_subset_size = max_subset_size

    def forward(self, object_list, cond_feat):
        """
        Forward pass of the CRN module.

        Args:
            object_list (list of Tensor): List of tensors or vectors.
            cond_feat (Tensor): Conditioning feature.

        Returns:
            list: List of output objects.
        """
        scales = [i for i in range(len(object_list), 1, -1)]
        relations_scales = []
        subsample_scales = []
        # Generate relation sets for different scales
        for scale in scales:
            relations_scale = self.relationset(len(object_list), scale)
            relations_scales.append(relations_scale)
            subsample_scales.append(min(self.spl_resolution, len(relations_scale)))

        crn_feats = []
        if len(scales) > 1 and self.max_subset_size == len(object_list):
            start_scale = 1
        else:
            start_scale = 0
        for scaleID in range(start_scale, min(len(scales), self.max_subset_size)):
            idx_relations_randomsample = np.random.choice(len(relations_scales[scaleID]),
                                                          subsample_scales[scaleID], replace=False)
            mono_scale_features = 0
            for id_choice, idx in enumerate(idx_relations_randomsample):
                clipFeatList = [object_list[obj].unsqueeze(1) for obj in relations_scales[scaleID][idx]]
                # g_theta
                g_feat = torch.cat(clipFeatList, dim=1).mean(1)
                if len(g_feat.size()) == 2:
                    h_feat = torch.cat((g_feat, cond_feat), dim=-1)
                elif len(g_feat.size()) == 3:
                    cond_feat_repeat = cond_feat.repeat(1, g_feat.size(1), 1)
                    h_feat = torch.cat((g_feat, cond_feat_repeat), dim=-1)
                if self.gating:
                    h_feat = self.activation(self.k_objects_fusion[scaleID](h_feat)) * torch.sigmoid(
                        self.gate_k_objects_fusion[scaleID](h_feat))
                else:
                    h_feat = self.activation(self.k_objects_fusion[scaleID](h_feat))
                mono_scale_features += h_feat
            crn_feats.append(mono_scale_features / len(idx_relations_randomsample))
        return crn_feats

    def relationset(self, num_objects, num_object_relation):
        """
        Generate relation sets for given number of objects and relation size.

        Args:
            num_objects (int): Number of objects.
            num_object_relation (int): Number of objects in each relation.

        Returns:
            list: List of tuples representing relations.
        """
        return list(itertools.combinations([i for i in range(num_objects)], num_object_relation))