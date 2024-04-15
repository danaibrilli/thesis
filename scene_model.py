from residual_egat import ResidualEdgeGATConv
import torch
import torch.nn as nn
import torch.nn.functional as F



class SCENE(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, num_heads=4, hidden_size=128, out_feats=256):
        super(SCENE, self).__init__()

        # Graph convolution layers
        #print("node_in_feats", node_in_feats, "edge_in_feats", edge_in_feats, "num_heads", num_heads, "hidden_size", hidden_size)
        self.conv1 = ResidualEdgeGATConv(node_in_feats, hidden_size, num_heads=num_heads, edge_feats=edge_in_feats)
        self.conv2 = ResidualEdgeGATConv(hidden_size, hidden_size, num_heads=num_heads)
        self.conv3 = ResidualEdgeGATConv(hidden_size, hidden_size, num_heads=num_heads)
        self.conv4 = ResidualEdgeGATConv(hidden_size, out_feats, num_heads=num_heads)

        #

    def forward(self, g, node_feats, edge_feats):
        # Applying graph convolutions
        h = F.relu(self.conv1(g, node_feats, edge_feats))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))

        #mean over attention heads

        return h