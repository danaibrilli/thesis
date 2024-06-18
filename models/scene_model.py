from models.residual_egat import ResidualEdgeGATConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class SCENE(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, num_heads=4, hidden_size=128, out_feats=256):
        """
        Initialize the SCENE model.

        Args:
            node_in_feats (int): Number of input features for the nodes.
            edge_in_feats (int): Number of input features for the edges.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            hidden_size (int, optional): Hidden size for the graph convolution layers. Defaults to 128.
            out_feats (int, optional): Output features size. Defaults to 256.
        """
        super(SCENE, self).__init__()

        # Graph convolution layers
        self.conv1 = ResidualEdgeGATConv(node_in_feats, hidden_size, num_heads=num_heads, edge_feats=edge_in_feats)
        self.conv2 = ResidualEdgeGATConv(hidden_size, hidden_size, num_heads=num_heads)
        self.conv3 = ResidualEdgeGATConv(hidden_size, hidden_size, num_heads=num_heads)
        self.conv4 = ResidualEdgeGATConv(hidden_size, out_feats, num_heads=num_heads)

    def forward(self, g, node_feats, edge_feats):
        """
        Forward pass for the SCENE model.

        Args:
            g (DGLGraph): The input graph.
            node_feats (torch.Tensor): Node features.
            edge_feats (torch.Tensor): Edge features.

        Returns:
            torch.Tensor: Output features after graph convolutions.
        """
        # Apply the first graph convolution layer with ReLU activation
        h = F.relu(self.conv1(g, node_feats, edge_feats))
        # Apply the second graph convolution layer with ReLU activation
        h = F.relu(self.conv2(g, h))
        # Apply the third graph convolution layer with ReLU activation
        h = F.relu(self.conv3(g, h))
        # Apply the fourth graph convolution layer with ReLU activation
        h = F.relu(self.conv4(g, h))

        return h