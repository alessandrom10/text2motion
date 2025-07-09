import numpy as np
import torch
import networkx as nx
import sys
import os
import models.SKINNING
from models.SKINNING import skinnet
from torch_geometric.data import Data
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def build_skeleton_graph(bones_edges):
    """
    Build the full skeleton graph from bone edges.
    bones_edges: list of tuples [(bone_idx1, bone_idx2), ...]
    """
    G = nx.Graph()
    G.add_edges_from(bones_edges)
    return G

def build_mesh_graph(vertex_adjacency):
    """
    vertex_adjacency: list of (v1, v2) edges representing mesh connectivity
    """
    G_mesh = nx.Graph()
    G_mesh.add_edges_from(vertex_adjacency)
    return G_mesh

def graph_to_edge_index(graph):
    """
    Convert NetworkX graph edges to edge_index tensor for PyG / GNN
    Returns a torch.LongTensor of shape (2, num_edges)
    """
    edges = list(graph.edges())
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def normalize_skin_weights(weights):
    """
    Normalize skinning weights so each vertex's weights sum to 1
    weights: Tensor shape [num_vertices, num_bones]
    """
    weights = torch.relu(weights)  # Optional: ensure non-negative
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8
    normalized = weights / weights_sum
    return normalized

def linear_blend_skinning(vertices, skin_weights, bone_transforms):
    """
    vertices: Tensor [num_vertices, 3]
    skin_weights: Tensor [num_vertices, num_bones]
    bone_transforms: Tensor [num_bones, 4, 4] (homogeneous transforms)
    
    Returns deformed_vertices: Tensor [num_vertices, 3]
    """
    num_vertices, num_bones = skin_weights.shape
    vertices_h = torch.cat([vertices, torch.ones(num_vertices, 1)], dim=1)  # to homogeneous [x,y,z,1]
    deformed = torch.zeros_like(vertices_h)

    for b in range(num_bones):
        T = bone_transforms[b]  # [4,4]
        w = skin_weights[:, b].unsqueeze(1)  # [num_vertices,1]
        transformed = (vertices_h @ T.T)  # [num_vertices, 4]
        deformed += w * transformed

    return deformed[:, :3]  # drop homogeneous coord

# Main skinning pipeline

def skinning_pipeline(mesh_vertices, vertex_adjacency, bones_edges, bone_transforms, nearest_bone, use_Dg, use_Lf):
    """
    mesh_vertices: Tensor [num_vertices, 3]
    vertex_adjacency: list of (v1, v2)
    bones_edges: list of (bone_idx1, bone_idx2)
    bone_transforms: Tensor [num_bones, 4, 4]
    nearest_bone: int, number of nearest bones considered
    use_Dg, use_Lf: flags controlling input features (from skinning.py)
    """

    # Step 1: Build skeleton graph and compute MST
    skeleton_graph = build_skeleton_graph(bones_edges)
    mst_graph = nx.minimum_spanning_tree(skeleton_graph) # Extract MST

    # Step 2: Build mesh graph
    mesh_graph = build_mesh_graph(vertex_adjacency)

    # Step 3: Convert graphs to edge_index format for GNN
    tpl_edge_index = graph_to_edge_index(mst_graph)   # skeleton graph edges (non-MST)
    geo_edge_index = graph_to_edge_index(mesh_graph)      # mesh graph edges

    # Step 4: Prepare skin_input feature tensor (dummy example, real input depends on your data)
    # Usually contains geometric features, bone-related features per vertex
    # For now, let's create a placeholder tensor with the correct shape:
    num_vertices = mesh_vertices.shape[0]
    skin_input_dim = 3 + nearest_bone * 8  # max dimension according to skinning.py
    skin_input = torch.randn(num_vertices, skin_input_dim)  # Replace with real feature construction!

    data = Data(
        pos=mesh_vertices,                 # [num pvertices, 3]
        skin_input=skin_input,            # [num_vertices, skin_input_dim]
        tpl_edge_index=tpl_edge_index,    # [2, num_skeleton_edges] (non-MST edges)
        geo_edge_index=geo_edge_index,    # [2, num_mesh_edges]
        batch=torch.zeros(num_vertices, dtype=torch.long)  # if single mesh; otherwise build batch vector
    )

    # Step 6: Initialize SKINNET and predict skin weights
    model = skinnet(nearest_bone=nearest_bone, use_Dg=use_Dg, use_Lf=use_Lf)
    model.eval()  # set to eval mode if not training
    with torch.no_grad():
        skin_weights_raw = model(data)  # shape [num_vertices, nearest_bone]

    # Step 7: Normalize skin weights
    skin_weights = normalize_skin_weights(skin_weights_raw)

    # Step 8: Apply linear blend skinning
    deformed_vertices = linear_blend_skinning(mesh_vertices, skin_weights, bone_transforms)

    return deformed_vertices, skin_weights