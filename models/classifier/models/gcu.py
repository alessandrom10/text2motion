import torch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_max, scatter_mean
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch.nn import Sequential, Dropout, Linear, ReLU, InstanceNorm1d, Parameter


def MLP(channels, batch_norm=True):
    if batch_norm:
        return Sequential(*[Sequential(Linear(channels[i - 1], channels[i]), ReLU(), InstanceNorm1d(channels[i]))
                              for i in range(1, len(channels))])
    else:
        return Sequential(*[Sequential(Linear(channels[i - 1], channels[i]), ReLU()) for i in range(1, len(channels))])


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, nn, aggr='max', **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn

    def forward(self, x, edge_index):
        output_list = []
        for i in range(x.size(0)):
            curr_x = x[i]
            
            current_edge_index, _ = remove_self_loops(edge_index[i].to(x.device))
            current_edge_index, _ = add_self_loops(current_edge_index, num_nodes=curr_x.size(0))
            
            processed_output = self.propagate(current_edge_index, x=curr_x)
            output_list.append(processed_output)

        return torch.stack(output_list, dim=0)


    def message(self, x_i, x_j):
        return self.nn(torch.cat([x_i, (x_j - x_i)], dim=1))

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.out_channels)
        return aggr_out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GCU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, aggr='max'):
        super(GCU, self).__init__()
        self.edge_conv_tpl = EdgeConv(in_channels=in_channels, out_channels=out_channels // 2,
                                      nn=MLP([in_channels * 2, out_channels // 2, out_channels // 2]), aggr=aggr)
        self.mlp = MLP([out_channels // 2, out_channels])

    def forward(self, x, tpl_edge_index, geo_edge_index):
        x_tpl = self.edge_conv_tpl(x, tpl_edge_index)
        x_out = self.mlp(x_tpl)
        return x_out