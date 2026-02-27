import torch
from GNN.partitioning.partitioner import Partitioner
import numpy as np

class RectCentGridPartitioner(Partitioner):
    """
    Partitions a torch_geometric Dataset object into a rectangular spatial grid,
    including ghost nodes (padding) around each partition.
    """

    def __init__(self, nrows, ncols, padding, crust, cell_width, cell_height, **kwargs):
        assert isinstance(nrows, int) and nrows > 0
        assert isinstance(ncols, int) and ncols > 0
        assert padding >= 0
        self.nrows = nrows
        self.ncols = ncols
        self.crust = crust
        self.padding = padding
        self.cell_width = cell_width
        self.cell_height = cell_height
        super().__init__(padding=padding, **kwargs)

    def get_partition(self, data, num_edge_types=1):
        eps = 1e-5

        x_min, y_min = data.pos.min(dim=0).values
        x_max, y_max = data.pos.max(dim=0).values

        subdomain_width = (x_max + eps - x_min) / self.ncols
        subdomain_height = (y_max + eps - y_min) / self.nrows
        
        
        #print('cell_width', cell_width)
        #print('cell_height', cell_height)

        num_nodes = data.pos.shape[0]
        cells = []

        for i in range(self.ncols):
            x_start = x_min + i * subdomain_width
            x_end = x_min + (i + 1) * subdomain_width
            #print('x_start', x_start)
            #print('x_end', x_end)
            for j in range(self.nrows):
                y_start = y_min + j * subdomain_height
                y_end = y_min + (j + 1) * subdomain_height
                #print('y_start', y_start)
                #print('y_end', y_end)
                # Cell boundary including padding (ghost nodes)
                x_start_padded = x_start - self.padding * self.cell_width
                x_end_padded = x_end + self.padding * self.cell_width
                y_start_padded = y_start - self.padding * self.cell_height
                y_end_padded = y_end + self.padding * self.cell_height
                #print('padded', x_start_padded, x_end_padded, y_start_padded, y_end_padded)
                

                # Identify nodes within the padded cell
                padded_cell_mask = (
                    (data.pos[:, 0] >= x_start_padded)
                    & (data.pos[:, 0] < x_end_padded)
                    & (data.pos[:, 1] >= y_start_padded)
                    & (data.pos[:, 1] < y_end_padded)
                )
                padded_nodes_in_cell = torch.arange(num_nodes)[padded_cell_mask]

                # Nodes strictly within the padded cell (except the outermost nodes)
                internal_cell_mask = (
                    (data.pos[:, 0] >= x_start_padded + 1 * self.cell_width)
                    & (data.pos[:, 0] < x_end_padded - 1 * self.cell_width)
                    & (data.pos[:, 1] >= y_start_padded + 1 * self.cell_height)
                    & (data.pos[:, 1] < y_end_padded - 1 * self.cell_height)
                )
                internal_nodes_in_cell = torch.arange(num_nodes)[internal_cell_mask]
                internal_node_mask = torch.isin(padded_nodes_in_cell, internal_nodes_in_cell)

                # For each edge type, select edges connecting nodes within padded region
                edge_masks = []
                for k in range(num_edge_types):
                    edge_index = data[f"edge_index_{k}"]
                    src, tgt = edge_index

                    src_in_padded = torch.isin(src, padded_nodes_in_cell)
                    tgt_in_padded = torch.isin(tgt, padded_nodes_in_cell)

                    # Edges fully within padded region
                    edge_mask = src_in_padded & tgt_in_padded
                    edge_masks.append(edge_mask)

                if num_edge_types == 1:
                    edge_masks = edge_masks[0]
                
                cells.append((padded_nodes_in_cell, edge_masks, internal_node_mask))
        
                #print('internal_node_mask', np.shape(internal_node_mask))
                #print('padded_nodes_in_cell', np.shape(padded_nodes_in_cell))
                #print('internal_node_mask.sum().item() ', internal_node_mask.sum().item())
                print('OTHER padded_nodes_in_cell, padded_nodes_in_cell, internal_node_mask',padded_cell_mask.sum().item(), internal_node_mask.sum().item())
        # Take another subdomain in the centre:
        x_cent_start = x_min + (self.crust) * self.cell_width
        x_cent_end = x_max - (self.crust) * self.cell_width 
        
        y_cent_start = y_min + (self.crust) * self.cell_height
        y_cent_end = y_max - (self.crust) * self.cell_height
        
        
        
        # Identify nodes within the padded cell
        padded_cell_mask = (
            (data.pos[:, 0] >= x_cent_start)
            & (data.pos[:, 0] < x_cent_end + eps)
            & (data.pos[:, 1] >= y_cent_start)
            & (data.pos[:, 1] < y_cent_end + eps)
        )
        padded_nodes_in_cell = torch.arange(num_nodes)[padded_cell_mask]
        # Nodes strictly within the original cell (internal nodes)
        internal_cell_mask = (
            (data.pos[:, 0] >= x_cent_start + 1 * self.cell_width)
            & (data.pos[:, 0] < x_cent_end - 1 * self.cell_width)
            & (data.pos[:, 1] >= y_cent_start + 1 * self.cell_height)
            & (data.pos[:, 1] < y_cent_end - 1 * self.cell_height)
        )
        internal_nodes_in_cell = torch.arange(num_nodes)[internal_cell_mask]
        internal_node_mask = torch.isin(padded_nodes_in_cell, internal_nodes_in_cell)
        
        # For each edge type, select edges connecting nodes within padded region
        edge_masks = []
        for k in range(num_edge_types):
            edge_index = data[f"edge_index_{k}"]
            src, tgt = edge_index
            src_in_padded = torch.isin(src, padded_nodes_in_cell)
            tgt_in_padded = torch.isin(tgt, padded_nodes_in_cell)
            # Edges fully within padded region
            edge_mask = src_in_padded & tgt_in_padded
            edge_masks.append(edge_mask)
        if num_edge_types == 1:
            edge_masks = edge_masks[0]
        
        cells.append((padded_nodes_in_cell, edge_masks, internal_node_mask))
        
        #print('cent internal_node_mask', np.shape(internal_node_mask))
        #print('cent padded_nodes_in_cell', np.shape(padded_nodes_in_cell))
        #print('internal_node_mask.sum().item() cent', internal_node_mask.sum().item())
        print('Cent padded_nodes_in_cell, padded_nodes_in_cell, internal_node_mask',padded_cell_mask.sum().item(), internal_node_mask.sum().item())
        return cells

    def __str__(self):
        return f"rectcentgrid_ncols{self.ncols}_nrows{self.nrows}_crust{self.crust}_padding{self.padding}"
