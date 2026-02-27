import torch
from GNN.partitioning.partitioner import Partitioner

class Rect3DGridPartitioner(Partitioner):
    """
    Partitions a torch_geometric Dataset object into a rectangular spatial grid,
    including ghost nodes (padding) around each partition.
    """

    def __init__(self, nx, ny, nz, padding, cell_width, cell_height, cell_length, **kwargs):
        assert isinstance(nx, int) and nx > 0
        assert isinstance(ny, int) and ny > 0
        assert isinstance(nz, int) and nz > 0
        assert padding >= 0
        self.nx = nx   # no of columns
        self.ny = ny   # no of rows
        self.nz = nz   # no of plan slices
        self.padding = padding
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.cell_length = cell_length
        super().__init__(padding=padding, **kwargs)

    def get_partition(self, data, num_edge_types=1):
        eps = 1e-6

        x_min, y_min, z_min = data.pos.min(dim=0).values
        x_max, y_max, z_max = data.pos.max(dim=0).values
        

        subdomain_width = (x_max + eps - x_min) / self.ny
        subdomain_height = (y_max + eps - y_min) / self.nx
        subdomain_length = (z_max + eps - z_min) / self.nz
        
        
        #print('cell_width', cell_width)
        #print('cell_height', cell_height)

        num_nodes = data.pos.shape[0]
        cells = []

        for i in range(self.nx):
            x_start = x_min + i * subdomain_width
            x_end = x_min + (i + 1) * subdomain_width
            #print('x_start', x_start)
            #print('x_end', x_end)
            for j in range(self.ny):
                y_start = y_min + j * subdomain_height
                y_end = y_min + (j + 1) * subdomain_height
                #print('y_start', y_start)
                #print('y_end', y_end)
                for k in range(self.nz):
                    z_start = z_min + j * subdomain_length
                    z_end = z_min + (j + 1) * subdomain_length
                    # Cell boundary including padding (ghost nodes)
                    x_start_padded = x_start - self.padding * self.cell_width
                    x_end_padded = x_end + self.padding * self.cell_width
                    y_start_padded = y_start - self.padding * self.cell_height
                    y_end_padded = y_end + self.padding * self.cell_height
                    z_start_padded = z_start - self.padding * self.cell_length
                    z_end_padded = z_end + self.padding * self.cell_length
                    #print('padded', x_start_padded, x_end_padded, y_start_padded, y_end_padded)
                

                    # Identify nodes within the padded cell
                    padded_cell_mask = (
                        (data.pos[:, 0] >= x_start_padded)
                        & (data.pos[:, 0] < x_end_padded)
                        & (data.pos[:, 1] >= y_start_padded)
                        & (data.pos[:, 1] < y_end_padded)
                        & (data.pos[:, 2] >= z_start_padded)
                        & (data.pos[:, 2] < z_end_padded)
                    )
                    padded_nodes_in_cell = torch.arange(num_nodes)[padded_cell_mask]
    
                    # Nodes strictly within the original cell (internal nodes)
                    internal_cell_mask = (
                        (data.pos[:, 0] >= x_start)
                        & (data.pos[:, 0] < x_end)
                        & (data.pos[:, 1] >= y_start)
                        & (data.pos[:, 1] < y_end)
                        & (data.pos[:, 2] >= z_start)
                        & (data.pos[:, 2] < z_end)
                    )
                    internal_nodes_in_cell = torch.arange(num_nodes)[internal_cell_mask]
                    internal_node_mask = torch.isin(padded_nodes_in_cell, internal_nodes_in_cell)
    
                    # For each edge type, select edges connecting nodes within padded region
                    edge_masks = []
                    for l in range(num_edge_types):
                        edge_index = data[f"edge_index_{l}"]
                        src, tgt = edge_index
    
                        src_in_padded = torch.isin(src, padded_nodes_in_cell)
                        tgt_in_padded = torch.isin(tgt, padded_nodes_in_cell)
    
                        # Edges fully within padded region
                        edge_mask = src_in_padded & tgt_in_padded
                        edge_masks.append(edge_mask)
    
                    if num_edge_types == 1:
                        edge_masks = edge_masks[0]
    
                    cells.append((padded_nodes_in_cell, edge_masks, internal_node_mask))

        return cells

    def __str__(self):
        return f"rectgrid_ncols{self.nx}_nrows{self.ny}_nplanes{self.nz}_padding{self.padding}"