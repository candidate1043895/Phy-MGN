from GNN.partitioning.grid_partitioner import GridPartitioner
from GNN.partitioning.null_partitioner import NullPartitioner
from GNN.partitioning.range_partitioner import RangePartitioner
from GNN.partitioning.modularity_partitioner import ModularityPartitioner
from GNN.partitioning.rectgrid_partitioner import RectGridPartitioner
from GNN.partitioning.rectcentgrid_partitioner import RectCentGridPartitioner
from GNN.partitioning.rectgrid_samedom_partitioner import RectGridSameDomPartitioner
from GNN.partitioning.rect3Dgrid_partitioner import Rect3DGridPartitioner

def get_partitioner(params_dict):
    """
    returns a Partitioner object based on the parameters specified in params_dict

    params_dict is a dictionary. It must contain the name of a partition method,
    keyed by "partition_method", and the relevant parameters for that method.

    See the partitioner classes in GNN.partitioning for expected params
    and the config files under config_files for examples
    """
    partitioning_method = params_dict["partitioning_method"]
    if partitioning_method == "grid":
        return GridPartitioner(
            nrows=params_dict["nrows"],
            ncols=params_dict["ncols"],
            padding=params_dict["padding"],
        )
    elif partitioning_method == "rect":
        return RectGridPartitioner(
            nrows=params_dict["nrows"],
            ncols=params_dict["ncols"],
            padding=params_dict["padding"],
            cell_width=params_dict["cell_width"],
            cell_height=params_dict["cell_height"]
        )
    elif partitioning_method == "rectcent":
        return RectCentGridPartitioner(
            nrows=params_dict["nrows"],
            ncols=params_dict["ncols"],
            crust=params_dict["crust"],
            padding=params_dict["padding"],
            cell_width=params_dict["cell_width"],
            cell_height=params_dict["cell_height"]
        )
    elif partitioning_method == "rect_samedom":
        return RectGridSameDomPartitioner(
            nrows=params_dict["nrows"],
            ncols=params_dict["ncols"],
            nwidth=params_dict["nwidth"],
            nheight=params_dict["nheight"],
            padding=params_dict["padding"],
            cell_width=params_dict["cell_width"],
            cell_height=params_dict["cell_height"]
        )
    elif partitioning_method == "rect3D":
        return Rect3DGridPartitioner(
            nx=params_dict["nx"], # ncols
            ny=params_dict["ny"], # nrows
            nz=params_dict["nz"], # nplanes
            padding=params_dict["padding"],
            cell_width=params_dict["cell_width"],    # x
            cell_height=params_dict["cell_height"],   # y
            cell_length=params_dict["cell_length"]   # z
        )
    elif partitioning_method == "range":
        range_keys = [key for key in params_dict.keys() if key.startswith("range_")]
        range_keys = sorted(range_keys, lambda x: int(x.split("_")[1]))
        ranges = [params_dict[key] for key in range_keys]
        return RangePartitioner(ranges=ranges, padding=params_dict["padding"])
    elif partitioning_method == "modularity":
        return ModularityPartitioner(padding=params_dict["padding"])
    elif partitioning_method == "null":
        return NullPartitioner()
    else:
        raise ValueError("Unrecognized partitioning method: %s" % partitioning_method)
