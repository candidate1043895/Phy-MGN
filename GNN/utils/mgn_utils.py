from torch import tensor, cat, int64, is_tensor
from torch.nn.functional import one_hot
from numpy import ones, squeeze, diff, tile, concatenate
from numpy.random import normal
import numpy as np


def process_node_window(
    node_data,
    node_coordinates=None,
    node_types=None,
    input_type=None,
    apply_onehot=False,
    onehot_classes=0,
    inlet_velocity=None,
):

    """
    Concatenates with node features with one-hot encoding of node types

    node_data: time x node x feature array
    node_coordinates: node x dimension array; do not use if node_data already has preprocessed node coordinate information
    node_types: node x one-hot dim array; do not use if node_data already has preprocessed node type information
    apply_onehot: boolean
    onehot_classes: integer
    #inlet_velocity: float; we'll use it as a node feature for now, unless there's a better place to use it

    """
    num_nodes = node_data.shape[1]
    
    if input_type == "velocity":
        node_data = node_data[:, :, :2]
    elif input_type == "velocity & density":
        # input velocity density
        node_data = node_data[:, :, :3]
    elif input_type == "velocity & density & pressure":
        # input velocity density pressure
        node_data = node_data[:, :, :4]
    elif input_type == "velocity & density & B":
        # input velocity density
        node_data = node_data[:, :, [0, 1, 2, 5, 6]]
    elif input_type is None:
        raise ValueError("input_type is None — must be one of: 'velocity', 'velocity & density', etc.")
        
    node_types_ = (
        one_hot(tensor(node_types, dtype=int64).flatten(), onehot_classes)
        if apply_onehot
        else node_types
    )
    if hasattr(node_types_, "device"):
        node_types_ = node_types_.to(node_data.device)
    node_data = [node_data.transpose(1, 0).reshape(num_nodes, -1)]
    
    if node_coordinates is not None:
        node_data.append(
            node_coordinates
            if is_tensor(node_coordinates)
            else tensor(node_coordinates)
        )

    if node_types_ is not None:
        node_data.append(node_types_ if is_tensor(node_types_) else tensor(node_types_))

    if inlet_velocity is not None:
        node_data.append(tensor(inlet_velocity * ones((num_nodes, 1))))
    
    if len(node_data) > 1:
        return cat(node_data, axis=-1)
    
    return node_data[0]


def get_sample(
    dataset,
    source_node_idx,
    time_idx,
    window_length=5,
    output_type="acceleration",
    input_type="velocity & density", 
    pres_history_flag=False, 
    noise_sd=None,
    noise_gamma=0.1,
    shift_source=False,
):

    """
    Returns position data window (with noise) and output velocity

    source_node_idx: source node indices
    time_idx: current time index
    window_length: input window length
    output_type: output type; one of 'state', 'velocity','acceleration', 'acceleration & pressure' or 'velocity & pressure'
    noise_sd: noise standard deviation
    noise_gamma: noise gamma (see noise details in arXiv:2010.03409)
    shift_source: if True, shift input source nodes ahead by one timestep; noise is not included
    """
    
    
    mask = ones(dataset.shape[1], dtype=bool)
    mask[source_node_idx] = False
    
    #only feed the first 2 elements velocity
    node_data = dataset[time_idx : (time_idx + window_length), :, :].copy()

    # for sources, shift window ahead by 1 timepoint, do not add noise (masked below)
    # also need to update MeshGraphNets.py update_function and rollout functions (and loss?)
    # to do: add a config parameter to turn this off
    if shift_source:
        node_data[:, source_node_idx, :] = dataset[
            (time_idx + 1) : (time_idx + window_length + 1), source_node_idx, :
        ].copy()

    # compute output
    if output_type == "acceleration":
        outputs = dataset[
            (time_idx + window_length - 2) : (time_idx + window_length + 1), :, :
        ]
        outputs = squeeze(diff(diff(outputs, axis=0), axis=0), axis=0)
    elif output_type == "velocity":
        outputs = dataset[
            (time_idx + window_length - 1) : (time_idx + window_length + 1), :, :
        ]
        outputs = squeeze(diff(outputs, axis=0), axis=0)
    elif output_type == "acceleration & pressure":
        acc = dataset[
            (time_idx + window_length - 2) : (time_idx + window_length + 1), :, :2
        ]
        acc = squeeze(diff(diff(acc, axis=0), axis=0), axis=0)
        pres = dataset[time_idx + window_length, :, 2]
        outputs = concatenate([acc, pres[:, None]], axis=1)

    elif output_type == "velocity & pressure":
        vel = dataset[
            (time_idx + window_length - 1) : (time_idx + window_length + 1), :, :2
        ]
        pres = dataset[time_idx + window_length, :, 2]
        vel = squeeze(diff(vel, axis=0), axis=0) 
        outputs = concatenate([vel, pres[:, None]], axis=1)

    elif output_type == "velocity & density & pressure":
        # input velocity and density, output acceleratioin and d(rho)/dt --then Euler integrate
        vel = dataset[
            (time_idx + window_length - 1) : (time_idx + window_length + 1), :, :2
        ]
        dens = dataset[
            (time_idx + window_length - 1) : (time_idx + window_length + 1), :, 2
        ]
        
        pres = dataset[time_idx + window_length, :, 3]
        pres_history = dataset[time_idx + window_length -1, :, 3]
        vel = squeeze(diff(vel, axis=0), axis=0) 
        dens = squeeze(diff(dens, axis=0), axis=0)
        
        if not pres_history_flag:
            outputs = concatenate([vel,dens[:, None], pres[:, None]], axis=1)
        elif pres_history_flag:
            outputs = concatenate([vel,dens[:, None], pres[:, None], pres_history[:, None]], axis=1)
            
    elif output_type == "velocity & density & pressure & B":
        vel = dataset[
            (time_idx + window_length - 1) : (time_idx + window_length + 1), :, :2
        ]
        dens = dataset[
            (time_idx + window_length - 1) : (time_idx + window_length + 1), :, 2
        ]
        
        pres = dataset[time_idx + window_length, :, 3]
        
        B = dataset[
            (time_idx + window_length - 1) : (time_idx + window_length + 1), :, 4:6
        ]
        pres_history = dataset[time_idx + window_length -1, :, 3]
        vel = squeeze(diff(vel, axis=0), axis=0)  # U, V
        dens = squeeze(diff(dens, axis=0), axis=0)
        B = squeeze(diff(B, axis=0), axis=0) # B_x, B_y
        if not pres_history_flag:
            outputs = concatenate([vel,dens[:, None], pres[:, None], B], axis=1)
        elif pres_history_flag:
            outputs = concatenate([vel,dens[:, None], pres[:, None], pres_history[:, None], B], axis=1)
            
    
        
    elif output_type == "dens_dt" or output_type == "pres_dt":  
        state = dataset[
            (time_idx + window_length - 1) : (time_idx + window_length + 1), :, :
        ]
        print('time_idx', time_idx)
        print('state shape', state.shape)
        state_np = np.asarray(state)
        diff_arr = np.diff(state_np, axis=0)
        print("STATE SHAPE:", state_np.shape, "DIFF SHAPE:", diff_arr.shape)
        outputs = squeeze(diff(state, axis=0), axis=0)
        
    elif output_type == "dens" or output_type == "pres":  
        outputs = dataset[time_idx + window_length, :, :].copy()
        
    else:
        outputs = dataset[time_idx + window_length, :, :].copy()
        
        
    

    # add noise to position and output
    if noise_sd is not None:
        noise = tile(noise_sd, (node_data.shape[1], 1))
        noise = normal(0, noise)
        # input noise
        
        input_features = node_data[-1].shape[1]
        node_data[-1][mask] += noise[:, :input_features][mask]
        # output adjustment

        if output_type == "acceleration":
            # acceleration_p = x_{t+1} - 2 * x_{t} + x_{t-1} - 2 * noise
            # acceleration_v = x_{t+1} - 2 * x_{t} + x_t{-1} - noise
            # adjustment = 2 * gamma * noise + (1-gamma) * noise = noise * (1 + gamma)
            outputs[mask] -= (1 + noise_gamma) * noise[mask]
        elif output_type == "velocity":
            # velocity_adj = x_{t+1} - (x_{t} + noise)
            outputs[mask] -= noise[mask]
        elif output_type == "acceleration & pressure":
            # acceleration_p = x_{t+1} - 2 * x_{t} + x_{t-1} - 2 * noise
            # acceleration_v = x_{t+1} - 2 * x_{t} + x_t{-1} - noise
            # adjustment = 2 * gamma * noise + (1-gamma) * noise = noise * (1 + gamma)
            outputs[mask] -= (1 + noise_gamma) * noise[mask]

        elif output_type in ("velocity & pressure", "velocity & density & pressure", "velocity & density & pressure & B"):
            # velocity_adj = x_{t+1} - (x_{t} + noise)
            outputs[mask] -= noise[mask]
            #just make sure noise_sdt for pressure is 0---so nothing for pressure 
       
        # Nothing for states

        # else: nothing for state
    #vut pressure from node_data dont feed pressure to MGN, it is the output
    if input_type == "velocity":
        node_data = node_data[:, :, :2]
    
    elif input_type == "velocity & density":
        # input velocity density
        node_data = node_data[:, :, :3]
    elif input_type == "velocity & density & pressure":
        # input velocity density pressure
        node_data = node_data[:, :, :4]
    elif input_type == "velocity & density & B":
        node_data = node_data[:, :, [0, 1, 2, 4, 5]]  
    else:
        node_data = node_data
    
              

    return node_data, outputs

