from importlib import import_module
import os
from sys import path
import time
start_time = time.time()
print('start time', start_time)
import random
import numpy as np
import tqdm
import datetime
from pathlib import Path
from functools import partial


import torch
#data.DataLoader is deprecated, changed to loader.DataLoader
#from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

#########
# local application imports
# get path to root of the project
mgn_code_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
path.append(mgn_code_dir)

from GNN.ModelConfig.ConfigMGN import ModelConfig
from GNN.MeshGraphNets import MeshGraphNets
from GNN.MultiMeshGraphNets import MultiMeshGraphNets
from GNN.utils.train_utils import (
    train_epoch,
    train_epoch_MSE,
    train_epoch_LBFGS,
    val_epoch,
    save_model,
    load_model,
    load_model_train_vars,
    get_lr_scheduler,
    interval_actions,
    setup_sbp_train_loader,
)
from GNN.utils.utils import get_delta_minute
from GNN.utils.data_utils import BoundedDistributedSampler
from GNN.utils.loss_utils import PINNLoss_alw, PINNLoss

# ddp:
from argparse import ArgumentParser
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from GNN.utils.dist import allreduce_hook_sum
from torch import distributed

from GNN.utils.cluster import CustomLSFEnvironment, CustomManualEnvironment

end_time0 = time.time()
print('imported all', end_time0 - start_time)

def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Config file to train model. See ../configfiles for examples",
    )
    return parser

def set_seed(seed: int = 42):
    """Set random seed for reproducibility across Python, NumPy, and Torch (CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Manual DDP Process Setup
def manual_proc_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "23111"

    # initialize the process group
    distributed.init_process_group("nccl", rank=rank, world_size=world_size)


# Manual DDP Process Cleanup
def manual_proc_cleanup():
    distributed.destroy_process_group()


# Manual DDP Process Launcher
def manual_proc_launcher(proc_run_fn, m):

    world_size = m.get_world_size()
    print('mp.spawn starts')
    mp.spawn(proc_run_fn, args=(m,), nprocs=world_size, join=True)
    print('mp.spawn ends')


def manual_proc_runner(rank, m):
    world_size = m.get_world_size()
    print('world_size',world_size)
    print('rank', rank)
    manual_proc_setup(rank=rank, world_size=world_size)

    # ----- DDP Manual Env Setup -----

    env = CustomManualEnvironment(world_size=world_size, rank=rank)
    print('run starts')
    print('m.get_world_size()',m.get_world_size())
    print('env.world_size', env.world_size) 
    run(m=m, env=env)


def run(m, env):

    assert (
        m.get_world_size() == env.world_size
    ), f"World size {m.get_world_size()} in config does not match world size {env.world_size} in this current run"

    print("ENV:", env.__dict__)
    device = torch.device(f"cuda:{env.local_rank}")
    
    seed = m.get_random_seed()
    print(f'Set random seed {seed}')
    if seed == None:
        print('NOT USEING RANDOM SEED')
        pass
    else:
        set_seed(seed)
        print(f'USING RANDOM SEED {seed}')

    # ----- Datasets -----

    train_data_params = m.get_train_data_params()
    test_data_params = m.get_test_data_params()
    
    ds_class_name = m.get_class_name()
    print('ds_class_name', ds_class_name)
    DS = getattr(import_module("GNN.DatasetClasses." + ds_class_name), ds_class_name)
    train_dataset = DS(model_config=m, train_set=True, env=env, **train_data_params)
    test_dataset = DS(model_config=m, train_set=False, env=env, **test_data_params)
    print('dataset ready')    

    # ----- Device Settings -----
    
    print('Device settings starts')
    batch_size = m.get_batch_size()
    test_batch_size = m.get_test_batch_size()

    # hard-coded since this script is for ddp usage!
    use_parallel = False
    use_ddp = True
    # hard-coded to be set to False due to stability issues
    use_amp = False
    use_sbp = m.get_use_sbp()

    data_num_workers = m.get_data_num_workers()
    #data_num_workers = 1
    print('data_num_workers', data_num_workers)
    # NOTE: with distributed mode, we have one dataset that represents all train data
    # Each data item will be lazily loaded (from file).
    # We define this distributed sampler for each rank (process) to sample a subset of that dataset.
    sampler_class = DistributedSampler
    if m.get_distributed_sampler_bounds() is not None:
        sampler_class = partial(
            BoundedDistributedSampler, m.get_distributed_sampler_bounds()
        )
    train_sampler = sampler_class(
        train_dataset, num_replicas=env.world_size, rank=env.rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=data_num_workers,
        pin_memory=m.get_pin_memory(),
        sampler=train_sampler,
    )

    # all ranks do validation
    if m.get_distributed_sampler_bounds() is not None:
        sampler_class = partial(
            BoundedDistributedSampler, (1, 1e20)
        )  # force use of full patch for validation
    test_sampler = sampler_class(
        test_dataset, num_replicas=env.world_size, rank=env.rank, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        num_workers=data_num_workers,
        pin_memory=m.get_pin_memory(),
        sampler=test_sampler,
    )
    print('Device settings ready')

    # ----- Model & Training Setup -----

    print('Model & Training Setup Starts')
    mgn_dim = m.get_mgn_dim()
    mgn_layers = m.get_mgn_layers()
    mgn_layers_processor = m.get_mgn_layers_processor()
    print('mgn layers', mgn_layers)
    model_arch = m.get_model_arch()
    
    train_time_limit_hours=m.get_train_time_limit()
    val_time_limit_hours=m.get_validation_time_limit()
    print('train time limit' , train_time_limit_hours, 'hours')
    print('validation time limit',  val_time_limit_hours, 'hours')

    print('model arch', model_arch)

    assert model_arch in [
        "MeshGraphNets",
        "MultiMeshGraphNets",
    ], "Invalid `model_arch`, set to either 'MeshGraphNets' or 'MultiMeshGraphNets'"

    if model_arch == "MeshGraphNets":
        model = MeshGraphNets(
            train_dataset.num_node_features,
            train_dataset.num_edge_features,
            train_dataset.output_dim,
            out_dim_node=mgn_dim,
            out_dim_edge=mgn_dim,
            hidden_dim_node=mgn_dim,
            hidden_dim_edge=mgn_dim,
            hidden_layers_node=mgn_layers, #added
            hidden_layers_edge=mgn_layers, #added
            hidden_dim_processor_node=mgn_dim,
            hidden_dim_processor_edge=mgn_dim,
            hidden_layers_processor_node=mgn_layers_processor,  #added
            hidden_layers_processor_edge=mgn_layers_processor,  #added
            hidden_dim_decoder=mgn_dim,
            hidden_layers_decoder=mgn_layers,   #added
            node_normalizer_mask=train_dataset.node_normalizer_mask,
            edge_normalizer_mask=train_dataset.edge_normalizer_mask,
            mp_iterations=m.get_mp_iterations(),
            mlp_norm_type=m.get_mlp_norm_type(),
            output_type=m.get_output_type(),
            connection_type=m.get_connection_type(),
            connection_alpha=m.get_connection_alpha(),
            connection_aggregation=m.get_connection_aggregation(),
            graph_processor_type=m.get_graph_processor_type(),
            integrator=m.get_integrator(),
        )
    elif model_arch == "MultiMeshGraphNets":
        model = MultiMeshGraphNets(
            train_dataset.num_node_features,
            train_dataset.num_edge_features,
            train_dataset.output_dim,
            num_edge_types=len(train_dataset.graph_types),
            out_dim_node=mgn_dim,
            out_dim_edge=mgn_dim,
            hidden_dim_node=mgn_dim,
            hidden_dim_edge=mgn_dim,
            hidden_dim_processor_node=mgn_dim,
            hidden_dim_processor_edge=mgn_dim,
            hidden_dim_decoder=mgn_dim,
            node_normalizer_mask=train_dataset.node_normalizer_mask,
            edge_normalizer_mask=train_dataset.edge_normalizer_mask,
            mp_iterations=m.get_mp_iterations(),
            mlp_norm_type=m.get_mlp_norm_type(),
            output_type=m.get_output_type(),
            connection_type=m.get_connection_type(),
            connection_alpha=m.get_connection_alpha(),
            connection_aggregation=m.get_connection_aggregation(),
            graph_processor_type=m.get_graph_processor_type(),
            integrator=m.get_integrator(),
        )
    print('model ready \n')
    
    print('prepare loss function')
    # FUTURE PROJECT: change loss function from MSE to PDE residule (PINN)
    
    
    use_alw = m.get_use_alw()

    if use_alw:
        print('use adaptive loss weight')
        log_alpha = m.get_log_alpha()
        log_beta = m.get_log_beta()
        print('log_alpha', log_alpha, 'log_beta', log_beta)
        loss_function = PINNLoss_alw(log_alpha, log_beta)
    else:
        print('NOT using adaptive loss weight')
        loss_function = PINNLoss(device, m)

    print(f'loss function:{loss_function.__class__.__name__} \n')
    print('loss function ready \n')
    
    print('reload model')
    # reload model, if available
    checkpoint_dir = m.get_checkpoint_dir()
    load_prefix = m.get_load_prefix()

    model = load_model(
        checkpoint_dir=checkpoint_dir,
        model_filename="{0}_model.pt".format(load_prefix),
        model=model,
    )
    
    print('find place on device')
    # place on device
    model = model.to(device)
    print('place found \n')
   
    print('device id', env.local_rank)
    # We put model on DDP after loading checkpoint. Be sure to save model without the DP attached.
    # Must put model on device before using DDP.
    model = DistributedDataParallel(model, device_ids=[env.local_rank])
    
    print('check ddp hook, ddp_hook is', m.get_ddp_hook() )

    if m.get_ddp_hook() == "sum":
        model.register_comm_hook(None, allreduce_hook_sum)
    elif m.get_ddp_hook() == "mean":
        # default behavior, pass
        pass
    else:
        raise NotImplementedError(m.get_ddp_hook())

    # ----- Optimizer Settings -----
    print('prepare optimizer')
    
    optimiser = m.get_optimiser()
    if use_alw and optimiser == 'AdamW': # Using Apadtive weight in PINNLoss function
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(loss_function.parameters()),  # Include alpha and beta
            lr=m.get_lr(),
            weight_decay=m.get_weight_decay(),
        )
        print("Using Apadtive weight in optimizer")
    elif not use_alw and optimiser == 'LBFGS':
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=m.get_lr(), max_iter=1, history_size=1, line_search_fn="strong_wolfe"
        )
        print(' Using LBFGS, nut no Adaptive weight')
    elif not use_alw and optimiser == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=m.get_lr(), weight_decay=m.get_weight_decay()
        )
        print('Just good old AdamW')
        
    lr_scheduler = get_lr_scheduler(
        model_config=m, train_loader=train_loader, optimizer=optimizer
    )
    lr_scheduler, optimizer, last_epoch, info, best_loss = load_model_train_vars(
        checkpoint_dir=checkpoint_dir,
        scheduler_filename="{0}_scheduler.pt".format(load_prefix),
        scheduler=lr_scheduler,
        optimizer_filename="{0}_optimizer.pt".format(load_prefix),
        optimizer=optimizer,
        epoch_filename="{0}_epoch.pt".format(load_prefix),
        info_filename="{0}_info.pt".format(load_prefix),
        best_loss_filename="best_val_loss.pt",
    )
    print('optimizer ready \n')    

    # ----- Training Prep -----

    #########
    # tensorboard settings
    tb_dir = m.get_tensorboard_dir()
    tb_rate = m.get_tb_rate()
    # log_rate = m.get_log_rate()

    # create a summary writer.
    tb_writer = None
    if env.rank == 0 and m.get_use_tensorboard():
        tb_writer = SummaryWriter(tb_dir)
    global_step = (last_epoch + 1) * len(
        train_loader
    )  # assumes batch size does not change on continuation

    # ----- Train -----
    epoch_numer_haha = m.get_epochs()
    print('total epoch number', epoch_numer_haha)
    switch_epoch = m.get_switch_epoch()
    switch_epoch_flag = m.get_switch_epoch_flag()
    if switch_epoch_flag:
        print(f'switch train_epoch method at {switch_epoch} epoch')
    else:
        print('Not switch to any other train epoch method')
    ########
    # training loop
    # ----- barrier -----
    distributed.barrier()
    if best_loss is None:
        best_loss = val_epoch(
            val_loader=test_loader,
            m_config=m,
            model=model,
            loss_function=loss_function,
            device=device,
            parallel=use_parallel,
            ddp=use_ddp,
            amp=use_amp,
            env=env,
            time_limit_hours=m.get_validation_time_limit(),
            sample_limit=m.get_sample_limit(),
        )
        if np.isnan(best_loss):
            best_loss = np.inf
    distributed.barrier()
    grad_accum_steps = m.get_grad_accum_steps()
    sample_wise_losses = []

    start_time = time.time()

    # find training time, and (if resuming training) where training left off
    steps = m.get_steps()
    if steps:
        pbar = tqdm.tqdm(total=steps, position=0, leave=True, unit_scale=True)
        global_step = info["global_step"]
        pbar.update(global_step)
    else:
        pbar = tqdm.tqdm(total=m.get_epochs(), position=0, leave=True)
        global_step = (last_epoch + 1) * len(
            train_loader
        )  # assumes batch size does not change on continuation; assumes entire epoch finishes before each model save
        pbar.update(last_epoch + 1)

    while pbar.n < pbar.total:
        epoch = global_step // len(train_loader) if steps else pbar.n
        pbar.set_description(f"Step {global_step}" if steps else f"Epoch {epoch}")
        prior_step = global_step

        if use_sbp:
            train_loader = setup_sbp_train_loader(
                epoch=epoch,
                sbp_start_epoch=m.get_sbp_start_epoch(),
                sbp_rate=m.get_sbp_rate(),
                sbp_percent=m.get_sbp_percent(),
                sbp_randomess=m.get_sbp_randomness(),
                sample_wise_losses=sample_wise_losses,
                orig_train_dataset=train_dataset,
                batch_size=batch_size,
                use_parallel=use_parallel,
            )

        # DDP requires this line to make each epoch have its own data order
        train_sampler.set_epoch(epoch)
        
        
        # ----- barrier -----
        distributed.barrier()
        if optimiser == 'AdamW':
            if epoch <= switch_epoch or not switch_epoch_flag:
                
                train_loss, global_step, sample_wise_losses = train_epoch_MSE(
                    train_loader=train_loader,
                    model=model,
                    m_config=m,
                    loss_function=loss_function,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    device=device,
                    parallel=use_parallel,
                    ddp=use_ddp,
                    grad_accum_steps=grad_accum_steps,
                    amp=use_amp,
                    tb_writer=tb_writer,
                    tb_rate=tb_rate,
                    epoch=epoch,
                    global_step=global_step,
                    steps=steps,
                    env=env,
                    time_limit_hours=m.get_train_time_limit(),
                )
            else:
                print('switched to pinn')
                train_loss, global_step, sample_wise_losses = train_epoch(
                    train_loader=train_loader,
                    model=model,
                    m_config=m,
                    loss_function=loss_function,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    device=device,
                    parallel=use_parallel,
                    ddp=use_ddp,
                    grad_accum_steps=grad_accum_steps,
                    amp=use_amp,
                    tb_writer=tb_writer,
                    tb_rate=tb_rate,
                    epoch=epoch,
                    global_step=global_step,
                    steps=steps,
                    env=env,
                    time_limit_hours=m.get_train_time_limit(),
                )
        elif optimiser == 'LBFGS':
            train_loss, global_step, sample_wise_losses = train_epoch_LBFGS(
                train_loader=train_loader,
                model=model,
                m_config=m,
                loss_function=loss_function,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                device=device,
                parallel=use_parallel,
                ddp=use_ddp,
                grad_accum_steps=grad_accum_steps,
                amp=use_amp,
                tb_writer=tb_writer,
                tb_rate=tb_rate,
                epoch=epoch,
                global_step=global_step,
                steps=steps,
                env=env,
                time_limit_hours=m.get_train_time_limit(),
            )
        distributed.barrier()
        # ----- barrier -----

        pbar.refresh()
        val_loss = torch.tensor(0, device=device)

        # ----- barrier -----
        distributed.barrier()
        val_loss = val_epoch(
            val_loader=test_loader,
            model=model,
            m_config=m,
            loss_function=loss_function,
            device=device,
            parallel=use_parallel,
            ddp=use_ddp,
            amp=use_amp,
            tb_writer=tb_writer,
            global_step=global_step,
            epoch=epoch,
            env=env,
            time_limit_hours=m.get_validation_time_limit(),
            sample_limit=m.get_sample_limit(),
        )
        distributed.barrier()
        # ----- barrier -----

        pbar.refresh()

        if tb_writer is not None:
            tb_writer.add_scalar(
                "Opt/lr",
                scalar_value=optimizer.param_groups[0]["lr"],
                global_step=global_step,
            )
            tb_writer.add_scalar(
                "Profile/epoch_time",
                scalar_value=get_delta_minute(start_time),
                global_step=global_step,
            )

        if "ReduceLROnPlateau" in str(lr_scheduler.__class__):
            lr_scheduler.step(val_loss)

        if env.rank == 0:
            if val_loss < best_loss:
                best_loss = val_loss
                save_model(
                    checkpoint_dir=checkpoint_dir,
                    model_filename="best_val_model.pt",
                    model=model,
                    scheduler_filename="best_val_scheduler.pt",
                    scheduler=lr_scheduler,
                    optimizer_filename="best_val_optimizer.pt",
                    optimizer=optimizer,
                    epoch_filename="best_val_epoch.pt",
                    epoch=epoch,
                    best_loss_filename="best_val_loss.pt",
                    best_loss=best_loss,
                    info_filename="best_val_info.pt",
                    info={"global_step": global_step} if steps else None,
                )
            if epoch == switch_epoch-1:
                print(f'save state at {switch_epoch}-1')
                save_model(
                    checkpoint_dir=checkpoint_dir,
                    model_filename=f"{switch_epoch}_epoch_model.pt",
                    model=model,
                    scheduler_filename=f"{switch_epoch}_epoch_scheduler.pt",
                    scheduler=lr_scheduler,
                    optimizer_filename=f"{switch_epoch}_epoch_optimizer.pt",
                    optimizer=optimizer,
                    epoch_filename=f"{switch_epoch}_epoch.pt",
                    epoch=epoch,
                    best_loss_filename=f"{switch_epoch}_epoch_loss.pt",
                    best_loss=best_loss,
                    info_filename=f"{switch_epoch}_epoch_info.pt",
                    info={"global_step": global_step} if steps else None,
                )
            record_epoch = 49
            if epoch == record_epoch - 1:
                print(f'save state at {record_epoch} - 1')
                save_model(
                    checkpoint_dir=checkpoint_dir,
                    model_filename=f"{record_epoch}_epoch_model.pt",
                    model=model,
                    scheduler_filename=f"{record_epoch}_epoch_scheduler.pt",
                    scheduler=lr_scheduler,
                    optimizer_filename=f"{record_epoch}_epoch_optimizer.pt",
                    optimizer=optimizer,
                    epoch_filename=f"{record_epoch}_epoch.pt",
                    epoch=epoch,
                    best_loss_filename=f"{record_epoch}_epoch_loss.pt",
                    best_loss=best_loss,
                    info_filename=f"{record_epoch}_epoch_info.pt",
                    info={"global_step": global_step} if steps else None,
                )
                
            record_epoch2 = 2
            if epoch == record_epoch2 - 1:
                print(f'save state at {record_epoch2} - 1')
                save_model(
                    checkpoint_dir=checkpoint_dir,
                    model_filename=f"{record_epoch2}_epoch_model.pt",
                    model=model,
                    scheduler_filename=f"{record_epoch2}_epoch_scheduler.pt",
                    scheduler=lr_scheduler,
                    optimizer_filename=f"{record_epoch2}_epoch_optimizer.pt",
                    optimizer=optimizer,
                    epoch_filename=f"{record_epoch2}_epoch.pt",
                    epoch=epoch,
                    best_loss_filename=f"{record_epoch2}_epoch_loss.pt",
                    best_loss=best_loss,
                    info_filename=f"{record_epoch2}_epoch_info.pt",
                    info={"global_step": global_step} if steps else None,
                )
            save_model(
                checkpoint_dir=checkpoint_dir,
                model_filename="last_epoch_model.pt",
                model=model,
                scheduler_filename="last_epoch_scheduler.pt",
                scheduler=lr_scheduler,
                optimizer_filename="last_epoch_optimizer.pt",
                optimizer=optimizer,
                epoch_filename="last_epoch_epoch.pt",
                epoch=epoch,
                info_filename="last_epoch_info.pt",
                info={"global_step": global_step} if steps else None,
            )

        pbar.update(1 if not steps else global_step - prior_step)

    # After training finishes, write out a finished flag to stop the chaining of jobs.
    if env.rank == 0:
        finished_flag_file = Path(m.get_train_output_dir()) / "finished.flag"
        with open(finished_flag_file, mode="w") as f:
            f.close()

        # perform `interval_actions` every m.get_save_interval() epochs/steps
        if m.get_save_interval() is not None and env.rank == 0:
            interval_actions(
                pbar,
                m,
                checkpoint_dir,
                model,
                lr_scheduler,
                optimizer,
                epoch,
                global_step,
                steps,
                test_dataset,
                tb_writer,
                device,
            )


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    end_time1 = time.time()
    print(args.config_file)
    print(end_time1-start_time)

    # ----- Config/Parser -----
    m = ModelConfig(args.config_file)

    ddp_type = m.get_ddp_type()
    end_time2 = time.time()
    print("ddp_type:", ddp_type)
    print(end_time2-start_time)

    assert ddp_type in [
        "manual",
        "srun",
    ], "Invalid `ddp_type`, set to either 'manual' or 'srun'"

    if ddp_type == "manual":
        print(ddp_type)
        manual_proc_launcher(proc_run_fn=manual_proc_runner, m=m)

    elif ddp_type == "srun":
        env = CustomLSFEnvironment()
        print('master_addr', os.environ["MASTER_ADDR"])
        os.environ["MASTER_PORT"] = "23111"

        end_time3 = time.time()
        print('init process starts', end_time3-start_time)
        world_size=env.world_size
        print('world size', world_size)
        rank=env.rank
        print('rank', rank)
        distributed.init_process_group(
            "nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=3600),
            world_size=env.world_size,
            rank=env.rank,
        )
        end_time4 = time.time()
        print('init process finishes', end_time4-start_time)
        run(m=m, env=env)
