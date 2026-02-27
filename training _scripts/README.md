# Training Scripts

*Note: The submission script is written for STFC SCARF cluster https://www.scarf.rl.ac.uk/access.html*





## Single-Node, Single-Process Training

`python train_mgn_generalized.py --config [CONFIG_FILE]`
- `[CONFIG_FILE]` - Must be an **absolute path** to a config file
- Trains a model on an interactive single node

## Single-Node, Multi-Process Distributed Data Parallel (DDP) Training

`python train_mgn_generalized_ddp.py --config [CONFIG_FILE]`
- `[CONFIG_FILE]` - Must be an **absolute path** to a config file
- Be sure to set `ddp_type` to `manual` in the configuration file
- Trains a model on an interactive single node, using DDP

## Multi-Node, Multi-Process Distributed Data Parallel (DDP) Training

On SCARF HTC cluster
1. Request GPU resources

Request two nodes, 8 GPUs with equal number of tasks per node

`salloc -p gpu -N 2 -n 8 --gpus-per-node 4 --gpu-bind none --ntasks-per-node 4 --mem=0!`

2. On MASTER node, import all modules and activate python evnironment

`module purge`

`module load AMDmodules`

`module load CUDA/11.7.0`

`module load Python/3.11.3-GCCcore-12.3.0`

`module list`

`source /work4/clf/scarf1271/MGNenv/bin/activate`

3. Go to MGN/training_script, set values to slurm environment on both nodes
   
`export NCCL_SOCKET_IFNAME=eth0`

`export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)`

`export RDZV_PORT=53380`


4. Srun executes same command on both nodes in parllel

`srun python train_mgn_generalized_ddp.py --config [CONFIG_FILE]`

- `[CONFIG_FILE]` - Must be an **absolute path** to a config file
- Trains a model on an interactive double node, using DDP


## Multi-Node, Multi-Process Distributed Data Parallel (DDP) Training using submission script
### Setup
- Be sure to update settings in submission_2node.sh:
  - `account` - Account to charge compute hours
  - `TIME` - Time limite for each job, in HH:MM:SS format
  - `partition` - The partition of cluster to request resources
  - `module load` - modules need to be loaded (Python environment, CUDA)
  - `source /path/to/python/environment/bin/activate` -Activate Python environment
-  In your config file, be sure to set `ddp_type` to `srun`
  
