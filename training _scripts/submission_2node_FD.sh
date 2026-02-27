#! /bin/bash

#SBATCH -N 2
#SBATCH -p gpu
#SBATCH -n 8
# #SBATCH --nodelist=gn3001,gn3002,gn3003
#SBATCH --gpus-per-node 4
#SBATCH --gpu-bind none
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 4
#SBATCH --mem 0

#SBATCH --account clfextern
#SBATCH --time 20:00:00

#SBATCH -o Sedov_128_to_64_FD_smooth_v2_PDE_weight4e-5.log
#SBATCH -e Sedov_128_to_64_FD_smooth_v2_PDE_weight4e-5.err


module purge
module load  CUDA/11.7.0
module load Python/3.10.8-GCCcore-12.3.0
source /work4/clf/scarf1271/MODenv2/bin/activate

export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_NCCL_ENABLE_MONITORING=1
# use command   ip a   to check available socket ifname
#export NCCL_SOCKET_IFNAME=primary0
export NCCL_SOCKET_IFNAME=eth0,primary0
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export RDZV_PORT=53380

#srun python train_mgn_generalized_ddp.py --config /work4/clf/scarf1271/GNN/data_sets/cylinder_flow_uvp2/cylinder_flow_ddp_2node_full_domain.ini

#srun python train_mgn_generalized_ddp_set_seed.py --config /work4/clf/scarf1271/GNN/data_sets/Sedov/500_smoothed_MGN_v6/sedov_explosion_ddp_2node_full_domain.ini
srun python train_mgn_generalized_ddp.py --config /work4/clf/scarf1271/GNN/data_sets/Sedov/500train_128_to_64_FD_smooth_v2_PDE_weight4e-5/sedov_explosion_ddp_2node_full_domain.ini
#srun python train_mgn_generalized_ddp_set_seed.py --config_file /work4/clf/scarf1271/GNN/data_sets/Sedov/500_muscl_4pde_v2/sedov_explosion_ddp_2node_full_domain.ini
#srun python train_mgn_generalized_ddp.py --config /work4/clf/scarf1271/GNN/data_sets/RHD_Riemann2D/480_train_MGN_v6/riemann_ddp_2node_full_domain.ini



#srun python train_mgn_generalized_ddp_set_seed.py --config /work4/clf/scarf1271/GNN/data_sets/Sedov/smoothed_single_state/density/500_smoothed_MGN_density_v2/sedov_explosion_ddp_2node_full_domain.ini

#srun python train_mgn_generalized_ddp_set_seed.py --config /work4/clf/scarf1271/GNN/data_sets/Sedov/smoothed_single_state/density/500_smoothed_MGN_density_v4/sedov_explosion_ddp_2node_full_domain.ini

#srun python train_mgn_generalized_ddp_set_seed.py --config /work4/clf/scarf1271/GNN/data_sets/Sedov/smoothed_single_state/density/500_smoothed_MGN_density_v5/sedov_explosion_ddp_2node_full_domain.ini

#srun python train_mgn_generalized_ddp_set_seed.py --config /work4/clf/scarf1271/GNN/data_sets/Sedov/smoothed_single_state/density/500_smoothed_MGN_density_v6/sedov_explosion_ddp_2node_full_domain.ini

#srun python train_mgn_generalized_ddp_set_seed.py --config /work4/clf/scarf1271/GNN/data_sets/Sedov/smoothed_single_state/density/500_smoothed_MGN_density_v7/sedov_explosion_ddp_2node_full_domain.ini
