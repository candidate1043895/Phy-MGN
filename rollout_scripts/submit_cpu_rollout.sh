#! /bin/bash


#SBATCH -N 1	
#SBATCH -p scarf
#SBATCH -n 4
#SBATCH --mem 0 
#SBATCH --account clfextern
#SBATCH --time 10:00:00

#SBATCH -o rollout.log
#SBATCH -e rollout.err

module purge
module load AMDmodules
module load CUDA/11.7.0
module load Python/3.11.3-GCCcore-12.3.0


source /work4/clf/scarf1271/MGNenv/bin/activate

#python rollout_full_domains.py  /work4/clf/scarf1271/GNN/MGN/config_files/CylinderFlow/cylinder_flow_ddp_2node_full_domain.ini --cpu --viz_rollout_num=60 --GT -test --rollout_num=60

python rollout_full_domains.py /work4/clf/scarf1271/GNN/data_sets/cylinder_flow/output/test/DeepMind_CylinderFlow_DDP_2node_epochs_25_bs_8_useamp_False_partitioning_null/velocity/config.ini --cpu --viz_rollout_num 70 --GT  -test  --rollout_num 70

