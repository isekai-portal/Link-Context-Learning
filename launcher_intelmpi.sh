#!/bin/bash
#SBATCH --partition=mm_v100_32g
#SBATCH --job-name=unify_mm_v100
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 1
#SBATCH -c 64
#SBATCH --gres=gpu:8
#SBATCH --output=%x_%j.out
#SBATCH --comment "wbsR-SC230999.001.02"
#SBATCH --exclusive

set -e
# I have absolutely no idea what these export does. just copy from https://gist.github.com/rom1504/474f97a95a526d40ae44a3fc3c657a2e.
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
#export NCCL_ALGO=ring
export NCCL_DEBUG=info
#export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,COLL
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0
#export NCCL_P2P_DISABLE=1
#export NCCL_IBEXT_DISABLE=1
#export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
if [ "$MASTER_PORT" == "" ]; then
  export MASTER_PORT=12802
  echo "MASTER_PORT IS NULL. USE DEFAULT PORT: ${MASTER_PORT}"
fi
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo "*********** sbatch launch summary ****************"
echo MASTER_PORT: $MASTER_PORT
echo MASTER_ADDR: $MASTER_ADDR
echo COUNT_NODE: $COUNT_NODE
echo HOSTNAMES: $HOSTNAMES
echo "**************************************************"

set -x
/mnt/lustre/share/intel64_cluster/impi/2017.1.132/bin64/mpirun -n $COUNT_NODE -perhost 1 start_in_container.sh $@
