export PATH=/mnt/cache/share/gcc/gcc-7.3.0/bin:$PATH
export LD_LIBRARY_PATH=/mnt/cache/share/gcc/gcc-7.3.0/lib64:/mnt/cache/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH

export CUDA_HOME=/mnt/cache/share/cuda-11.7
export PATH=/mnt/cache/share/cuda-11.7/bin::$PATH
export LD_LIBRARY_PATH=/mnt/cache/share/cuda-11.7/lib64:$LD_LIBRARY_PATH

partition=$1
node_num=$2

srun --mpi=pmi2 -p $partition -n$node_num --quotatype=auto --gres=gpu:1 \
    --ntasks-per-node=$node_num --cpus-per-task=8  --job-name=GAMMA300 --phx-priority P0 --comment wbsM-SC220015.001 -x SH-IDC1-10-142-4-105 \
    python3.10 clip_merge.py
