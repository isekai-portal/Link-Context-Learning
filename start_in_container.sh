#!/usr/bin/env bash
#sleep 30
#fi_info -p efa -t FI_EP_RDM

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE are coming from the main script

set -e
echo "*********** node summary ***********"
echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID

#source ${CONDA_PREFIX_1}/bin/activate llava
echo python3 version = `python3 --version`
echo "************************************"
python -c "import torch"

set -x
accelerate launch  \
    --num_processes $(( 8 * $COUNT_NODE ))  \
    --num_machines $COUNT_NODE  \
    --multi_gpu  \
    --machine_rank $THEID  \
    --main_process_ip $MASTER_ADDR  \
    --main_process_port $MASTER_PORT  \
    $@
