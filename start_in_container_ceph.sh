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

# ceph
mount_point='/mnt/cache/chenkeqin/mount'
mkdir -p $mount_point

# public_bucket
# s3fs openmmlab /mnt/lustre/share_data/taiyan/ceph/coco -o url=http://10.142.4.254:80 -o use_path_request_style -o public_bucket=1

# check if is available
df | grep $mount_point
if [ $? -ne -0 ]; then
    fusermount -u $mount_point
fi

mounted=`df | grep $mount_point | wc -l`
# not mounted
if [[ mounted -eq 0 ]]; then
    echo "mount" $bucket $mount_point
    s3fs multimodal $mount_point -o url=http://10.117.39.245:80 -o use_path_request_style -o uid=$(id -u) -o gid=$(id -g) -o nonempty
# mounted
else
    echo "already mounted. nothing to do." $mount_point
fi
# end ceph

accelerate launch  \
    --num_processes $(( 8 * $COUNT_NODE ))  \
    --num_machines $COUNT_NODE  \
    --multi_gpu  \
    --machine_rank $THEID  \
    --main_process_ip $MASTER_ADDR  \
    --main_process_port $MASTER_PORT  \
    $@

fusermount -u $mount_point