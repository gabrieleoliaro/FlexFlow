#!/bin/bash
nb=$1
dev=$2
conf=$3
ubs_lin=$4
ubs_par=$5

hs=2304
nh=12
nl=16
sl=512

if [ $conf == 2 ]; then
hs=1024
nh=16
nl=12
sl=256
fi

if [ $conf == 3 ]; then
hs=512
nh=8
nl=12
sl=256
fi

log_file_lin="concat-synthetic$nb-$conf-lin-ubs$ubs_lin"
log_file_par="concat-synthetic$nb-$conf-par-ubs$ubs_par"
st_lin="/home/shicao/ff-gpp/examples/cpp/CLIP/st$dev/$log_file_lin.txt"
st_par="/home/shicao/ff-gpp/examples/cpp/CLIP/st$dev/$log_file_par.txt"
lib="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
count=1

mbs1=$(($ubs_lin * 32))
mbs2=$(($mbs1 * 2))
mbs3=$(($mbs2 * 2))
mbs4=$(($mbs3 * 2))


while [ $count -gt 0 ]; do
    for MBS in $mbs1 $mbs2 $mbs3 $mbs4; do
    log="$log_file_lin-$MBS-nd$dev.log"
    
    cmd="mpirun -x LD_LIBRARY_PATH=$lib -np 4 -H g0003,g0001,g0002,g0004 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import $st_lin -b $MBS --num-branch $nb --hidden-size $hs --num-heads $nh --num-layer $nl --seq-len $sl --sequential-pipeline -ll:util 4 -e 10 --fusion | tail -n 4 >> $log"
    # cmd="mpirun -x LD_LIBRARY_PATH=$lib -np 2 -H g0003,g0001 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import $st_lin -b $MBS --num-branch 4 --hidden-size 2304 --num-heads 12 --num-layer 16 --seq-len 512 -ll:util 4 -e 10 --fusion | tail -n 4 >> $log"
    # cmd="mpirun -x LD_LIBRARY_PATH=$lib -np 2 -H g0003,g0001 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import $st_lin -b $MBS --num-branch $nb --hidden-size $hs --num-heads $nh --num-layer $nl --seq-len $sl --sequential-pipeline -ll:util 4 -e 10 --fusion | tail -n 4 >> $log"
    if [ $dev == 8 ]; then
    cmd="mpirun -x LD_LIBRARY_PATH=$lib -np 2 -H g0003,g0001 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import $st_lin -b $MBS --num-branch $nb --hidden-size $hs --num-heads $nh --num-layer $nl --seq-len $sl --sequential-pipeline -ll:util 4 -e 10 --fusion | tail -n 4 >> $log"
    fi
    echo $cmd
    # echo $cmd >> $log

    # eval $cmd
    done # MBS

    for MBS in $mbs1 $mbs2 $mbs3 $mbs4; do
    log="$log_file_par-$MBS-nd$dev.log"
    cmd="mpirun -x LD_LIBRARY_PATH=$lib -np 4 -H g0003,g0001,g0002,g0004 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import $st_par -b $MBS --num-branch $nb --hidden-size $hs --num-heads $nh --num-layer $nl --seq-len $sl -ll:util 4 -e 10 --fusion | tail -n 4 >> $log"
    # cmd="mpirun -x LD_LIBRARY_PATH=$lib -np 2 -H g0003,g0001 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import $st_par -b $MBS -ll:util 4 -e 10 --fusion | tail -n 4 >> $log"
    # cmd="mpirun -x LD_LIBRARY_PATH=$lib -np 2 -H g0003,g0001 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import $st_par -b $MBS --num-branch $nb --hidden-size $hs --num-heads $nh --num-layer $nl --seq-len $sl -ll:util 4 -e 10 --fusion | tail -n 4 >> $log"
    if [ $dev == 8 ]; then
    cmd="mpirun -x LD_LIBRARY_PATH=$lib -np 2 -H g0003,g0001 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import $st_par -b $MBS --num-branch $nb --hidden-size $hs --num-heads $nh --num-layer $nl --seq-len $sl -ll:util 4 -e 10 --fusion | tail -n 4 >> $log"
    fi
    echo $cmd
    # echo $cmd >> $log

    # eval $cmd
    done # MBS

    count=$(( $count - 1))
done