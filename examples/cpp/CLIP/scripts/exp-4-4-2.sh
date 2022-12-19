mpirun -x LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64: -np 1 -H g0001 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import /home/shicao/opt-n/build/st_concat-synthetic4-2_hwp100_ubs4_nd4_lin -b 64 --num-branch 4 --hidden-size 1024 --num-heads 16 --num-layer 12 --seq-len 256 --sequential-pipeline -ll:util 4 -e 10 --fusion | tail -n 4 >> concat-synthetic4-2-lin-ubs4-64-nd4.log
mpirun -x LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64: -np 1 -H g0001 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import /home/shicao/opt-n/build/st_concat-synthetic4-2_hwp100_ubs4_nd4_lin -b 128 --num-branch 4 --hidden-size 1024 --num-heads 16 --num-layer 12 --seq-len 256 --sequential-pipeline -ll:util 4 -e 10 --fusion | tail -n 4 >> concat-synthetic4-2-lin-ubs4-128-nd4.log
mpirun -x LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64: -np 1 -H g0001 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import /home/shicao/opt-n/build/st_concat-synthetic4-2_hwp100_ubs4_nd4_lin -b 256 --num-branch 4 --hidden-size 1024 --num-heads 16 --num-layer 12 --seq-len 256 --sequential-pipeline -ll:util 4 -e 10 --fusion | tail -n 4 >> concat-synthetic4-2-lin-ubs4-256-nd4.log
mpirun -x LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64: -np 1 -H g0001 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import /home/shicao/opt-n/build/st_concat-synthetic4-2_hwp100_ubs4_nd4_lin -b 32 --num-branch 4 --hidden-size 1024 --num-heads 16 --num-layer 12 --seq-len 256 --sequential-pipeline -ll:util 4 -e 10 --fusion | tail -n 4 >> concat-synthetic4-2-lin-ubs4-32-nd4.log
mpirun -x LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64: -np 1 -H g0001 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import /home/shicao/opt-n/build/st_concat-synthetic4-2_hwp100_ubs8_nd4_par -b 64 --num-branch 4 --hidden-size 1024 --num-heads 16 --num-layer 12 --seq-len 256 -ll:util 4 -e 10 --fusion | tail -n 4 >> concat-synthetic4-2-par-ubs8-64-nd4.log
mpirun -x LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64: -np 1 -H g0001 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import /home/shicao/opt-n/build/st_concat-synthetic4-2_hwp100_ubs8_nd4_par -b 128 --num-branch 4 --hidden-size 1024 --num-heads 16 --num-layer 12 --seq-len 256 -ll:util 4 -e 10 --fusion | tail -n 4 >> concat-synthetic4-2-par-ubs8-128-nd4.log
mpirun -x LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64: -np 1 -H g0001 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import /home/shicao/opt-n/build/st_concat-synthetic4-2_hwp100_ubs8_nd4_par -b 256 --num-branch 4 --hidden-size 1024 --num-heads 16 --num-layer 12 --seq-len 256 -ll:util 4 -e 10 --fusion | tail -n 4 >> concat-synthetic4-2-par-ubs8-256-nd4.log
mpirun -x LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64: -np 1 -H g0001 ./clip -ll:gpu 4 -ll:fsize 15000 -ll:zsize 4000 --import /home/shicao/opt-n/build/st_concat-synthetic4-2_hwp100_ubs8_nd4_par -b 32 --num-branch 4 --hidden-size 1024 --num-heads 16 --num-layer 12 --seq-len 256 -ll:util 4 -e 10 --fusion | tail -n 4 >> concat-synthetic4-2-par-ubs8-32-nd4.log