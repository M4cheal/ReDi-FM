#!/bin/bash
clear

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1

seeds=(0 1 2)
datasets=("BKAI" "CVC-ClinicDB" "ETIS-LaribPolypDB" "Kvasir-SEG")
for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo '------------------------------------------------------------------'
        echo $dataset
        echo $seed
        
        /root/miniconda3/envs/CTTAKAN/bin/python \
            /root/code/ReDi-FM/polyp_SAM/ReDi-FM.py \
            --Source_Dataset $dataset --exp_name "ReDi-FM/seed_${seed}" \
            --loss_bate_soft 1.0 --seed $seed --lr 1e-5
        echo '------------------------------------------------------------------'
    done
done




