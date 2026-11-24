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
datasets=("BIDMC" "BMC" "HK" "I2CVB" "RUNMC" "UCL")
for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo '------------------------------------------------------------------'
        echo $dataset
        echo $seed
        
        /root/miniconda3/envs/CTTAKAN/bin/python \
            /root/code/ReDi-FM/Prostate_SAM/ReDi-FM.py \
            --Source_Dataset $dataset --exp_name "ReDi-FM/seed_${seed}" \
            --loss_bate_soft 1.0 --lr 1e-4 --seed $seed \
            --model_root "/root/autodl-tmp/save/models_sam/ProstateSlice/"
        echo '------------------------------------------------------------------'
    done
done





