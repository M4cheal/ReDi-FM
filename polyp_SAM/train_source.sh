#!/bin/bash
clear
datasets=("BKAI" "CVC-ClinicDB" "ETIS-LaribPolypDB" "Kvasir-SEG")


for dataset in "${datasets[@]}"; do
    echo '------------------------------------------------------------------'
    echo $dataset
    
    /root/miniconda3/envs/CTTAKAN/bin/python \
        /root/code/ReDi-FM/polyp_SAM/train_source.py \
        --Source_Dataset $dataset --lr 1e-4 --backbone "b0" --path_save_model "/root/autodl-tmp/save/test/models_sam/polyp_b0/"
    
    echo '------------------------------------------------------------------'
done
