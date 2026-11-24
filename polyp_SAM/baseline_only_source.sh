#!/bin/bash
clear
datasets=("BKAI" "CVC-ClinicDB" "ETIS-LaribPolypDB" "Kvasir-SEG")


for dataset in "${datasets[@]}"; do
    echo '------------------------------------------------------------------'
    echo $dataset
    /root/miniconda3/envs/CTTAKAN/bin/python \
        /root/code/ReDi-FM/polyp_SAM/baseline_only_source.py \
        --Source_Dataset $dataset --backbone "b0" --model_root "/root/autodl-tmp/save/models_sam/polyp_b0/"
    
    echo '------------------------------------------------------------------'
done
