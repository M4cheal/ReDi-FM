#!/bin/bash
clear
datasets=("BIDMC" "BMC" "HK" "I2CVB" "RUNMC" "UCL")



for dataset in "${datasets[@]}"; do
    echo '------------------------------------------------------------------'
    echo $dataset
    
    /root/miniconda3/envs/CTTAKAN/bin/python \
        /root/code/ReDi-FM/Prostate_SAM/baseline_only_source.py \
        --Source_Dataset $dataset
    
    echo '------------------------------------------------------------------'
done
