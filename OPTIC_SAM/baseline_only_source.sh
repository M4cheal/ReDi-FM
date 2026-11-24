#!/bin/bash
clear
datasets=("Drishti_GS" "ORIGA" "REFUGE" "REFUGE_Valid" "RIM_ONE_r3")


for dataset in "${datasets[@]}"; do
    echo '------------------------------------------------------------------'
    echo $dataset
    
    /root/miniconda3/envs/CTTAKAN/bin/python \
        /root/code/ReDi-FM/OPTIC_SAM/baseline_only_source.py \
        --Source_Dataset $dataset
    
    echo '------------------------------------------------------------------'
done
