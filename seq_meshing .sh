#!/bin/bash
output=ICL2_reservoir_union_10000

for ((i=1;i<=87;i++));
do
    frame=$[10*$i]
    echo "frame-${frame}"
    python experiment_scripts/test_sdf.py --checkpoint_path=logs/network/${output}/${frame}/checkpoints/model_current.pth --experiment_name=model/${output}/${frame}/ --resolution=512
done
