#!/bin/bash
start=${1:-1}
end=${2:-1}

for i in $( seq $start $end )
do
   python train_volume.py --config ./config_volume/config_photon_acorn.ini --pmt_id ${i} --normalize_input
   python train_volume.py --config ./config_volume/config_photon_acorn.ini --pmt_id ${i} --load ../logs/photon_${i} --export --upsample 1 --normalize_input
done