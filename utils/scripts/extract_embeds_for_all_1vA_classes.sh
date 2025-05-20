#!/bin/bash

declare -a arr=("Crescent" "Sclerosis" "Normal" "Podocitopatia" "Hypercelularidade" "Membranous")

for i in "${arr[@]}"; do
    echo "Processing $i"
        ./bin/python3 -m embedding.extraction\
        -i /datasets/terumo-data-jpeg \
        -o /media/david/Expansion\ Drive/david/fm-embedding-eval/extracted-embeddings-one_vs_all/$i \
        --mae-size large \
        --one-vs-all $i \
        --classes terumo
done
