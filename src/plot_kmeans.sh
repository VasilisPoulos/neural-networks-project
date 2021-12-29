#!/bin/bash
gcc kmeans.c utility.c -lm
./a.out
python plot_dataset.py ../out/labeled_data.txt ../out/kmeans_clusters.txt