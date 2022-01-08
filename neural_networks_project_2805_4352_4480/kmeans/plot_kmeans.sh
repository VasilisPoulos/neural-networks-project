#!/bin/bash
gcc run_kmeans.c kmeans.c utility.c -lm
./a.out
python ../datasets/plot_dataset.py ../kmeans/labeled_data_final.txt ../kmeans/kmeans_clusters_final.txt