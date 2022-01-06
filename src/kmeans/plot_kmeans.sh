#!/bin/bash
gcc run_kmeans.c kmeans.c utility.c -lm
./a.out
python ../datasets/plot_dataset.py ../../out/SEL_labeled_data.txt ../../out/SEL_kmeans_clusters.txt