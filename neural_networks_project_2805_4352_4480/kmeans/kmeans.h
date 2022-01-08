#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "utility.h"

#define NUM_OF_CLUSTERS 3
#define EUCLIDEAN(dx,dy) sqrt((dx*dx)+(dy*dy))
#define LEN_2D(x)  (sizeof(x[0]) / sizeof(x[0][0]))

typedef struct {
	double	x;
	double	y;
	int		group;
} Cluster;

Cluster cluster_list[NUM_OF_CLUSTERS];
void intialize_clusters(float** dataset, int len_of_dataset);
void reset_array(float array[NUM_OF_CLUSTERS][3]);
void set_labels(float** dataset, int len_of_dataset);
void reposition_cluster_centers(float** dataset, float cluster_sum_info[NUM_OF_CLUSTERS][3], int len_of_dataset);
int clusters_converged(float previous_clusters[NUM_OF_CLUSTERS][2]);
void print_tables(float cluster_sum_info[NUM_OF_CLUSTERS][3], int epoch);
void write_labeled_dataset_to_file(char* filename, float** dataset, int len_dataset);
void write_kmeans_clusters_to_file(char* filename);
float intra_cluster_variance(float** dataset, int len_of_dataset);
float kmeans(char* filename, int max_iter);
