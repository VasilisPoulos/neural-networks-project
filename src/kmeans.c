#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "utility.h"

#define NUM_OF_CLUSTERS 10
// When comparing distances we can omit the square root of the euclidean 
// distance function and we can make it into a macro so we don'thave to pay for 
// function call overhead.
#define EUCLIDEAN(dx,dy) sqrt((dx*dx)+(dy*dy))
#define LEN_2D(x)  (sizeof(x[0]) / sizeof(x[0][0]))

typedef struct {
	double	x;
	double	y;
	int		group;
} Cluster;

Cluster cluster_list[NUM_OF_CLUSTERS];
void intialize_clusters(Cluster* cluster_list, float** dataset, \
    int len_of_dataset);
void reset_array(float array[NUM_OF_CLUSTERS][3]);
void set_labels(float** dataset, int len_of_dataset);
void reposition_cluster_centers(float** dataset, \
    float cluster_sum_info[NUM_OF_CLUSTERS][3], \
    Cluster* cluster_list, int len_of_dataset);
int clusters_converged(Cluster* cluster_list, float previous_clusters[NUM_OF_CLUSTERS][2]);
void print_tables(float cluster_sum_info[NUM_OF_CLUSTERS][3], Cluster* cluster_list, \
    int epoch);
void write_labeled_dataset_to_file(char* filename, float** dataset, int len_dataset);
void write_kmeans_clusters_to_file(char* filename, Cluster* cluster_list);
float* error_calc(Cluster* cluster_list, float** dataset, int len_of_dataset);
float kmeans(int);

void intialize_clusters(Cluster* cluster_list, float** dataset, int len_of_dataset){
    srand(time(NULL)); 
    Cluster cluster;
    for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
    {
        int dataset_idx = generate_random_float(0, len_of_dataset - 1);
        cluster.x = dataset[dataset_idx][0];
        cluster.y = dataset[dataset_idx][1];
        cluster.group = idx;
        cluster_list[idx] = cluster;
    }
}

void reset_array(float array[NUM_OF_CLUSTERS][3]){
    for(int i = 0; i < NUM_OF_CLUSTERS; i++) {
        for (int j = 0; j < 3; j++) {
            array[i][j] = 0;  
        }
    }
}

void set_labels(float** dataset, int len_of_dataset){
    float dx = 0;
    float dy = 0;
    float min_distance = 0;
    for (size_t idx = 0; idx < len_of_dataset; idx++)
    {
        dx = dataset[idx][0] - cluster_list[0].x;
        dy = dataset[idx][1] - cluster_list[0].y;
        min_distance = EUCLIDEAN(dx, dy);
        dataset[idx][2] = 0;
        for (size_t cluster_idx = 0; cluster_idx < NUM_OF_CLUSTERS; cluster_idx++)
        {
            dx = dataset[idx][0] - cluster_list[cluster_idx].x;
            dy = dataset[idx][1] - cluster_list[cluster_idx].y;
            if (min_distance > EUCLIDEAN(dx, dy))
            {
                // printf("Changed %f, %f g:%f -> g:%ld \n min: %f -> %f\n", \
                // dataset[idx][0], dataset[idx][1], dataset[idx][2], cluster_idx,\
                // min_distance, EUCLIDEAN(dx, dy));
                min_distance = EUCLIDEAN(dx, dy);
                dataset[idx][2] = cluster_idx;
            }
        }
    }
}
    
void reposition_cluster_centers(float** dataset, \
    float cluster_sum_info[NUM_OF_CLUSTERS][3], \
    Cluster* cluster_list, int len_of_dataset){
    int cluster_num = 0;
    for (size_t idx = 0; idx < len_of_dataset; idx++)
    {
        cluster_num = (int) dataset[idx][2];
        cluster_sum_info[cluster_num][0] += dataset[idx][0];
        cluster_sum_info[cluster_num][1] += dataset[idx][1];
        cluster_sum_info[cluster_num][2] += 1;
    }
    
    for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
    {
        if (!cluster_sum_info[idx][2] == 0)
        {
            cluster_list[idx].x = cluster_sum_info[idx][0] / cluster_sum_info[idx][2];
            cluster_list[idx].y = cluster_sum_info[idx][1] / cluster_sum_info[idx][2];
        }
    }
}

int clusters_converged(Cluster* cluster_list, float previous_clusters[NUM_OF_CLUSTERS][2]){
    int cluster_conv_table[NUM_OF_CLUSTERS] = {0};

    for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
    {
        if (cluster_list[idx].x == previous_clusters[idx][0] && 
            cluster_list[idx].y == previous_clusters[idx][1])
        {
            cluster_conv_table[idx] = 1;
        }
    }
    for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
    {
        //printf("%d", cluster_conv_table[idx]);
        if (cluster_conv_table[idx] == 0){
            return 0;
        }
    }
    return 1;
}

void print_tables(float cluster_sum_info[NUM_OF_CLUSTERS][3], Cluster* cluster_list, int epoch){
    printf("%d =================================\n", epoch);
    for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
    {
        printf("x: %f, y: %f, group: %d\n", cluster_list[idx].x, \
        cluster_list[idx].y, cluster_list[idx].group);
    }
    printf("\n");
    for (size_t i = 0; i < NUM_OF_CLUSTERS; i++)
    {
        printf("%f, %f, %f \n", cluster_sum_info[i][0], cluster_sum_info[i][1],  cluster_sum_info[i][2]);
    } 
    printf("\n\n");
}

void write_labeled_dataset_to_file(char* filename, float** dataset, int len_dataset){
    FILE * fp;
    fp = fopen(filename, "w");
    for (size_t idx = 0; idx < len_dataset; idx++)
    {
        fprintf(fp, "%f, %f, %f\n", dataset[idx][0], dataset[idx][1], dataset[idx][2]);
    }
    fclose(fp);
}

void write_kmeans_clusters_to_file(char* filename, Cluster* cluster_list){
    FILE * fp;
    fp = fopen(filename, "w");
    for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
    {
        fprintf(fp, "%f, %f, %d\n", cluster_list[idx].x, cluster_list[idx].y, cluster_list[idx].group);
    }
    fclose(fp);
}

float* error_calc(Cluster* cluster_list, float** dataset, int len_of_dataset){ 
    float *category_sum = (float*) calloc(NUM_OF_CLUSTERS, sizeof(float));

    for (size_t idx = 0; idx < len_of_dataset; idx++) 
    {
        int label = dataset[idx][2];
        float dx = dataset[idx][0] - cluster_list[label].x; 
        float dy = dataset[idx][1] - cluster_list[label].y;
        category_sum[label] +=  EUCLIDEAN(dx, dy);
    }
    return category_sum;
}


float kmeans(int num)
{  
    char* filename = "../data/dataset2.txt";
    float** dataset;
    dataset = read_file(filename, UNLABELED_SET);
    int len_dataset = get_file_len(filename);
    //printf("Lines of data: %d \n", len_dataset);
  
    float previous_clusters[NUM_OF_CLUSTERS][2] = {0};
    float cluster_sum_info[NUM_OF_CLUSTERS][3] = {0};
    intialize_clusters(cluster_list, dataset, len_dataset);
    //print_tables(cluster_sum_info, cluster_list, -1);
    for (size_t epoch = 0; epoch < 50; epoch++)
    {
        reset_array(cluster_sum_info);
        set_labels(dataset, len_dataset);
        reposition_cluster_centers(dataset, cluster_sum_info, cluster_list, len_dataset);
        //print_tables(cluster_sum_info, cluster_list, epoch);
        if(clusters_converged(cluster_list, previous_clusters) == 1){
            break;
        }
        for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
        {
            previous_clusters[idx][0] = cluster_list[idx].x;
            previous_clusters[idx][1] =  cluster_list[idx].y;
        }
        
    }
    //print_tables(cluster_sum_info, cluster_list, -1);
    write_labeled_dataset_to_file("../out/labeled_data.txt", dataset, len_dataset);
    write_kmeans_clusters_to_file("../out/kmeans_clusters.txt", cluster_list);
    float* error_table = error_calc(cluster_list, dataset, len_dataset);
    float total_error = 0.0;
    for (int i = 0; i < NUM_OF_CLUSTERS; i++)
    {   
        total_error += error_table[i];
        //printf("Error: %f at cluster %d.\n", error_table[i], i);
    }
    printf("Total error: %f\n", total_error);
    
    free(dataset);
    return total_error;
    
}