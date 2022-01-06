#include "kmeans.h"

void intialize_clusters(float** dataset, int len_of_dataset){
    unsigned long seed = mix(clock(), time(NULL), getpid());
    srand(seed); 
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
    
void reposition_cluster_centers(float** dataset, float cluster_sum_info[NUM_OF_CLUSTERS][3], int len_of_dataset){
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

int clusters_converged(float previous_clusters[NUM_OF_CLUSTERS][2]){
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

void print_tables(float cluster_sum_info[NUM_OF_CLUSTERS][3], int epoch){
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

void write_kmeans_clusters_to_file(char* filename){
    FILE * fp;
    fp = fopen(filename, "w");
    for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
    {
        fprintf(fp, "%f, %f, %d\n", cluster_list[idx].x, cluster_list[idx].y, cluster_list[idx].group);
    }
    fclose(fp);
}

float intra_cluster_variance(float** dataset, int len_of_dataset){ 
    float intra_cluster_variance = 0.0;
    float cluster_variance[NUM_OF_CLUSTERS] = {0};
    for (size_t idx = 0; idx < len_of_dataset; idx++) 
    {
        int label = dataset[idx][2];
        float dx = dataset[idx][0] - cluster_list[label].x; 
        float dy = dataset[idx][1] - cluster_list[label].y;
        cluster_variance[label] +=  EUCLIDEAN(dx, dy);
    }

    for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
    {
        intra_cluster_variance += cluster_variance[idx];
    }
    
    return intra_cluster_variance;
}

float kmeans(char* filename, int max_iter)
{  
    float** dataset;
    float error = 0.0;
    dataset = read_file(filename, UNLABELED_SET);
    int len_dataset = get_file_len(filename);
    float previous_clusters[NUM_OF_CLUSTERS][2] = {0};
    float cluster_sum_info[NUM_OF_CLUSTERS][3] = {0};
    intialize_clusters(dataset, len_dataset);
    for (size_t epoch = 0; epoch < max_iter; epoch++)
    {
        reset_array(cluster_sum_info);
        set_labels(dataset, len_dataset);
        reposition_cluster_centers(dataset, cluster_sum_info, len_dataset);
        if(clusters_converged(previous_clusters) == 1){
            break;
        }
        for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
        {
            previous_clusters[idx][0] = cluster_list[idx].x;
            previous_clusters[idx][1] =  cluster_list[idx].y;
        }
        
    }
    write_labeled_dataset_to_file("../../out/labeled_data.txt", dataset, len_dataset);
    write_kmeans_clusters_to_file("../../out/kmeans_clusters.txt");
    error = intra_cluster_variance(dataset, len_dataset);
    free(dataset);
    return error;
}
