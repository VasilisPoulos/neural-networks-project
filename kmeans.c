#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "utility.h"

#define NUM_OF_CLUSTERS 4
// When comparing distances we can omit the square root of the euclidean 
// distance function and we can make it into a macro so we don'thave to pay for 
// function call overhead.
#define EUCLIDEAN(dx,dy) (dx*dx)+(dy*dy) 
#define LEN_2D(x)  (sizeof(x[0]) / sizeof(x[0][0]))

typedef struct {
	double	x;
	double	y;
	int		group;
} Cluster;

Cluster cluster_list[NUM_OF_CLUSTERS];

float** read_file(char* filename);
float* parse_line(char* line);
int get_num_of_lines(FILE *fp);
void set_labels(float** dataset, int len_of_dataset);

int main()
{  
    char* filename = "dataset2.txt";
    float** dataset;
    dataset = read_file(filename);

    FILE * fp;
    int len_of_dataset = 0;
    fp = fopen(filename, "r");
    len_of_dataset = get_num_of_lines(fp);
    fclose(fp);

    printf("Lines of data: %d \n", len_of_dataset);
    srand(time(NULL)); 
    for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
    {
        Cluster cluster;
        int dataset_idx = generate_random_float(0, len_of_dataset - 1);
        cluster.x = dataset[dataset_idx][0];
        cluster.y = dataset[dataset_idx][1];
        cluster.group = idx;
        cluster_list[idx] = cluster;
    }

    for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
    {
        printf("x: %f, y: %f, group: %d\n", cluster_list[idx].x, \
        cluster_list[idx].y, cluster_list[idx].group);
    }

    float temp_centers[NUM_OF_CLUSTERS][3] = {0};
    int cluster_num = 0;
    for (size_t epoch = 0; epoch < 50; epoch++)
    {
        // Reset temp array.
        for(int i = 0; i < NUM_OF_CLUSTERS; i++) {
            for (int j = 0; j < 3; j++) {
                temp_centers[i][j] = 0;  
            }
        }

        // Calculating distances.
        set_labels(dataset, len_of_dataset);

        // Changing cluster positions.
        for (size_t idx = 0; idx < len_of_dataset; idx++)
        {
            cluster_num = (int) dataset[idx][2];
            temp_centers[cluster_num][0] += dataset[idx][0];
            temp_centers[cluster_num][1] += dataset[idx][1];
            temp_centers[cluster_num][2] += 1;
        }
        printf("==\n");
        for (size_t i = 0; i < NUM_OF_CLUSTERS; i++)
        {
            printf("%f, %f, %f \n", temp_centers[i][0], temp_centers[i][1],  temp_centers[i][2]);
        }   
        printf("\n");

        for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
        {
            if (!temp_centers[idx][2] == 0)
            {
                cluster_list[idx].x = temp_centers[idx][0] / temp_centers[idx][2];
                cluster_list[idx].y = temp_centers[idx][1] / temp_centers[idx][2];
            }
        }
       
        for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
        {
            printf("x: %f, y: %f, group: %d\n", cluster_list[idx].x, \
            cluster_list[idx].y, cluster_list[idx].group);
        }

        // Check if clusters moved.

    }

    // for (size_t i = 0; i < len_of_dataset; i++)
    // {
    //     printf("%f, %f, %f\n", dataset[i][0], dataset[i][1], dataset[i][2]);  
    // }

    // for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
    // {
    //     printf("x: %f, y: %f, group: %d\n", cluster_list[idx].x, \
    //     cluster_list[idx].y, cluster_list[idx].group);
    // }

    // // dealocate memory
    // for (size_t idx = 0; idx < len_of_dataset; idx++)
    // {
    //     free(dataset[idx]);
    // }
    // free(dataset);

    return 0;
}

float** read_file(char* filename) {
    FILE * fp;
    char * line = NULL;
    float* coords;
    size_t len = 0;
    ssize_t read;
    int num_of_lines = 0;

    fp = fopen(filename, "r");
    if (fp == NULL)
        exit(-1);

    num_of_lines = get_num_of_lines(fp);

    float* values = calloc(num_of_lines*3, sizeof(float));
    float** dataset = malloc(num_of_lines*sizeof(float*));
    for (int i=0; i< num_of_lines; ++i)
    {
        dataset[i] = values + i*3;
    }

    for (size_t idx = 0; idx < num_of_lines; idx++)
    {
        read = getline(&line, &len, fp);
        coords = parse_line(line);
        dataset[idx][0] = coords[0];
        dataset[idx][1] = coords[1];
        dataset[idx][2] = -1;
    }

    fclose(fp);
    if (line)
        free(line);
    
    return dataset;
}

float* parse_line(char* line){
    char *next_token;
    float* coords = malloc(2*sizeof(float*));

    next_token = strtok(line,",");
    for (size_t idx = 0; idx < 2; idx++)
    {
        float number = strtof(next_token, NULL);
        next_token = strtok (NULL, ",");
        coords[idx] = number;
    }
    return coords;
}

int get_num_of_lines(FILE *fp){
    int lines = 0;
    int ch = 0;
    while(!feof(fp))
    {
        ch = fgetc(fp);
        if(ch == '\n')
        {
            lines++;
        }
    }
    fseek(fp, 0L, SEEK_SET);
    return lines;
}

void set_labels(float** dataset, int len_of_dataset){
    float dx = 0;
    float dy = 0;
    int distance_from_cluster = -1;
    for (size_t idx = 0; idx < len_of_dataset; idx++)
        {
            dx = dataset[idx][0] - cluster_list[0].x;
            dy = dataset[idx][1] - cluster_list[0].y;
            dataset[idx][2] = 0;
            distance_from_cluster = EUCLIDEAN(dx, dy);
            for (size_t cluster_idx = 1; cluster_idx < NUM_OF_CLUSTERS; cluster_idx++)
            {
                dx = dataset[idx][0] - cluster_list[cluster_idx].x;
                dy = dataset[idx][1] - cluster_list[cluster_idx].y;
                if (distance_from_cluster > EUCLIDEAN(dx, dy))
                {
                    distance_from_cluster = EUCLIDEAN(dx, dy);
                    dataset[idx][2] = cluster_idx;
                }
            }
        }
}