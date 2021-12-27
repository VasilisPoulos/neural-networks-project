#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    for (size_t idx = 0; idx < NUM_OF_CLUSTERS; idx++)
    {
        Cluster cluster;

        int dataset_idx = generate_random_float(0, len_of_dataset - 1);
        //printf("%d", dataset_idx);
        //printf("%f, %f", dataset[dataset_idx][0], dataset[dataset_idx][1]);
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

    float* values = calloc(num_of_lines*2, sizeof(float));
    float** dataset = malloc(num_of_lines*sizeof(float*));
    for (int i=0; i< num_of_lines; ++i)
    {
        dataset[i] = values + i*2;
    }

    for (size_t idx = 0; idx < num_of_lines; idx++)
    {
        read = getline(&line, &len, fp);
        coords = parse_line(line);
        dataset[idx][0] = coords[0];
        dataset[idx][1] = coords[1];
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