#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

float generate_random_float(float lowest, float highest) {
	float scale = rand() / (float) RAND_MAX;
	return lowest + scale * (highest - lowest);
}

float get_tanhf(float anglerad){
	float result = tanhf(anglerad);
	return result;
}

float relu(float input){
	if (input > 0){
		return 1.0;
	}
	else{ return 0.0; }
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

int get_file_len(char* filename){
    FILE * fp;
    fp = fopen(filename, "r");
    int len_of_dataset = get_num_of_lines(fp);
    fclose(fp);
    return len_of_dataset;
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