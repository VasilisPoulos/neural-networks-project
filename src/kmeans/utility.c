#include "utility.h"

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
	else{ return -1.0; }
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

float* parse_line(char* line, int type){
    char *next_token;
    float* coords = malloc(2*sizeof(float*));
	int size = 2;
	if (type == LABELED_SET)
	{
		size = 3;
	}
    next_token = strtok(line,",");
    for (size_t idx = 0; idx < size; idx++)
    {
        float number = strtof(next_token, NULL);
        next_token = strtok (NULL, ",");
        coords[idx] = number;
    }
    return coords;
}

float** read_file(char* filename, int type) {
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
        coords = parse_line(line, type);
        dataset[idx][0] = coords[0];
        dataset[idx][1] = coords[1];
		if (type == LABELED_SET){
			dataset[idx][2] = coords[2];
		}else{
			dataset[idx][2] = -1;
		}
    }

    fclose(fp);
    if (line)
        free(line);
    
    return dataset;
}

// Robert Jenkins' 96 bit Mix Function
unsigned long mix(unsigned long a, unsigned long b, unsigned long c)
{
    a=a-b;  a=a-c;  a=a^(c >> 13);
    b=b-c;  b=b-a;  b=b^(a << 8);
    c=c-a;  c=c-b;  c=c^(b >> 13);
    a=a-b;  a=a-c;  a=a^(c >> 12);
    b=b-c;  b=b-a;  b=b^(a << 16);
    c=c-a;  c=c-b;  c=c^(b >> 5);
    a=a-b;  a=a-c;  a=a^(c >> 3);
    b=b-c;  b=b-a;  b=b^(a << 10);
    c=c-a;  c=c-b;  c=c^(b >> 15);
    return c;
}
