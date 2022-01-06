#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define UNLABELED_SET 0
#define LABELED_SET 1

float generate_random_float(float lowest, float highest);
float get_tanhf(float anglerad);
float relu(float input);
float** read_file(char* filename, int type);
float* parse_line(char* line, int type);
int get_num_of_lines(FILE *fp);
int get_file_len(char* filename);
unsigned long mix(unsigned long a, unsigned long b, unsigned long c);