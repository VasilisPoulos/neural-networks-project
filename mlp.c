#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "utility.h"

//For the 1-7 tasks of ex.1

#define NUM_OF_INPUTS 5
#define NUM_OF_CATEGORIES 4
#define NUM_OF_NEURALS_FIRST 4
#define NUM_OF_NEURALS_SECOND 3
#define NUM_OF_NEURALS_THIRD 2
#define TYPE "relu" 

#define SET1_SIZE 8000
#define SET2_SIZE 1200
#define ERROR_PROPABILITY 0.1
#define RAND_LOW -1
#define RAND_HIGH 1

int load_files(){
    FILE *training_set_fp, *test_set_fp, *dataset2_fp;
	training_set_fp = fopen("./training_set.txt", "w");
	test_set_fp = fopen("./test_set.txt", "w");
	dataset2_fp = fopen("./dataset2.txt", "w");
}
