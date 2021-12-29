#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "utility.h"

//For the 1-7 tasks of ex.1

#define D 5
#define K 4
#define H1 4
#define H2 3
#define H3 2
#define ACTIVATION_FUNC "relu" 

#define ERROR_PROPABILITY 0.1
#define RAND_LOW -1
#define RAND_HIGH 1

int load_files(){
    FILE *training_set_fp, *test_set_fp, *dataset2_fp;
	training_set_fp = fopen("./training_set.txt", "w");
	test_set_fp = fopen("./test_set.txt", "w");
	dataset2_fp = fopen("./dataset2.txt", "w");
}
