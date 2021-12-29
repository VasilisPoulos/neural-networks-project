#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "utility.h"

#define D 5
#define K 4
#define H1 4
#define H2 3
#define H3 2
#define ACTIVATION_FUNC "relu" 

int main(){
	float** training_dataset = read_file("../data/training_set.txt", LABELED_SET);
	float** test_dataset = read_file("../data/test_set.txt", LABELED_SET);
	int training_set_len = get_file_len("../data/training_set.txt");
	for (int i = 0; i < training_set_len; i++)
	{
		printf("%f, %f, %f \n", test_dataset[i][0], test_dataset[i][1],\
		  test_dataset[i][2]);
	}	
	return 0;
}
