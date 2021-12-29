#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "utility.h"


#define SET1_SIZE 8000
#define SET2_SIZE 1200
#define ERROR_PROPABILITY 0.1
#define RAND_LOW -1
#define RAND_HIGH 1

int select_category(float x1, float x2) {
	if (pow((x1 - 0.5), 2) + pow((x2 - 0.5), 2) < 0.16 ){
		return 1;
	}else if (pow((x1 + 0.5), 2) + pow((x2 + 0.5), 2) < 0.16){
		return 1;
	}else if (pow((x1 - 0.5), 2) + pow((x2 + 0.5), 2) < 0.16){
		return 2;
	}else if (pow((x1 + 0.5), 2) + pow((x2 - 0.5), 2) < 0.16){
		return 2;
	}else if ((x1 < 0 && x2 > 0) || (x1 > 0 && x2 < 0)){
		return 3;
	}else {
		return 4;
	}
}

float propability(){
	return ((float)rand() / (float) RAND_MAX);
}

int change_category(int current_category){
	int new_category = (rand() % 5);
	while(new_category == current_category){
		new_category = (rand() % 5);
	}
	return new_category;
}

int generate_dataset_s1() {
	FILE *training_set_fp, *test_set_fp;
	training_set_fp = fopen("../data/training_set.txt", "w");
	test_set_fp = fopen("../data/test_set.txt", "w");

	float x1;
	float x2;
	int category;
	srand(time(0));
	for (int i = 0; i < SET1_SIZE; i++)
	{
		x1 = generate_random_float(RAND_LOW, RAND_HIGH);
		x2 = generate_random_float(RAND_LOW, RAND_HIGH);
		category = select_category(x1, x2);
		if (i < 4000){
			//printf("%d: %f,%f,%d\n", i, x1, x2, category);	
			fprintf(training_set_fp, "%f,%f,%d\n", x1, x2, category);
		}else{
			if (propability() <= ERROR_PROPABILITY)
			{
				category = change_category(category);
			}	
			//printf("%d: %f,%f,%d\n", i, x1, x2, category);	
			fprintf(test_set_fp, "%f,%f,%d\n", x1, x2, category);
		}
		
	}
	fclose(training_set_fp);
	fclose(test_set_fp);
	return 0;
}

int generate_dataset_s2(){
	FILE* dataset2_fp;
	dataset2_fp = fopen("../data/dataset2.txt", "w");
	srand(time(0));

	float x1, x2;
	for (int i = 0; i < SET2_SIZE; i++)
	{	
		if (i < 150){
			x1 = generate_random_float(0.75, 1.25);
			x2 = generate_random_float(0.75, 1.25);
		}else if (i < 300){
			x1 = generate_random_float(0, 0.5);
			x2 = generate_random_float(0, 0.5);
		}else if (i < 450){
			x1 = generate_random_float(0, 0.5);
			x2 = generate_random_float(1.5, 2);
		}else if (i < 600){
			x1 = generate_random_float(1.5, 2);
			x2 = generate_random_float(0, 0.5);
		}else if (i < 750){
			x1 = generate_random_float(1.5, 2);
			x2 = generate_random_float(1.5, 2);
		}else if (i < 825){
			x1 = generate_random_float(0.6, 0.8);
			x2 = generate_random_float(0, 0.4);
		}else if (i < 900){
			x1 = generate_random_float(0.6, 0.8);
			x2 = generate_random_float(1.6, 2);
		}else if (i < 975){
			x1 = generate_random_float(1.2, 1.4);
			x2 = generate_random_float(0, 0.4);
		}else if (i < 1050){
			x1 = generate_random_float(1.2, 1.4);
			x2 = generate_random_float(1.6, 2);
		}else{
			x1 = generate_random_float(0, 2);
			x2 = generate_random_float(0, 2);
		}
		fprintf(dataset2_fp, "%f,%f\n", x1, x2);
		//printf("%d: %f,%f\n", i, x1, x2);
	}
	fclose(dataset2_fp);
	return 0;
}

int main(int argc, char* argv[]) {
	generate_dataset_s1();
	generate_dataset_s2();
	return 0;
}