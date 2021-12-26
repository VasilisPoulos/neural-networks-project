#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "utility.h"


#define SET_SIZE 8000
#define ERROR_PROPABILITY 0.1

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
	return 0;
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

int generate_dataset() {
	FILE* training_set_fp, *test_set_fp;
	training_set_fp = fopen("./training_set.txt", "w");
	test_set_fp = fopen("./test_set.txt", "w");

	float x1;
	float x2;
	int category;
	srand(time(0));
	for (int i = 0; i < SET_SIZE; i++)
	{
		x1 = generate_random_float();
		x2 = generate_random_float();
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

int main(int argc, char* argv[]) {
	generate_dataset();
	return 0;
}