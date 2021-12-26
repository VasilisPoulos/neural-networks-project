#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define RAND_LOW -1
#define RAND_HIGH 1

float generate_random_float() {
	float scale = rand() / (float) RAND_MAX;
	return RAND_LOW + scale * (RAND_HIGH - RAND_LOW);
}

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

int generate_dataset() {
	FILE* fp;
	fp = fopen("./s1.txt", "w");
	float x1;
	float x2;
	int category;
	srand(time(0));
	for (size_t i = 0; i < 80; i++)
	{
		x1 = generate_random_float();
		x2 = generate_random_float();
		category = select_category(x1, x2);
		printf("%f,%f,%d\n", x1, x2, category);	
		fprintf(fp, "%f,%f,%d\n", x1, x2, category);
	}
	
	
	
	fclose(fp);
	return 0;
}

int main(int argc, char* argv[]) {
	generate_dataset();
	return 0;
}