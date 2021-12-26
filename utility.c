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