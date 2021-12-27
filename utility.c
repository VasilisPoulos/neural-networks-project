#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

float generate_random_float(float lowest, float highest) {
	float scale = rand() / (float) RAND_MAX;
	return lowest + scale * (highest - lowest);
}