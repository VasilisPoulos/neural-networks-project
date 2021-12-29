#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

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