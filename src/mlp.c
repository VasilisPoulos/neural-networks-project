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
#define NUM_OF_LAYERS 3
#define TANH(x) tanh(x)
#define RELU(x) x > 0 ? 1.0 : -1.0

typedef struct{
	float input;
	float output;
	float weight;
	float error;
}Neuron;


Neuron first_layer[H1];
Neuron second_layer[H2];
Neuron third_layer[H3]; 
Neuron output_layer[K];
Neuron* layers[NUM_OF_LAYERS + 1];
int* num_of_neurons_per_layer;
int _2layers[] = {H1, H2, K};
int _3layers[] = {H1, H2, H3, K};


// #if THREE_LAYER_NET
// Neuron* layers[3];
// Neuron third_layer[H3]; 
// *layers[0] = first_layer;
// *layers[1] = second_layer;
// *layers[2] = third_layer;
// #else
// Neuron* layers[2];
// *layers[0] = first_layer;
// *layers[1] = second_layer;
// #endif


void initiate_network();
void print_layer_weights();

int main(){
	float** training_dataset = read_file("../data/training_set.txt", LABELED_SET);
	float** test_dataset = read_file("../data/test_set.txt", LABELED_SET);
	int training_set_len = get_file_len("../data/training_set.txt");
	// for (int i = 0; i < training_set_len; i++)
	// {
	// 	printf("%f, %f, %f \n", test_dataset[i][0], test_dataset[i][1],\
	// 	  test_dataset[i][2]);
	// }	
	//printf("%f\n", RELU(-2));
	initiate_network();
	print_layer_weights();
	return 0;
}

void initiate_network(){
	layers[0] = first_layer;
	layers[1] = second_layer;

	if(NUM_OF_LAYERS == 3){
		layers[2] = third_layer;
		layers[3] = output_layer;
		
		num_of_neurons_per_layer = _3layers;
	}else{
		layers[2] = output_layer;
		int layers[] = {H1, H2, K};
		num_of_neurons_per_layer = _2layers;
	}

	for (size_t layer_idx = 0; layer_idx < NUM_OF_LAYERS + 1; layer_idx++)
	{
		for (size_t neuron_idx = 0; neuron_idx < num_of_neurons_per_layer[layer_idx]; neuron_idx++)
		{
			Neuron neuron = {.weight = generate_random_float(-1, 1)};
			layers[layer_idx][neuron_idx] = neuron;
		}
	}	
}

void print_layer_weights(){
	for (int layer_idx = 0; layer_idx < NUM_OF_LAYERS + 1; layer_idx++)
	{
		if(layer_idx == NUM_OF_LAYERS){
			printf("----OUTPUT LAYER----\n");
		}else{
			printf("----LAYER %d----\n", layer_idx + 1);
		}
		for (int neuron_idx = 0; neuron_idx < num_of_neurons_per_layer[layer_idx]; neuron_idx++)
		{
			printf("Neuron%d: %f\n", neuron_idx+1, layers[layer_idx][neuron_idx].weight);
		}
		printf("\n");
	}
}

