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
#define NUM_OF_HIDDEN_LAYERS 3
#define NUM_OF_LAYERS NUM_OF_HIDDEN_LAYERS + 2
#define BIAS 1


typedef struct{
	float input;
	float output;
	float* weight;
	float bias_weight;
	float error;
}Neuron;

Neuron input_layer[D];
Neuron first_layer[H1];
Neuron second_layer[H2];
Neuron third_layer[H3]; 
Neuron output_layer[K];
Neuron* layers[NUM_OF_LAYERS];
int* num_of_neurons_per_layer;
int _2layers[] = {D, H1, H2, K};
int _3layers[] = {D, H1, H2, H3, K};

void initiate_network();
void print_layer_weights();
void forward_pass(float *x, float *y, int k);
int main(){
	float** training_dataset = read_file("../data/training_set.txt", LABELED_SET);
	float** test_dataset = read_file("../data/test_set.txt", LABELED_SET);
	int training_set_len = get_file_len("../data/training_set.txt");
	// for (int i = 0; i < training_set_len; i++)
	// {
	// 	printf("%f, %f, %f \n", test_dataset[i][0], test_dataset[i][1],\
	// 	  test_dataset[i][2]);
	// }	

	initiate_network();
	//print_layer_weights();
	float *y;
	float array[] = {0, 1, 0, 1, 1};
	float *x = array;  
	forward_pass(x, y, 4);
	print_layer_weights();
	return 0;
}

void initiate_network(){
	layers[0] = input_layer;
	layers[1] = first_layer;
	layers[2] = second_layer;

	if(NUM_OF_HIDDEN_LAYERS == 3){
		layers[3] = third_layer;
		layers[4] = output_layer;
		num_of_neurons_per_layer = _3layers;
	}else{
		layers[3] = output_layer;
		num_of_neurons_per_layer = _2layers;
	}

	for (size_t layer_idx = 0; layer_idx < NUM_OF_LAYERS; layer_idx++)
	{
		for (size_t neuron_idx = 0; neuron_idx < num_of_neurons_per_layer[layer_idx]; neuron_idx++)
		{
			Neuron neuron;
			neuron.bias_weight = generate_random_float(-1, 1);
			if(layer_idx == NUM_OF_LAYERS - 1){
				neuron.weight = (float*) calloc(1, sizeof(float));
				neuron.weight[0] = 1;
			}else{
				int weight_list_len = num_of_neurons_per_layer[layer_idx + 1];
				neuron.weight = (float*) calloc(weight_list_len, sizeof(float));
				for (size_t i = 0; i < num_of_neurons_per_layer[layer_idx + 1]; i++)
				{
					neuron.weight[i] = generate_random_float(-1, 1);
				}
			}
			layers[layer_idx][neuron_idx] = neuron;
		}
	}	
}

void print_layer_weights(){
	for (int layer_idx = 0; layer_idx < NUM_OF_LAYERS; layer_idx++)
	{
		if(layer_idx == NUM_OF_LAYERS-1){
			printf("----OUTPUT LAYER----\n");
		}else{
			printf("----LAYER %d----\n", layer_idx + 1);
		}
		for (int neuron_idx = 0; neuron_idx < num_of_neurons_per_layer[layer_idx]; neuron_idx++)
		{
			if((layer_idx == NUM_OF_LAYERS-1)){
				printf("Neuron %d-1: %f\n", neuron_idx+1, layers[layer_idx][neuron_idx].weight[0]);	
			}else{
				for (int i = 0; i <  num_of_neurons_per_layer[layer_idx + 1]; i++)
				{
					printf("Neuron %d-%d: %f\n", neuron_idx+1, i+1, layers[layer_idx][neuron_idx].weight[i]);	
				}
			}
			printf("Output: %f\n", layers[layer_idx][neuron_idx].output);
			printf("Bias weight: %f\n\n", layers[layer_idx][neuron_idx].bias_weight);
		}
		printf("\n");
	}
}

void forward_pass(float *x, float *y, int k){
	float sum = 0.0;
	float input = 0.0;
	float weight = 0.0;
	Neuron neuron;
	for (int layer_idx = 0; layer_idx < NUM_OF_LAYERS; layer_idx++)
	{
		sum = 0.0;	
		for (int neuron_idx = 0; neuron_idx < num_of_neurons_per_layer[layer_idx]; neuron_idx++)
		{	
			if(layer_idx == 0){
				layers[layer_idx][neuron_idx].output = x[neuron_idx];
			}else{
				for (size_t i = 0; i < num_of_neurons_per_layer[layer_idx-1]; i++){
					neuron = layers[layer_idx-1][i];
					input = neuron.output;
					weight = neuron.weight[i];
					sum += input * weight + neuron.bias_weight * BIAS;
				}
				layers[layer_idx][neuron_idx].output = sum;	
			}	 
		}
		printf("\n");
	}



}