#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "utility.h"

#define D 2
#define H1 2
#define H2 2
#define H3 2
#define K 2
#define LEARNGING_RATE 0.1
#define EPOCHS 700

#define ACTIVATION_FUNC "relu" 
#define TANH(x) tanhf(x)
#define RELU(x) x > 0 ? 1.0 : -1.0
#define SIG(x) 1/(1 + exp(-x))

#define RELU_DERIVATIVE(x) x > 0 ? 1.0 : 0.0
#define SIG_DERIVATIVE(x) x*(1-x)
#define TANH_DERIVATIVE(x) 1 - tanh(x)*tanh(x)

#define HIDDEN_LAYER_ACT_FUNC(x) RELU(x)
#define OUTPUT_LAYER_ACT_FUNC(x) SIG(x)

#define NUM_OF_HIDDEN_LAYERS 2
#define NUM_OF_LAYERS NUM_OF_HIDDEN_LAYERS + 2
#define BIAS 1

typedef struct{
	float input;
	float output;
	float* weights;
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
void print_layer_info();
void forward_pass(float *x, float **y, int k);
void backprop(float *x, int d, float *t, int k);
void calculate_output_error(float *t, int k);
void covert_num_category_output(float** category_output, int number);
void update_weights(float partial_sum);
float calculate_partial_der_sum();
void gradient_descent(float** training_dataset, int size_of_dataset);

int main(){
	float** training_dataset = read_file("../data/training_set.txt", LABELED_SET);
	float** test_dataset = read_file("../data/test_set.txt", LABELED_SET);
	int training_set_len = get_file_len("../data/training_set.txt");
	
	float *y;
	float array[] = {1, 1};
	float *x = array; 

	initiate_network();
	
	/*

	//for kathe epoch
	// for ola ta paradeimata (4000)
	forward_pass(x, &y, 4);
	print_layer_info();
	// backpass
	// calculate_output_error
	// (t - neuron.error)^2

	// for (size_t i = 0; i < 4; i++)
	// {
	// 	printf("%f\n", y[i]);
	// }
	free(y);

	float *output;
	covert_num_category_output(&output, 1);
	backprop(x, 2, output, K);
	print_layer_info();
	free(output);
	*/
	float data[2] = {0};
	data[0] = training_dataset[0][0];
	data[1] = training_dataset[0][1];
	printf("actual input %f, %f\n", data[0], data[1]);
	srand(time(NULL));
	forward_pass(data, &y, 4);
	print_layer_info();
	// gradient_descent(training_dataset, 4000);

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
				neuron.weights = (float*) calloc(1, sizeof(float));
				neuron.weights[0] = 1;
			}else{
				int weight_list_len = num_of_neurons_per_layer[layer_idx + 1];
				neuron.weights = (float*) calloc(weight_list_len, sizeof(float));
				for (size_t i = 0; i < num_of_neurons_per_layer[layer_idx + 1]; i++)
				{
					neuron.weights[i] = generate_random_float(-1, 1);
				}
			}
			layers[layer_idx][neuron_idx] = neuron;
		}
	}	
}

void print_layer_info(){
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
				printf("Neuron %d-1: %f\n", neuron_idx+1, layers[layer_idx][neuron_idx].weights[0]);	
			}else{
				for (int i = 0; i <  num_of_neurons_per_layer[layer_idx + 1]; i++)
				{
					printf("Neuron %d-%d: %f\n", neuron_idx+1, i+1, layers[layer_idx][neuron_idx].weights[i]);	
				}
			}
			printf("Input: %f\n", layers[layer_idx][neuron_idx].input);
			printf("Output: %f\n", layers[layer_idx][neuron_idx].output);
			printf("Bias weights: %f\n", layers[layer_idx][neuron_idx].bias_weight);
			printf("Neuron error: %f\n\n",  layers[layer_idx][neuron_idx].error);
		}
		printf("\n");
	}
}

void forward_pass(float *x, float **y, int k){
	*y = (float*) malloc(k *sizeof(float));
	float sum = 0.0;
	float input = 0.0;
	float weights = 0.0;
	Neuron previous_neuron;
	for (int layer_idx = 0; layer_idx < NUM_OF_LAYERS; layer_idx++)
	{
		for (int neuron_idx = 0; neuron_idx < num_of_neurons_per_layer[layer_idx]; neuron_idx++)
		{	
			sum = 0.0;	
			if(layer_idx == 0){
				// For the first 'virtual' layer, the networks input is passed 
				// to the layers output.
				layers[layer_idx][neuron_idx].input = x[neuron_idx];
				layers[layer_idx][neuron_idx].output = \
					layers[layer_idx][neuron_idx].input;
			}else{
				// For the hidden layers.
				// Calculate the sum of the neurons' output of the previous 
				// layer.  
				for (size_t previous_idx = 0; previous_idx < num_of_neurons_per_layer[layer_idx-1]; previous_idx++){
					previous_neuron = layers[layer_idx - 1][previous_idx];
					sum += previous_neuron.output * previous_neuron.weights[neuron_idx];
				}
				layers[layer_idx][neuron_idx].input = sum + layers[layer_idx][neuron_idx].bias_weight * BIAS;
				//Add bias of the current neuron
				//Use the activation function to calculate the output of the current neuron
				if(layer_idx == NUM_OF_LAYERS -1){
					layers[layer_idx][neuron_idx].output = 
						OUTPUT_LAYER_ACT_FUNC(sum + layers[layer_idx][neuron_idx].bias_weight * BIAS);
					(*y)[neuron_idx] = layers[layer_idx][neuron_idx].output;
				}else{
					layers[layer_idx][neuron_idx].output = 
						HIDDEN_LAYER_ACT_FUNC(sum + layers[layer_idx][neuron_idx].bias_weight * BIAS);
				} 	
			}	 
		}
	}
}

void backprop(float *x, int d, float *t, int k){
	Neuron next_neuron;
	Neuron neuron;
	float sum = 0.0;
	calculate_output_error(t, k);
	for (size_t layer_idx = NUM_OF_LAYERS-2; layer_idx > 0; layer_idx--)
	{

		for (size_t neuron_idx = 0; neuron_idx < num_of_neurons_per_layer[layer_idx]; neuron_idx++)
		{	
			sum = 0.0;
			neuron = layers[layer_idx][neuron_idx];
			for (size_t weight_idx = 0; weight_idx < num_of_neurons_per_layer[layer_idx + 1]; weight_idx++)
			{
				next_neuron = layers[layer_idx + 1][weight_idx];
				sum += neuron.weights[weight_idx] * next_neuron.error;  
			}
			layers[layer_idx][neuron_idx].error = TANH_DERIVATIVE(neuron.input) * sum;
		}		
	}	
}

void calculate_output_error(float *t, int k){
	Neuron neuron;
	for (size_t neuron_idx = 0; neuron_idx < num_of_neurons_per_layer[NUM_OF_LAYERS-1]; neuron_idx++)
	{
		neuron = layers[NUM_OF_LAYERS-1][neuron_idx];
		layers[NUM_OF_LAYERS-1][neuron_idx].error = SIG_DERIVATIVE(neuron.output) * (neuron.output - t[neuron_idx]);	
	}
	
}

void covert_num_category_output(float** category_output, int number){
	*category_output = (float*)calloc(K, sizeof(float));
	(*category_output)[number-1] = 1.0;	
}

void update_weights(float partial_sum){
	for (int layer_idx = 0; layer_idx < NUM_OF_LAYERS; layer_idx++)
	{
		for (int neuron_idx = 0; neuron_idx < num_of_neurons_per_layer[layer_idx]; neuron_idx++)
		{
			for (int weight_idx = 0; weight_idx < num_of_neurons_per_layer[layer_idx + 1]; weight_idx++)
			{
				layers[layer_idx][neuron_idx].weights[weight_idx] -= LEARNGING_RATE * partial_sum;
			}
			
		}
	}
}

float calculate_partial_der_sum(){
	float sum = 0.0;
	Neuron neuron;
	Neuron next_neuron;
	for (size_t layer_idx = 1; layer_idx < NUM_OF_LAYERS-1; layer_idx++)
	{
		for (size_t neuron_idx = 0; neuron_idx < num_of_neurons_per_layer[layer_idx]; neuron_idx++)
		{
			neuron = layers[layer_idx][neuron_idx];
			for (size_t weight_idx = 0; weight_idx < num_of_neurons_per_layer[layer_idx + 1]; weight_idx++)
			{
				next_neuron = layers[layer_idx + 1][weight_idx];
				sum += next_neuron.error * neuron.output;
			}
			sum += neuron.error; 
		}
	}

	for (size_t neuron_idx = 0; neuron_idx < num_of_neurons_per_layer[NUM_OF_LAYERS-1]; neuron_idx++)
	{
		sum += layers[NUM_OF_LAYERS-1][neuron_idx].error;
	}

	return sum;
}

void gradient_descent(float** training_dataset, int size_of_dataset){
	float *category;
	float data[2] = {0};
	float sum = 0.0;
	int update_counter = 0;

	float *y;

	for (size_t epoch = 0; epoch < 1; epoch++)
	{
		sum = 0.0;
		for (size_t i = 0; i < 100; i++)
		{
			covert_num_category_output(&category, training_dataset[i][2]);
			data[0] = training_dataset[i][0];
			data[1] = training_dataset[i][1];
			//printf("%f %f\n", data[0], data[1]);
			forward_pass(data, &y, 4);
			backprop(data, 2, category, 4);
			sum += calculate_partial_der_sum();

			if(i % 1 == 0){
				update_weights(sum);
				update_counter++;
			}
			
		}
		//update_weights(sum);
		
	}
	print_layer_info();	
	printf("%f\n", sum);
	printf("%d\n", update_counter);
}