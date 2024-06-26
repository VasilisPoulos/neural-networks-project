#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "utility.h"

#define D 2
#define H1 4
#define H2 4
#define H3 4
#define K 4
#define LEARNGING_RATE 0.0000001
#define BATCH_SIZE 1
#define MINIMUM_EPOCHS 700
#define TERMINATION_THRESHOLD 0.1
 
#define TANH(x) tanhf(x)
#define RELU(x) x > 0 ? x: 0
#define SIG(x) 1/(1 + exp(-x))

#define RELU_DERIVATIVE(x) x > 0 ? 1.0 : 0.0
#define SIG_DERIVATIVE(x) x*(1.0 - x)
#define TANH_DERIVATIVE(x) 1.0 - (tanhf(x)*tanhf(x))

#define HIDDEN_LAYER_ACT_FUNC(x) RELU(x)
#define OUTPUT_LAYER_ACT_FUNC(x) SIG(x)

#define HIDDEN_LAYER_DERIVATIVE(x) RELU_DERIVATIVE(x)
#define OUTPUT_LAYER_DERIVATIVE(x) SIG_DERIVATIVE(x) 

#define NUM_OF_HIDDEN_LAYERS 3
#define NUM_OF_LAYERS NUM_OF_HIDDEN_LAYERS + 2
#define BIAS 1

typedef struct{
	float input;
	float output;
	float* weights;
	float* derivatives;
	float bias_derivative;
	float bias_weight;
	float error;
}Neuron;

Neuron input_layer[D];
Neuron first_layer[H1];
Neuron second_layer[H2];
Neuron third_layer[H3]; 
Neuron output_layer[K];
Neuron* layers[NUM_OF_LAYERS];
int* layer_size;
int _2layers[] = {D, H1, H2, K};
int _3layers[] = {D, H1, H2, H3, K};

void initiate_network();
void print_layer_info();
void forward_pass(float *x, float **y, int k);
void backprop(float *x, int d, float *t, int k);
void calculate_output_error(float *t, int k);
void covert_label_to_array(float* array, int label);
void update_weights();
void gradient_descent(float** training_dataset, int size_of_dataset);
float square_error(float *t);
void test_network(float** test_dataset, int size_of_dataset);
float output_category(float *output);

int main(){
	float** training_dataset = read_file("../data/training_set.txt", LABELED_SET);
	float** test_dataset = read_file("../data/test_set.txt", LABELED_SET);
	int training_set_len = get_file_len("../data/training_set.txt");

	float array[] = {1, 1};
	float *x = array; 
	float data[2] = {0};
	data[0] = training_dataset[0][0];
	data[1] = training_dataset[0][1];
	float label = training_dataset[0][2];
	srand(time(NULL));
	initiate_network();
	gradient_descent(training_dataset, 4000);
	test_network(test_dataset, 4000);
	return 0;
}

void initiate_network(){
	layers[0] = input_layer;
	layers[1] = first_layer;
	layers[2] = second_layer;

	if(NUM_OF_HIDDEN_LAYERS == 3){
		layers[3] = third_layer;
		layers[4] = output_layer;
		layer_size = _3layers;
	}else{
		layers[3] = output_layer;
		layer_size = _2layers;
	}

	for (size_t layer_idx = 0; layer_idx < NUM_OF_LAYERS; layer_idx++)
	{
		for (size_t neuron_idx = 0; neuron_idx < layer_size[layer_idx]; neuron_idx++)
		{
			Neuron neuron;
			neuron.bias_weight = generate_random_float(-1, 1);
			if(layer_idx == NUM_OF_LAYERS - 1){
				neuron.weights = (float*) calloc(1, sizeof(float));
				neuron.weights[0] = 1;
			}else{
				int weight_list_len = layer_size[layer_idx + 1];
				neuron.weights = (float*) calloc(weight_list_len, sizeof(float));
				neuron.derivatives = (float*) calloc(weight_list_len, sizeof(float));
				for (size_t i = 0; i < layer_size[layer_idx + 1]; i++)
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
		for (int neuron_idx = 0; neuron_idx < layer_size[layer_idx]; neuron_idx++)
		{
			if((layer_idx == NUM_OF_LAYERS-1)){
				printf("Neuron %d-1: %f\n", neuron_idx+1, layers[layer_idx][neuron_idx].weights[0]);	
			}else{
				for (int i = 0; i <  layer_size[layer_idx + 1]; i++)
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
	float weighted_sum = 0.0;
	float input = 0.0;
	float weights = 0.0;
	Neuron previous_neuron;
	Neuron* current_neuron;
	for (int layer_idx = 0; layer_idx < NUM_OF_LAYERS; layer_idx++)
	{
		for (int neuron_idx = 0; neuron_idx < layer_size[layer_idx]; neuron_idx++)
		{		
			current_neuron = &layers[layer_idx][neuron_idx];
			if(layer_idx == 0){
				// For the first 'virtual' layer, the networks input is passed 
				// to the layers output.
				current_neuron->input = x[neuron_idx];
				current_neuron->output = current_neuron->input;
			}else{
				// For the hidden layers.
				// Calculate the weighted sum of the previous layer.
				weighted_sum = 0.0;  
				for (size_t previous_idx = 0; previous_idx < layer_size[layer_idx - 1]; previous_idx++){
					previous_neuron = layers[layer_idx - 1][previous_idx];
					weighted_sum += \
						previous_neuron.output * previous_neuron.weights[neuron_idx];
				}
				current_neuron->input = weighted_sum + current_neuron->bias_weight * BIAS;

				//Use the activation function to calculate the output of the current neuron.
				if(layer_idx == NUM_OF_LAYERS - 1){
					current_neuron->output = OUTPUT_LAYER_ACT_FUNC(current_neuron->input);
					(*y)[neuron_idx] = current_neuron->output;
				}else{
					current_neuron->output = HIDDEN_LAYER_ACT_FUNC(current_neuron->input);
				} 	
			}	 
		}
	}
}

void backprop(float *x, int d, float *t, int k){
	Neuron* next_neuron;
	Neuron* current_neuron;
	float weighted_sum = 0.0;
	float *y;
	
	forward_pass(x, &y, K);
	calculate_output_error(t, k);
	for (size_t hlayer_idx = NUM_OF_HIDDEN_LAYERS; hlayer_idx > -1; hlayer_idx--)
	{
		for (size_t neuron_idx = 0; neuron_idx < layer_size[hlayer_idx]; neuron_idx++)
		{	
			weighted_sum = 0.0;
			current_neuron = &layers[hlayer_idx][neuron_idx];

			// Calculating error on previous layers' neurons.
			for (size_t weight_idx = 0; weight_idx < layer_size[hlayer_idx + 1]; weight_idx++)
			{
				next_neuron = &layers[hlayer_idx + 1][weight_idx];
				weighted_sum += current_neuron->weights[weight_idx] * next_neuron->error;  
			}		
			current_neuron->error = HIDDEN_LAYER_DERIVATIVE(current_neuron->input) * weighted_sum;
			
			// Calculating partial derivatives.
			current_neuron->bias_derivative += current_neuron->error;

			for (size_t weight_idx = 0; weight_idx < layer_size[hlayer_idx + 1]; weight_idx++)
			{
				next_neuron = &layers[hlayer_idx + 1][weight_idx];
				current_neuron->derivatives[weight_idx] += current_neuron->output * next_neuron->error;
			}
		}		
	}	
}

void calculate_output_error(float *t, int k){
	Neuron *neuron;
	for (size_t neuron_idx = 0; neuron_idx < layer_size[NUM_OF_LAYERS - 1]; neuron_idx++)
	{
		neuron = &layers[NUM_OF_LAYERS - 1][neuron_idx];
		neuron->error = \
			OUTPUT_LAYER_DERIVATIVE(neuron->output) * (neuron->output - t[neuron_idx]);
	}	
}

void covert_label_to_array(float* array, int label){
	array[label - 1] = 1;
}

void update_weights(){
	Neuron *currect_neuron;
	Neuron *prev_layer_neuron;
	for (int layer_idx = 0; layer_idx < NUM_OF_LAYERS - 1; layer_idx++)
	{
		for (int neuron_idx = 0; neuron_idx < layer_size[layer_idx]; neuron_idx++)
		{
			currect_neuron = &layers[layer_idx][neuron_idx];
			currect_neuron->bias_weight -= LEARNGING_RATE * currect_neuron->bias_derivative;
			currect_neuron->bias_derivative = 0.0;

			for (int weight_idx = 0; weight_idx < layer_size[layer_idx]; weight_idx++)
			{
				currect_neuron->weights[weight_idx] \
					-= LEARNGING_RATE * currect_neuron->derivatives[weight_idx];
				currect_neuron->derivatives[weight_idx] = 0.0;
			}	
		}
	}
}

void gradient_descent(float** training_dataset, int size_of_dataset){
	float data[2] = {0};
	float label_array[K] = {0};
	float *y;
	float total_error = 0.0;
	float previous_total_error = 0.0;
	long int epoch = 0;

	while(1)
	{
		
		total_error = 0.0;
		for (size_t i = 0; i < 4000; i++)
		{
			memset(label_array, 0, sizeof(label_array));
			covert_label_to_array(label_array, training_dataset[i][2]);
			data[0] = training_dataset[i][0];
			data[1] = training_dataset[i][1];
			backprop(data, 2, label_array, 4);

			if(i % BATCH_SIZE == 0){
				update_weights();
			}	
			total_error += square_error(label_array);
		}
		
		total_error = 0.5 * total_error;
		
		printf("epoch %ld, error: %f\n", epoch + 1, total_error);
		epoch++;
		if(epoch > MINIMUM_EPOCHS && abs(previous_total_error - total_error)< TERMINATION_THRESHOLD ){
			break;
		}
		previous_total_error = total_error;

	}
	print_layer_info();	
}

float square_error(float *t){
	Neuron *current_neuron;
	float output_error = 0.0;
	for (size_t neuron_idx = 0; neuron_idx < K; neuron_idx++)
	{
		current_neuron = &layers[NUM_OF_LAYERS - 1][neuron_idx];
		output_error += pow(t[neuron_idx] - current_neuron->error,2);
	}
	return output_error;	
}

void test_network(float** test_dataset, int size_of_dataset){
	float *output;
	float data[2] = {0};
	float category = 0.0;
	int correct = 0;

	for (size_t i = 0; i < size_of_dataset; i++)
	{	
		data[0] = test_dataset[i][0];
		data[1] = test_dataset[i][1];
		forward_pass(data, &output, 2);
		category = output_category(output);

		printf("%f	%f	%f	%f	%f\n", output[0], output[1], output[2], output[3], category);
		if(category == test_dataset[i][2]){
			correct++;
		}
	}	
	printf("Correct: %.2f%% \n", ((float)correct/(float)size_of_dataset)*100);
}

float output_category(float *output){
	int category = 0;
	float value = 0.0;
	for (size_t i = 0; i < K; i++)
	{
		if(output[i] > value){
			category = i;
			value = output[i];
		}	
	}
	return (float) category;
}