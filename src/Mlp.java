import java.util.ArrayList;
import java.lang.Math;

public class Mlp {
    int numberOfLayers;
    int D = 2;
    int H1 = 2;
    int H2 = 2;
    int H3 = 2;
    int K = 4;
    int BIAS = 1;
    ArrayList<Integer> layerSize = new ArrayList<>();
    double LEARNING_RATE = 0.01;
    int BATCH_SIZE = 4000;
    int MINIMUM_EPOCHS = 700;
    double TERMINATION_THRESHOLD = 0.1;
    ArrayList<ArrayList<Neuron>> layers = new ArrayList<>();

    public Mlp(int numberOfLayers){
        this.numberOfLayers = numberOfLayers;
        if (numberOfLayers <= 3){  
            layerSize.add(D);
            layerSize.add(H1);
            layerSize.add(H2);
            if (numberOfLayers == 3){
                layerSize.add(H3);
            }
            layerSize.add(K); 
            for (int layerId = 0; layerId < numberOfLayers + 2; layerId++)
            {
                layers.add(new ArrayList<>());
                for (int neuronId = 0; neuronId < layerSize.get(layerId); neuronId++){
                    layers.get(layerId).add(new Neuron());
                }
            }
        }
        else{
            System.exit(-1);
        }
    }

    private double relu(double input){
        return input > 0 ? input : 0;
    }

    private double relu_derivative(double input){
        return input > 0 ? 1.0 : 0.0;
    }

    private double sig(double input){
        return 1/(1 + Math.exp(- input));
    }

    private double sig_derivative(double input){
        return input * (1.0 - input);
    }

    private double tanh_derivative(double input){
        return 1.0 - (Math.tanh(input) * Math.tanh(input));
    }

    private double getRandomNumber(double lower, double upper){
        return Math.random() * (upper - lower) + lower;
    }

    private double hiddenLayerFunction(double input){
        return relu(input);
    }

    private double hiddenLayerDerivative(double input){
        return relu_derivative(input);
    }

    private double outputLayerFunction(double input){
        return sig(input);
    }

    private double outputLayerDerivative(double input){
        return sig_derivative(input);
    }

    public void initWeights(){
        int numOfnextLayerNeurons;
        double randomWeight = 0.0;
        for (int layerId = 0; layerId < numberOfLayers + 1; layerId++)
        {
            numOfnextLayerNeurons = layers.get(layerId + 1).size();

            for (Neuron neuron: layers.get(layerId)){
                neuron.biasWeight = getRandomNumber(-1, 1);
                for (int weightId = 0; weightId < numOfnextLayerNeurons; weightId++) {
                    randomWeight = getRandomNumber(-1, 1);
                    neuron.weights.add(randomWeight);
                }
            }
        }
        for (Neuron neuron: layers.get(numberOfLayers + 1)){
            neuron.weights.add(1.0);
            neuron.biasWeight = getRandomNumber(-1, 1);
        }
    }

    private double[] forwardPass(double networkInput[]){
        double weightedSum = 0;
        Neuron previousNeuron;
        Neuron currentNeuron;
        double[] networkOutput = new double[K];

        for (int layerId = 0; layerId < numberOfLayers + 2; layerId++)
        {
            for (int neuronId = 0; neuronId < layerSize.get(layerId); neuronId++)
            {
                currentNeuron = layers.get(layerId).get(neuronId);
                if(layerId == 0){
                    // For the first 'virtual' layer, the networks input is passed
                    // to the layers output.
                    currentNeuron.input = networkInput[neuronId];
                    currentNeuron.output = currentNeuron.input;
                }else{
                    // For the hidden layers.
                    // Calculate the weighted sum of the previous layer.
                    weightedSum = 0.0;
                    for (int previousId = 0; previousId < layerSize.get(layerId - 1); previousId++){
                        previousNeuron = layers.get(layerId - 1).get(previousId);
                        weightedSum +=
                            previousNeuron.output * previousNeuron.weights.get(neuronId);
                    }
                    currentNeuron.input = weightedSum;

                    //Use the activation function to calculate the output of the current neuron.
                    if(layerId == numberOfLayers + 1){
                        currentNeuron.output =
                                outputLayerFunction(currentNeuron.input + currentNeuron.biasWeight * BIAS);
                        networkOutput[neuronId] = currentNeuron.output;
                    }else{
                        currentNeuron.output =
                                hiddenLayerFunction(currentNeuron.input + currentNeuron.biasWeight * BIAS);
                    }
                }
            }
        }
        return networkOutput;
    }

    private void backprop(double networkInput[], int data_label[]){

        Neuron lastNeuron;
        Neuron currentNeuron;
        Neuron nextNeuron;
        double weightedSum;

        forwardPass(networkInput);
        printLayerInfo();

        // Last layer error calculation.
        for (int neuronId = 0; neuronId < layerSize.get(numberOfLayers + 1); neuronId++)
        {
            lastNeuron = layers.get(numberOfLayers + 1).get(neuronId);
            lastNeuron.error = 
                outputLayerDerivative(lastNeuron.output) 
                    * (lastNeuron.output - data_label[neuronId]);
        }	

        for (int hiddenLayerId = numberOfLayers; hiddenLayerId > 0; hiddenLayerId--)
        {
            for (int neuronId = 0; neuronId < layerSize.get(hiddenLayerId); neuronId++)
            {	
                weightedSum = 0.0;
                currentNeuron = layers.get(hiddenLayerId).get(neuronId);
    
                // Calculating error on previous layers' neurons.
                for (int weightId = 0; weightId < layerSize.get(hiddenLayerId + 1); weightId++)
                {
                    nextNeuron = layers.get(hiddenLayerId + 1).get(weightId);
                    weightedSum += currentNeuron.weights.get(weightId) * nextNeuron.error;  
                }		
                currentNeuron.error = hiddenLayerDerivative(currentNeuron.input) * weightedSum;
                
                // Calculating partial derivatives.
                currentNeuron.biasDerivative += currentNeuron.error;
                double updatedDerivative = 0.0;
                for (int weightId = 0; weightId < layerSize.get(hiddenLayerId + 1); weightId++)
                {
                    nextNeuron = layers.get(hiddenLayerId + 1).get(weightId);
                    updatedDerivative = 
                        currentNeuron.derivatives.get(weightId) 
                        + currentNeuron.output * nextNeuron.error;
                    currentNeuron.derivatives.set(weightId, updatedDerivative);
                }
            }		
        }	
    }

    private void update_weights(){
        Neuron currentNeuron;
        double updatedWeight;
        for (int layerId = 0; layerId < numberOfLayers + 1; layerId++)
        {
            for (int neuronId = 0; neuronId < layerSize.get(layerId); neuronId++)
            {
                currentNeuron = layers.get(layerId).get(neuronId);
                currentNeuron.biasWeight -= LEARNING_RATE * currentNeuron.biasDerivative;
                currentNeuron.biasDerivative = 0.0;
    
                for (int weight_idx = 0; weight_idx < layerSize.get(layerId + 1); weight_idx++)
                {
                    updatedWeight = currentNeuron.weights.get(weight_idx)
                        - LEARNING_RATE * currentNeuron.derivatives.get(weight_idx);

                    currentNeuron.weights.set(weight_idx, updatedWeight);
                    currentNeuron.derivatives.set(weight_idx, 0.0);
                }	
            }
        }
    }

    public void printLayerInfo(){
        int numOfNeurons;
        Neuron neuron;
        for (int layerId = 0; layerId < numberOfLayers + 2; layerId++)
        {
            if (layerId == numberOfLayers + 1){
                System.out.println("------OUTPUT LAYER------");
            }else{
                System.out.println("------LAYER " + layerId + "------");
            }

            numOfNeurons = layers.get(layerId).size();
            for (int neuronId = 0; neuronId < numOfNeurons; neuronId++){
                System.out.println("Neuron " + neuronId);
                neuron = layers.get(layerId).get(neuronId);
                System.out.print("Weights: " + neuron.weights + "\n");
                System.out.println("Input:" + neuron.input);
                System.out.println("Output:" + neuron.output);
                System.out.println("Bias Weight: " + neuron.biasWeight);
                System.out.println();
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {

        Mlp mlp = new Mlp(2);
        mlp.initWeights();
        int label[] = {0, 1, 0, 0};
        double input[] = {-0.911440, 0.186664};
        mlp.backprop(input, label);
        mlp.printLayerInfo();
    }
}
