import java.util.ArrayList;
import java.lang.Math;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Mlp {
    int numberOfLayers;
    int D = 2;
    int H1 = 10;
    int H2 = 8;
    int H3 = 4;
    int K = 4;
    int BIAS = 1;
    ArrayList<Integer> layerSize = new ArrayList<>();
    double LEARNING_RATE = 0.005;
    int BATCH_SIZE = 1;
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
        return Math.tanh(input);
    }

    private double hiddenLayerDerivative(double input){
        return tanh_derivative(input);
    }

    private double outputLayerFunction(double input){
        return sig(input);
    }

    private double outputLayerDerivative(double input){
        return sig_derivative(input);
    }

    public void initWeights(){
        int numOfNextLayerNeurons;
        double randomWeight;
        for (int layerId = 0; layerId < numberOfLayers + 1; layerId++)
        {
            numOfNextLayerNeurons = layers.get(layerId + 1).size();

            for (Neuron neuron: layers.get(layerId)){
                neuron.biasWeight = getRandomNumber(-1, 1);
                neuron.biasDerivative = 0.0;
                for (int weightId = 0; weightId < numOfNextLayerNeurons; weightId++) {
                    randomWeight = getRandomNumber(-1, 1);
                    neuron.weights.add(randomWeight);
                    neuron.derivatives.add(0.0);
                }
            }
        }
        for (Neuron neuron: layers.get(numberOfLayers + 1)){
            neuron.weights.add(1.0);
            neuron.biasWeight = getRandomNumber(-1, 1);
        }
    }

    private double[] forwardPass(double[] networkInput){
        double weightedSum;
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
                    currentNeuron.input = weightedSum + currentNeuron.biasWeight * BIAS;

                    //Use the activation function to calculate the output of the current neuron.
                    if(layerId == numberOfLayers + 1){
                        currentNeuron.output =
                                outputLayerFunction(currentNeuron.input);
                        networkOutput[neuronId] = currentNeuron.output;
                    }else{
                        currentNeuron.output =
                                hiddenLayerFunction(currentNeuron.input);
                    }
                }
            }
        }
        return networkOutput;
    }

    private void backprop(double[] networkInput, double[] data_label){

        Neuron lastNeuron;
        Neuron currentNeuron;
        Neuron nextNeuron;
        double weightedSum;
        double updatedDerivative;

        forwardPass(networkInput);

        // Last layer error calculation.
        for (int neuronId = 0; neuronId < layerSize.get(numberOfLayers + 1); neuronId++)
        {
            lastNeuron = layers.get(numberOfLayers + 1).get(neuronId);
            lastNeuron.error =
                outputLayerDerivative(lastNeuron.output)
                    * (lastNeuron.output - data_label[neuronId]);
        }

        for (int hiddenLayerId = numberOfLayers; hiddenLayerId >= 0; hiddenLayerId--)
        {
            for (int neuronId = 0; neuronId < layerSize.get(hiddenLayerId); neuronId++)
            {
                weightedSum = 0.0;
                currentNeuron = layers.get(hiddenLayerId).get(neuronId);

                // Calculating error using next layers' neurons.
                for (int weightId = 0; weightId < layerSize.get(hiddenLayerId + 1); weightId++)
                {
                    nextNeuron = layers.get(hiddenLayerId + 1).get(weightId);
                    weightedSum += currentNeuron.weights.get(weightId) * nextNeuron.error;
                }
                currentNeuron.error = hiddenLayerDerivative(currentNeuron.input) * weightedSum;

                // Calculating partial derivatives.
                currentNeuron.biasDerivative += currentNeuron.error;

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
                System.out.println("Weights: " + neuron.weights);
                System.out.println("Derivatives: " + neuron.derivatives);
                System.out.println("Bias Weight: " + neuron.biasWeight);
                System.out.println("Input:" + neuron.input);
                System.out.println("Output:" + neuron.output);
                System.out.println("Bias Weight: " + neuron.biasWeight);
                System.out.println();
            }
            System.out.println();
        }
    }

    public double squareError(double[] data_label){
        Neuron currentNeuron;
        double outputError = 0.0;
        for (int neuronId = 0; neuronId < K; neuronId++) {
            currentNeuron = layers.get(numberOfLayers+1).get(neuronId);
            outputError += Math.pow((data_label[neuronId] - currentNeuron.error), 2);
        }

        return outputError;
    }

    private double outputCategory(double[] output){
        int category = 0;
        double value = 0.0;
        for (int i = 0; i < K; i++)
        {
            if(output[i] > value){
                category = i;
                value = output[i];
            }
        }
        return (float) category;
    }

    private ArrayList<Double[]> readFile(String fileName){
        ArrayList<Double[]> dataSetArray = new ArrayList<>();
        String[] lineSplit;
        try {
            File file = new File(fileName);
            Scanner myReader = new Scanner(file);
            int index = 0;
            while (myReader.hasNextLine()) {
                String data = myReader.nextLine();
                lineSplit = data.split(",");
                dataSetArray.add(new Double[3]);
                dataSetArray.get(index)[0] = Double.parseDouble(lineSplit[0]);
                dataSetArray.get(index)[1] = Double.parseDouble(lineSplit[1]);
                dataSetArray.get(index)[2] = Double.parseDouble(lineSplit[2]);
                index++;
            }
            myReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        return dataSetArray;
    }

    private void testNetwork(){
        ArrayList<Double[]> testDataSet = readFile("../../data/test_set.txt");
        double[] inputData = new double[2];
        double[] output;
        double category;
        int incorrect = 0;
        double errorPercentage;
        for (Double[] data: testDataSet) {
            inputData[0] = data[0];
            inputData[1] = data[1];
            output = forwardPass(inputData);
            category = outputCategory(output);
            if(category != data[2]){
                incorrect++;
            }
        }
        errorPercentage = ((double)incorrect/(double)testDataSet.size())* 100;
        System.out.println("Error: " + errorPercentage + "%");
    }

    private double[] covertLabelToArray(double label){
        double[] labelArray = new double[K];
        for (double num: labelArray) {
            num = 0;
        }

        labelArray[(int)label] = 1;
        return labelArray;
    }

    private void gradientDescent(ArrayList<Double[]> trainingDataset){
        double[] data = new double[D];
        double[] labelArray;
        double total_error;
        double previous_total_error = 0.0;
        int epoch = 0;

        while(true)
        {

            total_error = 0.0;
            for (int i = 0; i < 4000; i++)
            {
                labelArray = covertLabelToArray(trainingDataset.get(i)[2]);
                data[0] = trainingDataset.get(i)[0];
                data[1] = trainingDataset.get(i)[1];
                backprop(data, labelArray);

                if(i % BATCH_SIZE == 0){
                    update_weights();
                }
                total_error += squareError(labelArray);
            }

            total_error = 0.5 * total_error;

            System.out.println("Epoch " + epoch + " Error: " + total_error);
            epoch++;
            if(epoch > MINIMUM_EPOCHS && Math.abs(previous_total_error - total_error)< TERMINATION_THRESHOLD ){
                break;
            }
            previous_total_error = total_error;

        }
        //printLayerInfo();
    }

    public static void main(String[] args) {
        Mlp mlp = new Mlp(3);
        ArrayList<Double[]> trainingSet = mlp.readFile("../../data/training_set.txt");
        mlp.initWeights();
        //mlp.backprop(input, label);
        //mlp.printLayerInfo();
        mlp.gradientDescent(trainingSet);
        mlp.testNetwork();

    }
}
