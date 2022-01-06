import java.util.ArrayList;
import java.lang.Math;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.io.IOException;
import java.io.FileWriter;

public class Mlp {

    private int D;
    private int H1;
    private int H2;
    private int H3;
    private int K;
    private double learningRate;
    private int batchSize;
    private int minimumEpochs;
    private double terminationThreshold;
    private String hiddenLayerActivationFunction;

    private int BIAS = 1;
    private int numberOfHiddenLayers;
    private ArrayList<Integer> layerSize = new ArrayList<>();
    private ArrayList<ArrayList<Neuron>> layers = new ArrayList<>();


    public Mlp(int numberOfHiddenLayers, int D, int H1, int H2, int H3, int K, double learningRate,
               int Bsize, int minEpochs, double termThreshold, String func){
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        this.D = D;
        this.H1 = H1;
        this.H2 = H2;
        this.H3 = H3;
        this.K = K;
        this.learningRate = learningRate;
        this.batchSize = Bsize;
        this.minimumEpochs = minEpochs;
        this.terminationThreshold = termThreshold;
        this.hiddenLayerActivationFunction = func;

        if (numberOfHiddenLayers <= 3){
            layerSize.add(D);
            layerSize.add(H1);
            layerSize.add(H2);
            if (numberOfHiddenLayers == 3){
                layerSize.add(H3);
            }
            layerSize.add(K); 
            for (int layerId = 0; layerId < numberOfHiddenLayers + 2; layerId++)
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

    public void initWeights(){
        int numOfNextLayerNeurons;
        double randomWeight;
        for (int layerId = 0; layerId < numberOfHiddenLayers + 1; layerId++)
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
        for (Neuron neuron: layers.get(numberOfHiddenLayers + 1)){
            neuron.weights.add(1.0);
            neuron.biasWeight = getRandomNumber(-1, 1);
        }
    }

    public void gradientDescent(ArrayList<Double[]> trainingDataset){
        double[] data = new double[D];
        double[] labelArray;
        double total_error;
        double previous_total_error = 0.0;
        int epoch = 0;
        ArrayList<Double> errorArray = new ArrayList<>();
        while(true)
        {

            total_error = 0.0;
            for (int i = 0; i < trainingDataset.size(); i++)
            {
                labelArray = covertLabelToArray(trainingDataset.get(i)[2]);
                data[0] = trainingDataset.get(i)[0];
                data[1] = trainingDataset.get(i)[1];
                backprop(data, labelArray);

                if(i % batchSize == 0){
                    update_weights();
                }
                total_error += squareError(labelArray);
            }

            total_error = 0.5 * total_error;
            errorArray.add(total_error);

            System.out.println("Epoch " + epoch + " Error: " + total_error);
            epoch++;
            if(epoch > minimumEpochs && Math.abs(previous_total_error - total_error)< terminationThreshold){
                break;
            }
            previous_total_error = total_error;

        }
        writeToFile(errorArray);
        //printLayerInfo();

    }

    public void backprop(double[] networkInput, double[] data_label){

        Neuron lastNeuron;
        Neuron currentNeuron;
        Neuron nextNeuron;
        double weightedSum;
        double updatedDerivative;

        forwardPass(networkInput);

        // Last layer error calculation.
        for (int neuronId = 0; neuronId < layerSize.get(numberOfHiddenLayers + 1); neuronId++)
        {
            lastNeuron = layers.get(numberOfHiddenLayers + 1).get(neuronId);
            lastNeuron.error =
                    sig_derivative(lastNeuron.output)
                            * (lastNeuron.output - data_label[neuronId]);
        }

        for (int hiddenLayerId = numberOfHiddenLayers; hiddenLayerId >= 0; hiddenLayerId--)
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

    public double[] forwardPass(double[] networkInput){
        double weightedSum;
        Neuron previousNeuron;
        Neuron currentNeuron;
        double[] networkOutput = new double[K];

        for (int layerId = 0; layerId < numberOfHiddenLayers + 2; layerId++)
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
                    if(layerId == numberOfHiddenLayers + 1){
                        currentNeuron.output =
                                sig(currentNeuron.input);
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

    private void update_weights(){
        Neuron currentNeuron;
        double updatedWeight;
        for (int layerId = 0; layerId < numberOfHiddenLayers + 1; layerId++)
        {
            for (int neuronId = 0; neuronId < layerSize.get(layerId); neuronId++)
            {
                currentNeuron = layers.get(layerId).get(neuronId);
                currentNeuron.biasWeight -= learningRate * currentNeuron.biasDerivative;
                currentNeuron.biasDerivative = 0.0;

                for (int weight_idx = 0; weight_idx < layerSize.get(layerId + 1); weight_idx++)
                {
                    updatedWeight = currentNeuron.weights.get(weight_idx)
                            - learningRate * currentNeuron.derivatives.get(weight_idx);

                    currentNeuron.weights.set(weight_idx, updatedWeight);
                    currentNeuron.derivatives.set(weight_idx, 0.0);
                }
            }
        }
    }

    private double squareError(double[] data_label){
        Neuron currentNeuron;
        double outputError = 0.0;
        for (int neuronId = 0; neuronId < K; neuronId++) {
            currentNeuron = layers.get(numberOfHiddenLayers +1).get(neuronId);
            outputError += Math.pow((data_label[neuronId] - currentNeuron.error), 2);
        }

        return outputError;
    }

    public void testNetwork(){
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
        System.out.printf("\nGeneralization Error: %.3f%%\n", errorPercentage);
    }

    private double sig(double input){
        return 1/(1 + Math.exp(- input));
    }

    private double sig_derivative(double input){
        return input * (1.0 - input);
    }

    private double getRandomNumber(double lower, double upper){
        return Math.random() * (upper - lower) + lower;
    }

    private double hiddenLayerFunction(double input){
        if (hiddenLayerActivationFunction.equals("relu")){
            return input > 0 ? input : 0;
        }else if(hiddenLayerActivationFunction.equals("tanh")){
            return Math.tanh(input);
        }else{
            System.out.println("Unknown activation function!\n" + "Type \"relu\" or \"tanh\"");
            System.exit(-1);
        }
        return 0;
    }

    private double hiddenLayerDerivative(double input){
        if (hiddenLayerActivationFunction.equals("relu")){
            return input > 0 ? 1.0 : 0.0;
        }else if(hiddenLayerActivationFunction.equals("tanh")){
            return 1.0 - (Math.tanh(input) * Math.tanh(input));
        }else{
            System.out.println("Unknown activation function!");
            System.exit(-1);
        }
        return 0;
    }

    public void printLayerInfo(){
        int numOfNeurons;
        Neuron neuron;
        for (int layerId = 0; layerId < numberOfHiddenLayers + 2; layerId++)
        {
            if (layerId == numberOfHiddenLayers + 1){
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

    private double[] covertLabelToArray(double label){
        double[] labelArray = new double[K];
        for (double num: labelArray) {
            num = 0;
        }

        labelArray[(int)label] = 1;
        return labelArray;
    }

    public ArrayList<Double[]> readFile(String fileName){
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

    private void writeToFile(ArrayList<Double> dataArray){
        try {
            File errorFile = new File("../../out/mpl_error.txt");
            if (errorFile.createNewFile()) {
                System.out.println("\nFile created: " + errorFile.getName());
            } else {
                System.out.println("\n" + errorFile.getName() + " already exists. Updated File");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

        try {
            FileWriter errorWriter = new FileWriter("../../out/mpl_error.txt");
            for (Double line: dataArray) {
                errorWriter.write(String.valueOf(line) + "\n");
            }
            errorWriter.close();
            System.out.println("Successfully wrote to the file.");
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }
}
