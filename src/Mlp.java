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
    double LEARNGING_RATE = 0.01;
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
        return ((input > 0) ? input : 0);
    }

    private double sig(double input){
        return 1/(1 + Math.exp(- input));
    }

    private double hiddenLayerFunction(double input){
        return relu(input);
    }

    private double outputLayerFunction(double input){
        return 0;
    }

    private ArrayList<Double> forwardPass(ArrayList<Double> networkInput){
        double weightedSum = 0;
        double input = 0;
        double weights = 0;
        Neuron previousNeuron;
        Neuron currentNeuron;
        ArrayList<Double> networkOutput;

        for (int layerId = 0; layerId < numberOfLayers; layerId++)
        {
            for (int neuronId = 0; neuronId < layerSize.get(layerId); neuronId++)
            {
                currentNeuron = layers.get(layerId).get(neuronId);
                if(layerId == 0){
                    // For the first 'virtual' layer, the networks input is passed
                    // to the layers output.
                    currentNeuron.input = networkInput.get(neuronId);
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
                    if(layerId == numberOfLayers - 1){
                        currentNeuron.output = outputLayerFunction(currentNeuron.input);
                        networkOutput.add(currentNeuron.output);
                    }else{
                        currentNeuron.output = hiddenLayerFunction(currentNeuron.input);
                    }
                }
            }
            }
        }
    }

    public void initWeights(){
        int numOfnextLayerNeurons;
        double randomWeight = 0.0;
        for (int layerId = 0; layerId < numberOfLayers + 1; layerId++)
        {
            numOfnextLayerNeurons = layers.get(layerId + 1).size();

            for (Neuron neuron: layers.get(layerId)){
                for (int weightId = 0; weightId < numOfnextLayerNeurons; weightId++) {
                    randomWeight = getRandomNumber(-1, 1);
                    neuron.weights.add(randomWeight);
                }
            }
        }
        for (Neuron neuron: layers.get(numberOfLayers + 1)){
            neuron.weights.add(1.0);
        }
    }

    private double getRandomNumber(double lower, double upper){
        return Math.random() * (upper - lower) + lower;
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
                System.out.println();
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {

        Mlp mlp = new Mlp(2);

        mlp.initWeights();
        mlp.printLayerInfo();
    }
}
