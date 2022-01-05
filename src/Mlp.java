import java.util.ArrayList;

public class Mlp {
    int numberOfLayers;
    int D = 2;
    int H1 = 2;
    int H2 = 2;
    int H3 = 2;
    int K = 4;
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
    public static void main(String[] args) {

        Mlp mlp = new Mlp(2);
        Neuron n1 = new Neuron();
        System.out.println(mlp.layers);
    }
}
