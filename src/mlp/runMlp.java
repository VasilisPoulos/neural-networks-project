public class runMlp {

    public static void main(String[] args) {
        int numOfHiddenLayers = 3; // type "2" or "3"
        int D = 2;
        int H1 = 10;
        int H2 = 8;
        int H3 = 8; // Ignored if numOfHiddenLayers == 2
        int K = 4;
        double LEARNING_RATE = 0.0009;
        int BATCH_SIZE = 1;
        int MINIMUM_EPOCHS = 700;
        double TERMINATION_THRESHOLD = 0.1;
        String hiddenLayerActivationFunction = "tanh"; //type "relu" or "tanh"

        Mlp mlp = new Mlp(numOfHiddenLayers, D, H1, H2, H3, K, LEARNING_RATE, BATCH_SIZE,
                MINIMUM_EPOCHS, TERMINATION_THRESHOLD, hiddenLayerActivationFunction);

        mlp.initWeights();
        mlp.gradientDescent("../../data/training_set.txt");
        mlp.testNetwork("../../data/test_set.txt");
    }
}
