public class runMlp {

    public static void main(String[] args) {
        int numOfHiddenLayers = 3; // type "2" or "3"
        int D = 2;
        int H1 = 10;
        int H2 = 10;
        int H3 = 8; // Ignored if numOfHiddenLayers == 2
        int K = 4;
        String hiddenLayerActivationFunction = "tanh"; //type "relu" or "tanh"
        double LEARNING_RATE = 0.003;
        int BATCH_SIZE = 1;
        int MINIMUM_EPOCHS = 700;
        double TERMINATION_THRESHOLD = 0.01;

        Mlp mlp = new Mlp(numOfHiddenLayers, D, H1, H2, H3, K, LEARNING_RATE, BATCH_SIZE,
                MINIMUM_EPOCHS, TERMINATION_THRESHOLD, hiddenLayerActivationFunction);

        mlp.initWeights();
        long startTime = System.nanoTime();
        mlp.gradientDescent("training_set.txt");
        long endTime   = System.nanoTime();
        long totalTime = endTime - startTime;
        System.out.printf("\nTraining runtime: %.3fsec", totalTime/Math.pow(10,9));
        mlp.testNetwork("test_set.txt");

    }
}
