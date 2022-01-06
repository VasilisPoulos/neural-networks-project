import java.util.ArrayList;

public class runMlp {

    public static void main(String[] args) {
        int D = 2;
        int H1 = 10;
        int H2 = 8;
        int H3 = 4;
        int K = 4;
        double LEARNING_RATE = 0.005;
        int BATCH_SIZE = 1;
        int MINIMUM_EPOCHS = 700;
        double TERMINATION_THRESHOLD = 0.1;
        String hiddenLayerActivationFunction = "tanh";

        Mlp mlp = new Mlp(3, D, H1, H2, H3, K, LEARNING_RATE, BATCH_SIZE,
                MINIMUM_EPOCHS, TERMINATION_THRESHOLD, hiddenLayerActivationFunction);

        ArrayList<Double[]> trainingSet = mlp.readFile("../../data/training_set.txt");
        mlp.initWeights();
        mlp.gradientDescent(trainingSet);
        mlp.testNetwork();
    }
}
