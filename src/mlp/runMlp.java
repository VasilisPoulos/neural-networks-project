import java.util.ArrayList;

public class runMlp {

    public static void main(String[] args) {
        Mlp mlp = new Mlp(3);
        mlp.D = 2;
        mlp.H1 = 10;
        mlp.H2 = 8;
        mlp.H3 = 4;
        mlp.K = 4;
        mlp.hiddenLayerActivationFunction = "tanh";
        ArrayList<Double[]> trainingSet = mlp.readFile("../../data/training_set.txt");
        mlp.initWeights();
        //mlp.backprop(input, label);
        //mlp.printLayerInfo();
        mlp.gradientDescent(trainingSet);
        mlp.testNetwork();
    }
}
