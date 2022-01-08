import java.util.ArrayList;

class Neuron{
    public double input;
    public double output;
    public ArrayList<Double> weights = new ArrayList<Double>();
	public ArrayList<Double> derivatives = new ArrayList<Double>();;
	public double biasDerivative;
	public double biasWeight;
	public double error;
}