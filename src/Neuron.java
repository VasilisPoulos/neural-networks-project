import java.util.ArrayList;

class Neuron{
    public double input;
    public double out;
    public ArrayList<Double> weights = new ArrayList<Double>();
	public ArrayList<Double> derivatives = new ArrayList<Double>();;
	public double bias_derivative;
	public double bias_weight;
	public double error;
}