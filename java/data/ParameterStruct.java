package data;

import inference.InferenceAlgorithm;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;

import regularizer.Regularizer;

import utils.Real;

public abstract class ParameterStruct{
	///Create a copy of this struct
	public abstract ParameterStruct copy();

	/// Initialize data structures required for the SMD algorithm
    public abstract void initializeSMDstructures(Real learnRate);

    /// Update weights based on the estimate of the gradient contained in @grad
    public abstract boolean updateWeights(double rate, double score, boolean runSMD, double lambda, double mu, int batch_size, Regularizer r, double reg_beta);

    //Accumulate gradient for an example to the batch
    public abstract void accumulateGradient(InferenceAlgorithm ia, Regularizer r, double reg_beta);

    ///Compute the value of a regularization function for the given parameter values
    public abstract double evaluateRegularizer(Regularizer r, double mu);

    ///Initiate a factor graph for a particular example
    public abstract FactorGraph toFactorGraph(DataSample samp);

    ///Initialize the parameters with random values
    public abstract void initializeRandom();

    ///Get all parameters as a vector of doubles
    public abstract ArrayList<Double> getParams();

    ///Set the parameters to the provided values
    public abstract void setParams(ArrayList<Double> params);

    ///Print the parameter structure
    public abstract void print(PrintWriter out);
    
    public void print(String ofilename){
    	PrintWriter ofile;
    	try {
    		ofile = new PrintWriter(new FileWriter(ofilename));
    	} catch (IOException e) {
    		throw new RuntimeException(e);
    	}
    	print(ofile);
    }
    ///Print only the parameter weights
	public abstract String printWeights();


}
