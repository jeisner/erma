package regularizer;

import java.util.ArrayList;

import data.Probability;

import utils.Real;

public abstract class Regularizer {
	public abstract Real evaluate(ArrayList<Real> params, Real beta);
	public abstract ArrayList<Real> gradEvaluate(ArrayList<Real> params, Real beta);
	public abstract ArrayList<Real> gradEvaluate(ArrayList<Real> params, Real beta,int steps, int batchSize, double rate);
}
