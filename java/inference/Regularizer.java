package inference;

import java.util.ArrayList;

import data.Probability;

import utils.Real;

public abstract class Regularizer {
	public abstract Real evaluate(ArrayList<Real> params, Real beta);
	public abstract Real dEvaluate(ArrayList<Probability> par_vec, Real beta);
}
