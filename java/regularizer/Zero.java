package regularizer;

import java.util.ArrayList;

import utils.Real;
import data.Probability;

public class Zero extends Regularizer {

	@Override
	public ArrayList<Real> gradEvaluate(ArrayList<Real> params, Real beta) {
		ArrayList<Real> result = new ArrayList<Real>();
		for(int i=0; i<params.size(); i++){
			result.add(new Real(0.0));
		}
		return result;
	}

	@Override
	public Real evaluate(ArrayList<Real> params, Real beta) {
		return new Real(0.0);
	}
	public String toString(){
		return "none";
	}

	@Override
	public ArrayList<Real> gradEvaluate(ArrayList<Real> params, Real beta,
			int steps, int batchSize, double rate) {
		ArrayList<Real> result = new ArrayList<Real>();
		for(int i=0; i<params.size(); i++){
			result.add(new Real(0.0));
		}
		return result;
	}

}
