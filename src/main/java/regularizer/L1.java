package regularizer;

import java.util.ArrayList;

import utils.Real;
import data.Probability;

public class L1 extends Regularizer {

	@Override
	public ArrayList<Real> gradEvaluate(ArrayList<Real> parVec, Real beta) {
		ArrayList<Real> result = new ArrayList<Real>();
		for(int i=0; i<parVec.size(); i++){
			Real res = parVec.get(i).gt(0.0)?beta:Real.unMinus(beta);
			result.add(res);
		}
		return result;
	}

	@Override
	public Real evaluate(ArrayList<Real> params, Real beta) {
		Real result=new Real(0.0);
		for(int j=0;j<params.size(); j++){
			result.sumEquals(beta.product(new Real(Math.abs(params.get(j).getValue()),params.get(j).getAccum())));
		}
		return result;
	}
	public String toString(){
		return "L1";
	}

	@Override
	public ArrayList<Real> gradEvaluate(ArrayList<Real> params, Real beta,
			int steps, int batchSize, double rate) {
		ArrayList<Real> result = new ArrayList<Real>();
		for(int i=0; i<params.size(); i++){
			Real res = params.get(i).gt(0.0)?beta:Real.unMinus(beta);
			res = res.product(steps*batchSize);
			result.add(res);
		}
		return result;
	}
}
