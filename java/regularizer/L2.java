package regularizer;

import java.util.ArrayList;

import utils.Real;
import data.Probability;

public class L2 extends Regularizer {

	@Override
	public ArrayList<Real> gradEvaluate(ArrayList<Real> parVec, Real beta) {
		ArrayList<Real> result = new ArrayList<Real>();
		for(int i=0; i<parVec.size(); i++){
			Real res = beta.product(2.0).product(parVec.get(i));
			result.add(res);
		}
		return result;
	}

	@Override
	public Real evaluate(ArrayList<Real> params, Real beta) {
		Real result=new Real(0.0);
		for(int j=0;j<params.size(); j++){
			result.sumEquals(beta.product(params.get(j).product(params.get(j))));
		}
		return result;
	}

	public String toString(){
		return "L2";
	}

	@Override
	public ArrayList<Real> gradEvaluate(ArrayList<Real> params, Real beta,
			int steps, int batchSize, double rate) {
		ArrayList<Real> result = new ArrayList<Real>();
		//System.out.println("Reg "+beta+" steps "+steps);
		for(int i=0; i<params.size(); i++){
			Real res = new Real(0);
			Real cur = new Real(params.get(i));
			for(int t=0; t<steps; t++){
				Real reg = beta.product(2.0).product(cur).product(batchSize);
				res.sumEquals(reg);
				cur.sumEquals(reg.product(-rate));
			}
			result.add(res);
		}
		return result;
	}
}
