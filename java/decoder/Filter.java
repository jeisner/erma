package decoder;

import utils.Real;
import data.FactorGraph;
import data.Probability;
import data.RV;
import inference.InferenceAlgorithm;

public class Filter extends MarginalDecoder {
	public static Real sigmoid(double threshold,double beta,Real v){
		return new Real(1.0).divide(Real.exp(v.minus(threshold).product(-beta)).sum(1.0));
	}
	@Override
	public Probability decodeMarginal(RV v, Probability bel) {
		Probability decode = new Probability(v.numValues(),0.0);
		double threshold = 1.0/((double)bel.size()+1.0);
		for(int i=0; i<bel.size(); i++)
			if(bel.getValue(i).gt(threshold))
				decode.setValue(i, 1.0);
		return decode;
	}

	@Override
	public Probability reverseDecodeMarginal(RV v, Probability bel, double beta) {
		Probability grad = new Probability(v.getGadient());
		double threshold = 1.0/((double)bel.size()+1.0);
		for(int i = 0; i<grad.size(); i++){
			Real y = Real.exp(bel.getValue(i).minus(threshold).product(-beta));
			grad.setValue(i,grad.getValue(i).product(y.product(beta).divide(y.sum(1.0).product(y.sum(1.0)))));
		}
		return grad;
	}

	@Override
	public Probability softDecodeMarginal(RV v, Probability bel, double beta) {
		Probability decode = new Probability(v.numValues(),0.0);
		double threshold = 1.0/((double)bel.size()+1.0);
		for(int i=0; i<bel.size(); i++)
			decode.setValue(i, sigmoid(threshold,beta,bel.getValue(i)));
		return decode;
	}
	public String toString(){
		return "Filter";
	}
}
