package evaluation;

import inference.InferenceAlgorithm;
import utils.Real;
import data.FactorGraph;
import data.Probability;
import data.RV;
import decoder.Decoder;
import decoder.Max;

public class L1 extends LossFunction {
	protected boolean annealed = true;

	@Override
	public Real evaluate(InferenceAlgorithm ia) {
		FactorGraph fg = ia.getFactorGraph();
		Real loss = new Real(0.0);
		double denum = 0.0;
		for(int i=0; i<fg.numVariables(); i++){
			RV v = fg.getVariable(i);
			if(v.isOutput()){
				Real bel = v.getDecode().getValue(v.getValue());
				loss.sumEquals((new Real(1.0).minus(bel)).product(v.getWeight()));
				denum += v.getWeight();
			}
		}
		if(denum==0)
			return new Real(0.0);
		return loss.divide(denum);
	}
	

	@Override
	public Decoder getDecoder() {
		return new Max();
	}


	@Override
	public void revEvaluate(InferenceAlgorithm ia) {
		FactorGraph fg = ia.getFactorGraph();
        double num_vars= 0.0;
        for(RV v:fg.getVariables()) {
        	if(v.isOutput()){
        		num_vars+=v.getWeight();
            }
        }
        double one_over_n=1.0/num_vars;

        for(RV v:fg.getVariables()) {
        	v.setGradient(new Probability(v.numValues(),0.0));
        	if(v.isOutput()){
        		int val = v.getValue();
        		Real varW = new Real(v.getWeight());
        		//Probability orig = ia.beliefV(v.getFgNum());
        		v.getGadient().setValue(val, varW.product(-one_over_n));
        	}
        }
	}

}
