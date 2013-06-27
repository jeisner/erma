package evaluation;

import inference.InferenceAlgorithm;
import utils.Real;
import data.FactorGraph;
import data.Probability;
import data.RV;
import decoder.Decoder;
import decoder.Identity;
import decoder.MSEDecoder;
import decoder.Max;

public class MSE extends LossFunction {
	protected boolean annealed = false;

	@Override
	public Real evaluate(InferenceAlgorithm ia) {
		FactorGraph fg = ia.getFactorGraph();
		Real loss = new Real(0.0);
		double denum = 0.0;
		for(int i=0; i<fg.numVariables(); i++){
			RV v = fg.getVariable(i);

			if(v.isOutput()){
				if(Double.isNaN(v.getDecode().getValue(v.getValue()).getValue()))
					System.out.println(v.getDecode()+"\t "+v.getDecode().getValue(v.getValue()));

				Real bel = v.getDecode().getValue(v.getValue());
				loss.sumEquals((new Real(1.0).minus(bel).product(new Real(1.0).minus(bel))).product(v.getWeight()));
				denum += v.getWeight();
			}
		}
		if(denum==0)
			return new Real(0.0);
		return loss.divide(denum);
	}
	

	@Override
	public Decoder getDecoder() {
		return new Identity();//MSEDecoder();
	}


	@Override
	public void revEvaluate(InferenceAlgorithm ia) {
		FactorGraph fg = ia.getFactorGraph();
        Real num_vars= new Real(0);
        for(RV v:fg.getVariables()) {
        	if(v.isOutput()){
        		num_vars.sumEquals(v.getWeight());
            }
        }
        double one_over_n=1.0/num_vars.getValue();

        for(RV v:fg.getVariables()) {
        	v.setGradient(new Probability(v.numValues(),0.0));
        	if(v.isOutput()){
        		int val = v.getValue();
        		Real varW = new Real(v.getWeight());
        		Probability orig = ia.beliefV(v.getFgNum());
        		//grad[val]=var_w*one_over_n*((Real)2*b-(Real)2);
        		v.getGadient().setValue(val, varW.product(one_over_n).product(orig.getValue(val).product(2.0).minus(2.0)));
        	}
        }
	}

}
