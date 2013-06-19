package evaluation;

import inference.InferenceAlgorithm;
import utils.Real;
import data.FactorGraph;
import data.Probability;
import data.RV;
import decoder.Decoder;
import decoder.Identity;

public class FactoredLogL extends LossFunction {
	protected boolean annealed = false;

	@Override
	public Real evaluate(InferenceAlgorithm ia) {
		FactorGraph fg = ia.getFactorGraph();
		Real loss = new Real(0.0);
		double denum = 0.0;
		for(int i=0; i<fg.numVariables(); i++){
			RV v = fg.getVariable(i);
			Real bel = v.getDecode().getValue(v.getValue());
			if(v.isOutput()){
				//loss-=var_w*log(b);
				loss.sumEquals(Real.log(bel).product(-v.getWeight()));
				denum += v.getWeight();
			}
		}
		return loss.divide(denum);
	}
	

	@Override
	public Decoder getDecoder() {
		return new Identity();
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
        		//grad[val]= -var_w*one_over_n/b;
        		v.getGadient().setValue(val, varW.product(-one_over_n).divide(orig.getValue(val)));
        	}
        }
	}

}
