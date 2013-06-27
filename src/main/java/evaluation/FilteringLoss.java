package evaluation;

import inference.InferenceAlgorithm;
import utils.Real;
import data.FactorGraph;
import data.Probability;
import data.RV;
import decoder.Decoder;
import decoder.Filter;
import decoder.Identity;
import decoder.Max;

public class FilteringLoss extends LossFunction {
	protected boolean annealed = true;
	protected double alpha = .5;
	//Filtering loss for a structured prediction cascade
	public FilteringLoss(){
		super();
	}
	public FilteringLoss(double alpha){
		this.alpha=alpha;
	}
	@Override
	public Real evaluate(InferenceAlgorithm ia) {
		FactorGraph fg = ia.getFactorGraph();
		Real loss = new Real(0.0);
		double denum = 0.0;
		for(int i=0; i<fg.numVariables(); i++){
			RV v = fg.getVariable(i);
			if(v.isOutput()){
				//For each output RV, this loss is a combination of two factors
				// l = alpha*bel[output]
				denum+=1.0;
				Probability decode = v.getDecode();
				Real bel = decode.getValue(v.getValue());
				Real exLoss = ((new Real(1.0).minus(bel)).product(v.getWeight()*alpha));
				exLoss = exLoss.divide(v.getWeight());
				Real totalValues = new Real(0.0);
				for(int s = 0; s<decode.size(); s++){
					totalValues.sumEquals(decode.getValue(s));
				}
				// l += (1-alpha)*bel[output]
				exLoss.sumEquals(totalValues.divide((double)decode.size()).product(1-alpha));
				loss.sumEquals(exLoss);
			}
		}
		return loss.divide(denum);
	}
	

	@Override
	public Decoder getDecoder() {
		return new Filter();
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
        		
        		double mult = one_over_n*(1.0-alpha)/(double)v.numValues();
        		for(int s=0;s<v.numValues();s++)
        			v.getGadient().setValue(s, mult);

        		int val = v.getValue();
        		Real varW = new Real(v.getWeight());
        		//Probability orig = ia.beliefV(v.getFgNum());
        		v.getGadient().incrValue(val, varW.product(-one_over_n*alpha));
        	}
        }
	}

}
