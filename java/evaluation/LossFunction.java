package evaluation;

import inference.InferenceAlgorithm;
import utils.Real;
import utils.Utils;
import data.FactorGraph;
import data.Probability;
import data.RV;
import decoder.Decoder;

public abstract class LossFunction {
	protected String name = "loss";
	protected boolean useMicroAverage=false;
	protected boolean annealed = false;
	
	public boolean isUseMicroAverage() {
		return useMicroAverage;
	}
	public void setUseMicroAverage(boolean useMicroAverage) {
		this.useMicroAverage = useMicroAverage;
	}
	public LossFunction(){
		name = this.getClass().getName();
		name = name.substring(name.lastIndexOf('.')+1);
	}
	///Evaluate the loss function
	public abstract Real evaluate(InferenceAlgorithm ia);
	///Reverse the evaluation to compute gradient
	public abstract void revEvaluate(InferenceAlgorithm ia);
	
	public abstract Decoder getDecoder();

	public String getName() {
		return name;
	}
	
	public String printLoss(Real loss){
		return name+"="+Utils.df.format(loss.getValue());
	}
	public String printLoss(double loss){
		return name+"="+Utils.df.format(loss);
	}
	public boolean isInterpolated() {
		return false;
	}
	public String toString(){
		return name;
	}
	public boolean isAnnealed() {
		return annealed;
	}
	public double testBackProp(InferenceAlgorithm bp, double beta, double delta, boolean print){
		double diff = 0;
		//Compute finite-differences gradients. First for loss
		Real value = evaluate(bp);
		FactorGraph fg = bp.getFactorGraph();
		for(RV v:fg.getVariables()){
			Probability dec = v.getDecode();
			Probability grad = new Probability(dec.size(), 0.0);
			for(int j=0; j<dec.size(); j++){
				dec.setValue(j, dec.getValue(j).sum(delta));
				Real newValue = evaluate(bp);
				grad.setValue(j, newValue.minus(value).divide(delta));
				dec.setValue(j, dec.getValue(j).minus(delta));
			}
			if(print)
				System.out.println(v+": "+grad + " vs. "+v.getGadient().toString());
			diff+=Math.abs(Probability.distL1(grad, v.getGadient()));
		}
		return diff;
	}
}
