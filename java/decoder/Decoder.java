package decoder;

import utils.Real;
import data.Probability;
import data.RV;
import evaluation.LossFunction;
import inference.InferenceAlgorithm;

public abstract class Decoder {
	protected double threshold=0.5;

	public abstract void decode(InferenceAlgorithm ia);

	/// A soft or differentiable version of the decoder (needed when the decoder takes max, for instance)
	public abstract void softDecode(InferenceAlgorithm ia, double beta);
	
	public abstract void reverseDecode(InferenceAlgorithm ia, double beta);
	
	public double testBackProp(InferenceAlgorithm bp, LossFunction loss, double beta, double delta, boolean print){
		double diff = 0;
		//Compute finite-differences gradients. First for loss
		Real value = loss.evaluate(bp);
		for(RV v:bp.getFactorGraph().getVariables()){
			Probability bel = bp.beliefV(v.getFgNum());
			Probability grad = new Probability(bel.size(), 0.0);
			for(int j=0; j<bel.size(); j++){
				bel.setValue(j, bel.getValue(j).sum(delta));
				softDecode(bp,beta);
				Real newValue = loss.evaluate(bp);
				grad.setValue(j, newValue.minus(value).divide(delta));
				bel.setValue(j, bel.getValue(j).minus(delta));
			}
			if(print)
				System.out.println(v+": "+grad + " vs. "+v.getGadient().toString());
			diff+=Math.abs(Probability.distL1(grad, v.getGadient()));
		}
		softDecode(bp,beta);
		return diff;
	}

	public void setThreshold(double d) {
		threshold = d;		
	}
}
