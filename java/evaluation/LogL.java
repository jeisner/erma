package evaluation;

import java.util.HashMap;

import inference.BeliefPropagation;
import inference.InferenceAlgorithm;
import utils.Real;
import data.Edge;
import data.Factor;
import data.FactorGraph;
import data.Probability;
import data.RV;
import data.VariableSet;
import data.RV.VisibilityType;
import decoder.Decoder;
import decoder.Identity;
import decoder.Max;

public class LogL extends LossFunction {


	@Override
	public Real evaluate(InferenceAlgorithm ia) {
		//Used a clamped version of the original fg to marginalize out the hidden variables
	    FactorGraph fg1 = new FactorGraph(ia.getFactorGraph());

		for(RV v:fg1.getVariables()){
			if(!v.isHidden()){
				//clamp all but the hidden variables to their values for the second fg
				v.setVisType(VisibilityType.INPUT);
			}
		}
		HashMap<String, Object> properties= ia.getProperties();
		//properties.put("verbose", 5);
		BeliefPropagation bp = new BeliefPropagation( fg1, properties);
		bp.run();
//		System.out.println("Variable marginals:");
//		for(int i = 0; i < fg1.numVariables(); i++ ) // iterate over all variables in fg
//			System.out.println(fg1.getVariable(i)+": "+bp.beliefV(i)); // display the belief of bp for that variable
//		System.out.println("Factor marginals:");
//		for(int i = 0; i < fg1.numFactors(); i++ ) // iterate over all variables in fg
//			System.out.println(fg1.getFactor(i)+": "+bp.beliefF(i)); // display the belief of bp for that variable
		Real statePot = bp.logZ();

		
	    Real logZ = ia.logZ();
	    //System.out.println("StatePot="+statePot+" ia.logZ="+logZ+" logZ="+logZ.minus(statePot));
		return logZ.minus(statePot);
	}
	

	@Override
	public Decoder getDecoder() {
		return new Identity();
	}


	@Override
	public void revEvaluate(InferenceAlgorithm ia) {
		((BeliefPropagation)ia).setRunReverse(false); 
		//Used a clamped version of the original fg to marginalize out the hidden variables
	    FactorGraph fg1 = new FactorGraph(ia.getFactorGraph());

		for(RV v:fg1.getVariables()){
			if(!v.isHidden()){
				//clamp all but the hidden variables to their values for the second fg
				v.setVisType(VisibilityType.INPUT);
			}
		}
		HashMap<String, Object> properties= ia.getProperties();
		//properties.put("verbose", 5);
		BeliefPropagation bp = new BeliefPropagation( fg1, properties);
		bp.run();
		FactorGraph hidFg = ia.getFactorGraph();
		for(int I = 0; I < hidFg.numFactors(); I++ ) {
			Factor fI = hidFg.getFactor(I);
			Probability adj = new Probability(fI.states(),0.0);
			Probability belFI = ia.beliefF(I);
			Probability hid_bp_belF_I = bp.beliefF(I);
			//cout << "factor "<<I << endl;
			for(int Si = 0; Si < fI.states(); Si++ ) {
				//int ind = getFactorEntryForSt(hid_fg,I,sample);

				VariableSet vars = fI.getVars();
				boolean incons = false;
			    for(RV v :vars) {
			        
			        incons |= (v.isInput())&&(v.getValue()!=getVarEntryForSt(fI,v,Si));
				}

			    if(incons){
			    	adj.setValue(Si,(belFI.getValue(Si).divide(ia.getFactorGraph().getFactor(I).getCondTable().getValue(Si))));
			    }else{
			    	//Real base = Si==ind?(Real)1.0:(Real)0.0;
			    	//Real t = ;
			    	adj.setValue(Si,(belFI.getValue(Si).minus(hid_bp_belF_I.getValue(Si)).divide(ia.getFactorGraph().getFactor(I).getCondTable().getValue(Si))));
			    	//if(I==14&&3==Si)
			    	//	cout << "adj[14,3]=-("<<base<<"-"<<jt.beliefF(I)[Si]<<")/"<<_fg->factor(I).p()[Si]<<"="<<adj[Si]<<endl;
			    }

			}
			//cout << "push back" << endl;
			ia.getFactorGraph().getFactor(I).setGradient(adj);
		}
		//cout << "done computing addjoints" << endl;

	}
	int getVarEntryForSt(Factor f, RV v, int entry){
		int f_entry = entry;
	    for( int _j = 0; _j<f.nrNeighbors(); _j++ ) {
	    	Edge<Factor, RV> nj = f.getNeighbor(_j);
	        if(nj.getSecond().equals(v)){
	        	return f_entry % nj.getSecond().numValues();
	        }
	        f_entry /=nj.getSecond().numValues();
	    }
	    return 0;
	}

}
