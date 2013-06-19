package decoder;

import data.FactorGraph;
import data.Probability;
import data.RV;
import inference.InferenceAlgorithm;

public abstract class MarginalDecoder extends Decoder {

	public abstract Probability decodeMarginal(RV v, Probability bel);
	@Override
	public void decode(InferenceAlgorithm ia) {
		FactorGraph fg = ia.getFactorGraph();
		for(int i=0; i<fg.numVariables();i++){
			RV v=fg.getVariable(i);
			Probability decode = decodeMarginal(v, ia.beliefV(i));
			v.setDecode(decode);
		}
	}

	@Override
	public void reverseDecode(InferenceAlgorithm ia, double beta) {
		FactorGraph fg = ia.getFactorGraph();
		for(int i=0; i<fg.numVariables();i++){
			RV v=fg.getVariable(i);
			Probability grad = reverseDecodeMarginal(v, ia.beliefV(i), beta);
			v.setGradient(grad);
		}
	}

	public abstract Probability softDecodeMarginal(RV v, Probability bel, double beta);
	
	@Override
	public void softDecode(InferenceAlgorithm ia, double beta) {
		FactorGraph fg = ia.getFactorGraph();
		for(int i=0; i<fg.numVariables();i++){
			RV v=fg.getVariable(i);
			Probability decode = softDecodeMarginal(v, ia.beliefV(i), beta);
//			for(int j=0; j<decode.size(); j++)
//				System.out.println(decode.getValue(j));
//			System.out.println("MD: "+v.getDecode());
			v.setDecode(decode);
		}	
	}
	public abstract Probability reverseDecodeMarginal(RV v, Probability bel, double beta);
}
