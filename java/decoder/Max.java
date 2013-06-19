package decoder;

import utils.Real;
import data.FactorGraph;
import data.Probability;
import data.RV;
import inference.InferenceAlgorithm;

public class Max extends MarginalDecoder {

	@Override
	public Probability decodeMarginal(RV v, Probability bel) {
		Probability decode = new Probability(v.numValues(),0.0);
		decode.setValue(bel.argmax(),1.0);
		return decode;
	}

	@Override
	public Probability reverseDecodeMarginal(RV v, Probability bel, double beta) {
		Probability oldGrad = v.getGadient();
		Probability grad = new Probability(v.numValues(),0.0);
		Probability mod = bel.softArgmax(beta);
		for(int i = 0; i<grad.size(); i++){
			for(int j = 0; j<grad.size(); j++)
				if(i==j)
					//adj[i]+=orig[i]==0.0?0.0:adj_orig[i]*(beta*(1.0-mod[i])*(mod[i]))/orig[i];
					grad.setValue(i, grad.getValue(i).sumEquals(bel.getValue(i).getValue()==0.0?
							new Real(0.0):
								oldGrad.getValue(i).product(new Real(1.0).minus(mod.getValue(i))).product(mod.getValue(i)).product(beta).divide(bel.getValue(i))));
				else
					//adj[i]-=orig[i]==0.0?0.0:adj_orig[j]*(beta*mod[i]*mod[j])/orig[i];
					grad.setValue(i, grad.getValue(i).sumEquals(bel.getValue(i).getValue()==0.0?new Real(0.0):oldGrad.getValue(j).product(mod.getValue(i).product(mod.getValue(j).product(-beta))).divide(bel.getValue(i))));
		}
		return grad;
	}

	@Override
	public Probability softDecodeMarginal(RV v, Probability bel, double beta) {
//		System.out.println("Max:"+bel+" beta="+beta);
		Probability decode = bel.softArgmax(beta);
//		Probability result = new Probability(decode.size());
//		for(int i=0; i<decode.size(); i++){
//			result.setValue(i, bel.getValue(i));
//		}
//		v.setDecode(decode);
//		System.out.println("Decode:"+decode);
		return decode;
	}
	public String toString(){
		return "Max";
	}
}
