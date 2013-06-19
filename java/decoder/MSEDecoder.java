package decoder;

import java.util.ArrayList;

import utils.Real;

import data.Probability;
import data.RV;

public class MSEDecoder extends MarginalDecoder {

	@Override
	public Probability decodeMarginal(RV v, Probability bel) {
		if(bel.size()<=2)
			return new Probability(bel);
		Probability input = new Probability(bel);

		Probability resultV = optimizeMSE(input);
		for(int i=0; i<bel.size(); i++)
			bel.setValue(i, resultV.getValue(i));
		return bel;
	}
	private Probability optimizeMSE(Probability in){
		//System.out.println(in);
		Probability result = new Probability(in.size(),0.0);
		Probability p_not = new Probability(in.size(), 0.0);
		Real denom = new Real(0.0);
		for(int i=0; i<in.size(); i++){
			p_not.setValue(i, 1.0);
			for(int j=0; j<in.size(); j++){
				if(i!=j)
					try{
					p_not.setValue(i,
							p_not.getValue(i).product(
									in.getValue(j)));
					}catch(Exception e){
						System.out.println(i+": "+p_not.getValue(i)+" "+j+": "+in.getValue(j));
						throw new RuntimeException(e);
					}
			}
			denom.sum(p_not.getValue(i));
		}
		for(int i=0; i<in.size(); i++){
			result.setValue(i, denom.sum(p_not.getValue(i).product(1.0-in.size())).divide(denom));
			if(result.getValue(i).getValue()<=0.0){
				Probability interm = new Probability(in.size(),0.0);
				for(int j=0; j<in.size();j++){
					if(j!=i)
						interm.setValue(j,in.getValue(j));
				}
				Probability intermResult = optimizeMSE(interm);
				for(int j=0; j<in.size();j++){
					if(j==i)
						result.setValue(j,new Real(0.0));
					result.setValue(j,intermResult.getValue(j));
				}
				break;
			}
		}
		return result;

	}
	

	@Override
	public Probability reverseDecodeMarginal(RV v, Probability bel, double beta) {
		/* TO DO: only works for binary RVs now */
		return new Probability(v.getGadient());
	}

	@Override
	public Probability softDecodeMarginal(RV v, Probability bel, double beta) {
		return decodeMarginal(v, bel);
	}
	public String toString(){
		return "Identity";
	}

}
