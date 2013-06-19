package decoder;

import data.Probability;
import data.RV;

public class Identity extends MarginalDecoder {

	@Override
	public Probability decodeMarginal(RV v, Probability bel) {
		return new Probability(bel);
	}

	@Override
	public Probability reverseDecodeMarginal(RV v, Probability bel, double beta) {
		return new Probability(v.getGadient());
	}

	@Override
	public Probability softDecodeMarginal(RV v, Probability bel, double beta) {
		return new Probability(bel);
	}
	public String toString(){
		return "Identity";
	}

}
