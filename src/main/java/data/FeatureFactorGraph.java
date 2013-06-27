package data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;

import utils.State;
import utils.Utils;

public class FeatureFactorGraph extends FactorGraph {
	ArrayList<ArrayList<HashMap<Feature,Double>>> featureRefs;
	ArrayList<HashMap<Feature,Double>> getFactorFeatures(int I){
		return featureRefs.get(I);
	}

	FeatureFactorGraph( ArrayList<Factor> facs, ArrayList<ArrayList<HashMap<Feature,Double>> > fr ){ 
		super(facs);
		featureRefs=fr;
	}

	public String toString() {
		String result = "Total of " + numFactors() + " factors.\n";
		for( RV v:variables ){
			result+=v;
			if(v.getValue()>=0)
				result+= "=" + v.getType().getValName(v.getValue());
			result+= " ";
		}
		result +="\n";

		for(int I = 0; I < numFactors(); I++ ) {
			result+="\n";
			for( RV v:getFactor(I).getRVs() )
				result+= v.numValues() + " ";
			result+="\n";
			ArrayList<HashMap<Feature,Double>> feat_refs = featureRefs.get(I);
			//System.out.println(getFactor(I).getCondTable());
			for( int k = 0; k < getFactor(I).states(); k++ ){
				State st = new State(getFactor(I).getVars(),k);
				Map<RV,Integer> state = st.get();
				for (RV v:state.keySet()) {
					result+= v + "=" + v.getType().getValName(state.get(v)) + " ";
					//os << "(" << st_iter->first << "-->"<<st_iter->second << ") ";
				}//   if( fg.factor(I)[k] != (Real)0 )
				result+= "\t" + Utils.df.format(getFactor(I).getCondTable().getValue(k).getValue()) + " [ ";
				HashMap<Feature,Double> feats = feat_refs.get(k);
				Iterator<Feature> it = feats.keySet().iterator();
				for(int i=0; i<5 && it.hasNext(); i++){
					Feature f=it.next();
					result+= f.getOriginalName() + " ";
				}
				result+= "]\n";
			}
		}

		return result;
	}


}
