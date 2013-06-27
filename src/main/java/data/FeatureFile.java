package data;

import inference.InferenceAlgorithm;

import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

import regularizer.L1;
import regularizer.Regularizer;

import utils.Real;

public class FeatureFile extends ParameterStruct {
	static Random rndm = new Random();
	HashMap<String,Type> types = new HashMap<String, Type>();
	HashMap<String,Feature> features = new HashMap<String, Feature>();
	HashMap<String,ArrayList<Feature> > featureGroups = new HashMap<String, ArrayList<Feature>>();
	HashMap<String,ArrayList<Feature> > relations = new HashMap<String, ArrayList<Feature>>();
	String str;
	protected long curTime = 0;
	public double rate;
	public boolean runSMD;
	public double lambda;
	public double mu;
	public int batchSize;
	public Real regBeta;
	public Regularizer r;
	
	public String getString(){
		return str;
	}
	public void setString(String s){
		str=s;
	}
	public void setWeight(String name, Real w){
		//System.out.println("Set weight "+name+"="+w);
		features.get(name).setWeight(w);
	}
	public void setWeight(String name, ArrayList<String> values, Real w){
		name = name+"*";
	    String valName = "";
	     for(String val : values) {
			 if(val=="$") val = "X";
		     valName+="_"+val;
		 }
		name+=valName;
		features.get(name).setWeight(w);
	}
	@Override
	public FeatureFile copy() {
		FeatureFile result = new FeatureFile();
		result.curTime = curTime;
		result.types = types;
		result.features = new HashMap<String, Feature>(features.size());
		for(Entry<String, Feature> feat: features.entrySet()){
			Feature newF = new Feature(feat.getValue());
			newF.setFeatFile(result);
			result.features.put(feat.getKey(), newF);
		}
		result.featureGroups = new HashMap<String, ArrayList<Feature>>();
		for(Entry<String, ArrayList<Feature>> featG: featureGroups.entrySet()){
			ArrayList<Feature> feats = new ArrayList<Feature>();
			for(Feature f:featG.getValue()){
				feats.add(result.features.get(f.getName()));
			}
			result.featureGroups.put(featG.getKey(), feats);	
		}
		result.relations = new HashMap<String, ArrayList<Feature>>();
		for(Entry<String, ArrayList<Feature>> featG: relations.entrySet()){
			ArrayList<Feature> feats = new ArrayList<Feature>();
			for(Feature f:featG.getValue()){
				feats.add(result.features.get(f.getName()));
			}
			result.relations.put(featG.getKey(), feats);	
		}
		result.str = str;
		return result;
	}
	@Override
	public void initializeSMDstructures(Real learn_rate){
		for(Feature feat:features.values()){
			feat.setEta(learn_rate.getValue());
		}
	}

	@Override
	public void accumulateGradient(InferenceAlgorithm ia, Regularizer r, double reg_beta){
		FeatureFactorGraph bpFg = (FeatureFactorGraph)ia.getFactorGraph();
		
		for( int I = 0; I < bpFg.numFactors(); I++ ) {
			ArrayList<HashMap<Feature,Double>> featureRefs = bpFg.getFactorFeatures(I);
			Factor fI = bpFg.getFactor(I);
			//System.out.println(fI.getGradient());
			for( int j = 0; j < fI.getCondTable().size(); j++){
				HashMap<Feature,Double> features = featureRefs.get(j);
				for(Entry<Feature, Double> e: features.entrySet()){
					Feature f = e.getKey();
					f.accumGradient(fI.getGradient().getValue(j).product(fI.getCondTable().getValue(j)).product(e.getValue()));
					if(Double.isNaN(f.getGradient().getValue())){
						throw new RuntimeException("grad+="+bpFg.getFactor(I).getGradient().getValue(j)+"*"+bpFg.getFactor(I).getCondTable().getValue(j));
					}
				} 
			}
		}

	}
	
	@Override
	public double evaluateRegularizer(Regularizer r, double mu) {
		double result=0.0;
		for(Feature f : features.values()){
			Real temp = f.getWeight();
			ArrayList<Real> tempv = new ArrayList<Real>();
			tempv.add(temp);
			result += r.evaluate(tempv,new Real(mu)).getValue();
		}
		return result;
	}
	@Override
	public boolean updateWeights(double rate, double score, boolean runSMD, double lambda,
			double mu, int batchSize, Regularizer r, double regBeta) {
		//add the regularizer
		//System.out.println("Updating weights "+rate);
		this.rate = rate;
		this.runSMD = runSMD;
		this.lambda = lambda;
		this.mu = mu;
		this.batchSize = batchSize;
		this.r = r;
		this.regBeta = new Real(regBeta);
		//boolean done = false;
		//Iterate over the features
		curTime++;
		for(Feature f : features.values()){
			f.updateWeight(this);
		}
		return true;

	}
	
	@Override
	public FactorGraph toFactorGraph(DataSample samp) {
		return samp.toFactorGraph();
	}
	
	public void initializeRandom(){
		for(Feature f:features.values()){
			f.setWeight(new Real(rndm.nextGaussian()*1e-7));
		}
	}
	
	@Override
	public ArrayList<Double> getParams() {
		ArrayList<Double> result = new ArrayList<Double>();
		for(Feature f:features.values()){
			result.add(f.getWeight().getValue());
		}
	    return result;
	}

	public void setParams(ArrayList<Double> params){
		int index = 0;
		for(Feature f:features.values()){
			f.setWeight(new Real(params.get(index)));
			index++;
		}
	}
	public boolean addType(String name, Type t){
		if(types.containsKey(name))
			return false;
		types.put(name,t);
		return true;
	}

	public Type getType(String name){
		return types.get(name);
	}
	public Feature getFeature(String name){
		return features.get(name);
	}

	public boolean addFeatureTypenames(String oname, ArrayList<String> type_names, ArrayList<String> vals ){
		ArrayList<Type> ts = new ArrayList<Type>();
		for(int i=0; i<type_names.size();i++){
			Type t;
			if(type_names.get(i).equalsIgnoreCase("nil"))
				t = Type.getNilType();
			else
				t = types.get(type_names.get(i));
			ts.add(t);
			//cout + "type " +i+" = "+t+endl;
		}
		return addFeature(oname,ts,vals);
	}

	public boolean addFeature(String oname, ArrayList<Type> tps, ArrayList<String> vals ){
//		System.out.println("Adding feature "+oname);
//		for(Type t: tps)
//			System.out.print(t+", ");
//		System.out.println();
//		for(String s: vals)
//			System.out.print(s+", ");
//		System.out.println();
		if(tps.size()!=vals.size())
			return false;
		//count the number of features that need to be added
		int num = 1;
		for(int index=0; index<vals.size(); index++){
			String val = vals.get(index);
			int val_num = val=="*"?tps.get(index).numValues():1;
			num*=val_num;
		}
		Feature[] all_features=new Feature[num];
		//all_features = new feature[num];
		ArrayList<Feature> group=featureGroups.get(oname);
		if(group==null)
			group = new ArrayList<Feature>();
		for(int i=0; i<num; i++){
			Feature feat =  new Feature(oname,oname,tps.size(),this);
			all_features[i]= feat;
			String feat_name = oname+"*";
			int step = 1;
			for(int index=0; index<vals.size(); index++){
				String val = vals.get(index);
				int val_num=0;
				if(val=="*"){
					int new_step = step*tps.get(index).numValues();
					val_num = (i% new_step)/step;
					step = new_step;
					val = tps.get(index).getValName(val_num);
					//cout + val + endl;
				}else if(val=="$"){
					val = "X";
					val_num = -1;
				}else{
					val_num = tps.get(index).getValue(val);
				}

				feat_name+="_"+val;
				all_features[i].setValue(tps.get(index),val_num,index);

				//cout + feat_name + endl;
			}
			//System.out.println("Adding " + feat_name + "\t\t " +all_features[i].toString());
			all_features[i].setName(feat_name);
			all_features[i].setOriginalName(oname);
			//Make sure the current feature is not repeated
			if(features.containsKey(feat_name))
				throw new RuntimeException("Repeated feature: "+feat_name);
			features.put(feat_name,all_features[i]);
			group.add(features.get(feat_name));
			//delete all_features[i];
		}

		featureGroups.put(oname,group);
		//delete [] all_features;
		return true;
	}

	public boolean addRelation(String name, ArrayList<String> types, ArrayList<String> feature_groups ){
		ArrayList<Feature> feats=new ArrayList<Feature>();
		for(int i=0;i<feature_groups.size();i++){
			String group = feature_groups.get(i);
			ArrayList<Feature> group_feats = featureGroups.get(group);
			for(int j=0;j<group_feats.size();j++){
				//Check that the types correspond
				Feature f = group_feats.get(0);
				if(f.size()!=types.size())
					return false;
				for(int k=0;k<types.size();k++){
					if(f.getType(k).getName()!=types.get(k)){
						return false;
					}
				}
				feats.add(group_feats.get(j));
			}
		}
		relations.put(name,feats);
		return true;
	}
	public ArrayList<Feature> getRelation(String name){
		return relations.get(name);
	}
	public ArrayList<Feature> getFeatureGroup(String name){
		return featureGroups.get(name);
	}
	@Override
	public void print(PrintWriter out) {
		if(getString()!=null&&getString()!=""){
			out.println(getString());
			
			for(Feature f:features.values()) {
				out.print(f.getOriginalName() + "[");
				boolean first = true;
				for (int i = 0; i<f.size(); i++) {
					if(!first)
						out.print(",");
					out.print(f.getType(i).getValName(f.getValue(i)));
					first = first && false;
				}
				out.println("]=" + f.getWeight());
			}
		}else{
			out.println("\n" + "types:");
			for (Type t: types.values()) {
				out.println(t);
			}
			out.println("\n" + "features:");
			for (ArrayList<Feature> fl:featureGroups.values()) {
				for(Feature f1:fl){
					Feature f = features.get(f1.getName());
					out.print(f.getOriginalName() + "(");
					boolean first = true;
//					for (int i = 0; i<f.size(); i++) {
//						if(!first)
//							result+= ",";
//						result+= f.getType(i).getValName(f.getValue(i));
//						first = first && false;
//					}
					for (int i = 0; i<f.size(); i++) {
						if(!first)
							out.print(",");
						out.print(f.getType(i).getName());
						first = first && false;
					}
					out.print("):= [");
					first = true;
					for (int i = 0; i<f.size(); i++) {
						if(!first)
							out.print(",");
						out.print(f.getType(i).getValName(f.getValue(i)));
						first = first && false;
					}
					out.println("]=" + f.getWeight().getValue()); 
				}
			}
			out.println();
			out.println("\n" + "relations:");
			for (Entry<String,ArrayList<Feature>> feats:relations.entrySet()) {
				out.print(feats.getKey() + ": ");
				for(Feature f:feats.getValue()){
					out.print(f + ", ");
				}
				out.println();
			}
//			out.println("\n" + "weights:");
//			for (ArrayList<Feature> feats:featureGroups.values()) {
//				for(Feature f1:feats){
//					//out.print(iter1->first + ": ";
//					Feature f = features.get(f1.getName());
//					out.print(f.getOrigName() + "[");
//					boolean first = true;
//					for (int i = 0; i<f.size(); i++) {
//						if(!first)
//							out.print(",");
//						out.print(f.getType(i).getValName(f.getValue(i)));
//						first = first && false;
//					}
//					out.println("]=" + f.getWeight());
//				}
//			}
			out.println();
		}
		out.flush();
		return;
	}
	public String printWeights() {
		String result = "";
		if(getString()!=""){
			System.out.println(getString());
			for (Feature f:features.values()) {
				result += f.getOriginalName()+ "[";
				boolean first = true;
				for (int i = 0; i<f.size(); i++) {
					if(!first)
						result+= ",";
					result+= f.getType(i).getValName(f.getValue(i));
					first = first && false;
				}
				result+= "]=" + f.getWeight().getValue() +"\n";
			}
		}
		return result;
	}
	public Collection<Feature> getFeatures(){
		return features.values();
	}
	public String toString() {
		String result = "";
		if(getString()!=""){
			System.out.println(getString());
			for (Feature f:features.values()) {
				result += f.getOriginalName()+ "[";
				boolean first = true;
				for (int i = 0; i<f.size(); i++) {
					if(!first)
						result+= ",";
					result+= f.getValue(i)<0?"$":f.getType(i).getValName(f.getValue(i));
					first = first && false;
				}
				result+= "]=" + f.getWeight().getValue() +"\n";
			}
		}else{
			result+= "\ntypes:\n";
			for (Type t:types.values()) {
				result+=t+"\n";
			}
			result+= "\nfeatures:\n";
			for (ArrayList<Feature> fl:featureGroups.values()) {
				for(Feature f1:fl){
					Feature f = features.get(f1.getName());
					result+= f.getOriginalName() + "(";
					boolean first = true;
//					for (int i = 0; i<f.size(); i++) {
//						if(!first)
//							result+= ",";
//						result+= f.getType(i).getValName(f.getValue(i));
//						first = first && false;
//					}
					for (int i = 0; i<f.size(); i++) {
						if(!first)
							result+= ",";
						result+= f.getType(i).getName();
						first = first && false;
					}
					result += "):= [";
					first = true;
					for (int i = 0; i<f.size(); i++) {
						if(!first)
							result+= ",";
						result+= f.getType(i).getValName(f.getValue(i));
						first = first && false;
					}
					result+= "]=" + f.getWeight().getValue() +"\n"; 
				}
			}
			result+="\n";
//			for (ArrayList<Feature> fl:featureGroups.values()) {
//				for(Feature f:fl){
//					result+= f.toString() +"\n";
//				}
//			}
			result+= "\nrelations:\n";
			for (String key:relations.keySet()) {
				result+= key+ ": ";
				for(Feature f: relations.get(key)){
					result+= f + ", ";
				}
				result+="\n";
			}
			//result+= "\nweights:\n";

		}
		return result;
	}
	public long getCurrentTime(){
		return curTime;
	}

}
