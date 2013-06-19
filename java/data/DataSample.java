package data;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.TreeMap;
import java.util.Vector;
import java.util.Map.Entry;

import utils.Real;
import utils.State;

import data.RV.VisibilityType;

public class DataSample {
	private HashMap<String, RV> variables;
	private ArrayList<FeatureInstance> featureInstances;
	private FeatureFile ff;
	private FeatureFactorGraph factorGraph=null;
	public DataSample(FeatureFile ff){
		//System.out.println("New data sample");
		variables=new HashMap<String, RV>();
		featureInstances = new ArrayList<FeatureInstance>();
		this.ff = ff;
	} 
	
	public DataSample(HashMap<String, RV> vars, ArrayList<FeatureInstance> feats){
		variables = vars;
		featureInstances = feats;
	}

	public int size(){ 
		return variables.size();
	}
	
	public RV addVariable(String name, Type t, String value){
		RV var;
		RV existing = getVariable(name);
		if(existing!=null){
			existing.setWeight(existing.getWeight()+1.0);
			var = existing;
		}else{
			var = new RV(variables.size(),name, t);
			int val = 0;
			if(value!="*"){
				if(value.charAt(0)=='*' || value.charAt(0)=='$'){
					val = t.getValue(value.substring(1));
					var.setVisType(VisibilityType.INPUT);
				}else{
					val = t.getValue(value);
					var.setVisType(VisibilityType.OUTPUT);
				}
			}else{
				var.setVisType(VisibilityType.HIDDEN);
			}
			var.setValue(val);
			variables.put(name, var);
		}
		return var;
	}

	public RV getVariable(String v){
		return variables.get(v);
	}
	public boolean addFeatureInst(String varName, String varValue, String featureName,double weight){
		ArrayList<String> varNames = new ArrayList<String>();
		ArrayList<String> varValues = new ArrayList<String>();
		varNames.add(varName);
		varValues.add(varValue);
		ArrayList<Feature> relation = ff.getRelation(featureName);
		if(relation!=null){
				addFeatureInst(varNames,varValues,relation,weight);
		}else{
			ArrayList<Feature> group = ff.getFeatureGroup(featureName);
			if(group!=null){
				addFeatureInst(varNames,varValues,group,weight);
			}else{
				Feature f = ff.getFeature(featureName);

				if(f!=null){
					ArrayList<Feature> feat= new ArrayList<Feature>();
					feat.add(f);
					addFeatureInst(varNames,varValues,feat,weight);
				}else{
					throw new RuntimeException("Feature "+featureName+" not found.");
				}
			}
		}
		return true;
	}
	public boolean addFeatureInst(Collection<String> varNames, Collection<String> varValues, ArrayList<Feature> featGroup, double weight){
//		System.out.print("AddFeatureinstance [");
//		for(String var:varNames){
//			System.out.print(var+",");
//		}
//		System.out.println("]");
		ArrayList<RV> vars = new ArrayList<RV>();
		for(String varName : varNames){
			RV v = getVariable(varName);
			if(v==null)
				return false;
			vars.add(0,v);
		}
		FeatureInstance fi;
		if(varValues==null)
		  fi = new FeatureInstance(featGroup,vars);
		else{
			ArrayList<Integer> varValuesAr = new ArrayList<Integer>();
			Iterator<String> valIter = varValues.iterator();
			for(int i =0; i<varValues.size(); i++){
				String varVal = valIter.next();
				int value = vars.get(vars.size()-1-i).getType().getValue(varVal);
				varValuesAr.add(0,value);
			}
			fi = new FeatureInstance(featGroup, vars, varValuesAr);
		}
		//System.out.println("added feature inst "+fi);
		featureInstances.add(fi);
		return true;
	}
	public boolean addFeatureInst(LinkedHashMap<String, String> varNames, LinkedHashMap<String, String> varValues, String featureName){
		return addFeatureInst(varNames, varValues, featureName, 1.0);
	}
	public boolean addFeatureInst(LinkedHashMap<String, String> varNames, LinkedHashMap<String, String> varValues, String featureName,double weight){
		//System.out.println("AddFeatureinstance "+featureName);
		Collection<String> values = varValues==null?null:varValues.keySet();
		//for(String var:varNames.keySet()){
		//	System.out.println("\t"+var+" -- "+varNames.get(var));
		//}
		ArrayList<Feature> relation = ff.getRelation(featureName);
		if(relation!=null){
				addFeatureInst(varNames.keySet(),values,relation,weight);
		}else{
			ArrayList<Feature> group = ff.getFeatureGroup(featureName);
			if(group!=null){
				addFeatureInst(varNames.keySet(),values,group,weight);
			}else{
				Feature f = ff.getFeature(featureName);

				if(f!=null){
					ArrayList<Feature> feat= new ArrayList<Feature>();
					feat.add(f);
					addFeatureInst(varNames.keySet(),values,feat,weight);
				}else{
					throw new RuntimeException("Feature "+featureName+" not found.");
				}
			}
		}
		return true;
	}


	public String toString( ){
		String result = "";
		for (String vname: variables.keySet()) {
			RV v = variables.get(vname);
			result += v.getType().getName() + " " + vname;
			if(!v.isHidden())
				result += "="+v.getValueName();
			if(v.isInput())
				result += " in";
			result += ";\n";
		}
		result+="\nfeatures:\n";
		HashSet<String> printed = new HashSet<String>();
		for(int i=0; i<featureInstances.size(); i++){
			String featname = featureInstances.get(i).toString();
			int[] featVals = featureInstances.get(i).getValues();
			if(featVals!=null){
				result+= featname.replaceAll("\\n", "") + ":=";
				for(int j=0;j<featVals.length; j++){
					if(j>0) result+=",";
					result+=featureInstances.get(i).getVariables().get(j).getType().getValName(featVals[j]);
				}
				result+=";\n";
			}else{
				if(!printed.contains(featname)){
					result+= featname.replaceAll("\\n", "") + ";\n";
					printed.add(featname);
				}
			}
		}

		return result;
	}
	
	private RV getRV(RV v,HashMap<String,RV> vars){
		if(!vars.containsKey(v.getName())){
			vars.put(v.getName(), new RV(v));
		}
		return vars.get(v.getName());
	}
	public FeatureFactorGraph updateFGWeights(){
		if(factorGraph==null){
			return null;
		}
		for(int I=0; I<factorGraph.numFactors(); I++){
			Factor fac=factorGraph.getFactor(I);
			ArrayList<HashMap<Feature,Double>> feat_refs = factorGraph.getFactorFeatures(I);
			for(int state=0; state<fac.states(); state++){
				fac.getCondTable().setValue(state,1.0);
				HashMap<Feature,Double> feats = feat_refs.get(state);
				for(Entry<Feature, Double> e:feats.entrySet()){
					Feature feat1 = ff.getFeature(e.getKey().getName());
					fac.getCondTable().setValue(state,fac.getCondTable().getValue(state).product(Math.exp(feat1.getWeight().getValue())).product(e.getValue()));
				}
			}
		}
		return factorGraph;
	}
	public FactorGraph toFactorGraph(){
		if(factorGraph!=null){
			return updateFGWeights();
		}
		//Saves the variable set to factor HashMappings
	    HashMap<String,Factor> facs = new HashMap<String, Factor>();
	    HashMap<String,ArrayList<HashMap<Feature,Double> > > featureRefs = new HashMap<String, ArrayList<HashMap<Feature,Double>>>();

		//Go over the features and add each feature to the appropriate cell of the conditional
		//probability table of the appropriate factor
	    HashMap<String,RV> newVars=new HashMap<String, RV>();
	    int next = 0;
	    for(int j=0; j<featureInstances.size(); j++){
	    	FeatureInstance fi = featureInstances.get(j);
	    	//System.out.println("fi --> "+fi);
	    	//System.out.println(fi+"__");
	    	String key = makeKey(fi.getVariables());
	    	Factor fac;
	    	ArrayList<HashMap<Feature,Double>> featRef;
	    	if(!facs.containsKey(key)){
	    		VariableSet Ivars=new VariableSet();
	    		for(RV v:fi.getVariables()){
	    			Ivars.add(getRV(v, newVars));
	    		}
	    		fac = new Factor(next++,Ivars, 1.0);
	    		facs.put(key,fac);
	    		//ArrayList<set<feature* > > feat_r_vec;
	    		featRef = new ArrayList<HashMap<Feature,Double>>();
	    		for(int t=0;t<fac.states();t++){
	    			HashMap<Feature,Double> f=new HashMap<Feature,Double>();
	    			featRef.add(f);
	    		}
	    		featureRefs.put(key, featRef);
	    	}else{
	    		fac = facs.get(key);
	    		featRef = featureRefs.get(key);
	    	}
	    	ArrayList<Feature> featGroup = fi.getFeatures();
	    	for (int t=0; t<featGroup.size(); t++){
	    		Feature feat = featGroup.get(t);
	    		//System.out.println(feat);
	    		//Compute the state corresponding to the variable setting
	    		TreeMap<RV,Integer> varVals=new TreeMap<RV, Integer>();
	    		int k=0;
	    		for(RV v:fi.getVariables()){
	    			int varVal = feat.getValue(k);
	    			if(varVal<0){
	    				//System.out.println("&&&&&&&&&&&&&&&&&&&&&&&&");
	    				varVal = fi.getValue(k);
	    			}
	    			varVals.put(v,varVal);
	    			//System.out.println("variable " + v + " value "+feat.getValue(k)+"="+v.getType().getValName(feat.getValue(k)));
	    			k++;
	    		}
	    		int state = (new State(varVals)).getState();

	    		if(ff!=null){
	    			Feature feat1 = ff.getFeature(feat.getName());
	    			//    		if(feat!=feat1){
	    			//    			cout <<feat.get_name()<<": feat.get_weight()="<<feat.get_weight()<<" feat1.get_weight()="<<feat1.get_weight()<<endl;
	    			//    			cout << &feat << " vs " <<feat1<<endl;
	    			//    			DAI_THROW(RUNTIME_ERROR);
	    			//    		}
	    			/* TO DO: Need to update the derivatives to work with weighted feature instances*/
	    			fac.getCondTable().setValue(state,fac.getCondTable().getValue(state).product(Math.exp(feat1.getWeight().getValue()*fi.getWeight())));
	    		}else{
	    			fac.getCondTable().setValue(state,fac.getCondTable().getValue(state).product(Math.exp(feat.getWeight().getValue()*fi.getWeight())));
	    		}
	    		//System.out.println("Adding "+feat+" w "+feat.getWeight());
	    		featRef.get(state).put(feat,featRef.get(state).containsKey(feat)?featRef.get(state).get(feat)+fi.getWeight():fi.getWeight());
	    	}
	    	//Record that this feature was used for this factor value
	    }

	    ArrayList<Factor> facs_vec=new ArrayList<Factor>();
	    ArrayList<ArrayList<HashMap<Feature,Double>> > feature_ref_vec=new ArrayList<ArrayList<HashMap<Feature,Double>>>();
		for (String factKey:facs.keySet()) {
			Factor fact = facs.get(factKey);
	    	facs_vec.add(fact);
	    	ArrayList<HashMap<Feature,Double> > fr = (featureRefs.get(factKey));
	    	feature_ref_vec.add(fr);
	    }
	    FeatureFactorGraph ffg = new FeatureFactorGraph(facs_vec,feature_ref_vec);
	    //cout << "--de "<<endl;
//	    //HashMap<String, variable>::const_iterator iter;
//	    for (RV v: variables.values()) {
//	    	//cout << fv.label()<<"-."<<ffg.var(fv.label()).label() << endl;
//	    	//DAI_ASSERT(fv.label()==ffg.var(fv.label()).label());
//	    	int val = v.getValue();
//
//	    }
	    //System.out.println("Total of "+ffg.numFactors()+" factors");
	    //factorGraph = ffg;
	    return ffg;

	}
	private String makeKey(ArrayList<RV> variables){
		VariableSet vars = new VariableSet();
		for(RV v:variables)
			vars.add(v);
	    String result = "*";
	    for (RV v: vars){
	    	result += v.getName()+"*";
	    }
	    //System.out.println( "Key: " + result );
		return result;
	}

	public Collection<RV> getVariables() {
		return variables.values();
	}

}
