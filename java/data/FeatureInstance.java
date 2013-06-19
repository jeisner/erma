package data;

import java.util.ArrayList;

public class FeatureInstance {
	private ArrayList<Feature> feat_group;
	private ArrayList<RV> vars;
	private int[] values;
	private double weight;
	
	public double getWeight() {
		return weight;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}

	public FeatureInstance(ArrayList<Feature> featGroup, ArrayList<RV> v) {
		feat_group = featGroup;
		vars = v; 
		weight=1.0;
//		System.out.println("New feature instance");
//		for(RV var:vars){
//			System.out.println("\t"+var);
//		}
	}
	
	public FeatureInstance(ArrayList<Feature> featGroup, ArrayList<RV> v, ArrayList<Integer> vals) {
		feat_group = featGroup;
		vars = v; 
		values = new int[vals.size()];
		for(int i=0;i<vals.size();i++){
			values[i]=vals.get(i);
		}
		weight = 1.0;
	}
	public FeatureInstance(ArrayList<Feature> featGroup, ArrayList<RV> v, double weight) {
		feat_group = featGroup;
		vars = v; 
		this.weight = weight;
//		System.out.println("New feature instance");
//		for(RV var:vars){
//			System.out.println("\t"+var);
//		}
	}
	
	public FeatureInstance(ArrayList<Feature> featGroup, ArrayList<RV> v, ArrayList<Integer> vals, double weight) {
		feat_group = featGroup;
		vars = v; 
		values = new int[vals.size()];
		for(int i=0;i<vals.size();i++){
			values[i]=vals.get(i);
		}
		this.weight = weight;
	}
	public String toString(){
		String result = "";
		for(int f=0; f<feat_group.size(); f++){
			Feature feat = feat_group.get(f);
			result += feat.getOriginalName() + " (";
			for(int i=0; i<vars.size(); i++){
				result+= vars.get(i).getName();
				if(i!=vars.size()-1)
					result+= ", ";
			}
			if(weight==1.0)
				result+= ")\n";
			else
				result+= ")="+weight+"\n";
		}
		return result;
	}
	public ArrayList<Feature> getFeatures(){ return feat_group; }
	public ArrayList<RV> getVariables() { return vars;}
	public int getValue(int i){
		return values[i];
	}

	public int[] getValues() {
		return values;
	}
}
