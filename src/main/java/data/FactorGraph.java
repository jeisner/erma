package data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import utils.Utils;

public class FactorGraph {
	private ArrayList<Factor> factors;
	protected ArrayList<RV> variables;
	//ArrayList<GraphNode> nodes;

	private HashMap<Factor,Integer> factorRefs;
	private HashMap<String, Integer> variableRefs;

	public FactorGraph(){
		factors = new ArrayList<Factor>();
		variables = new ArrayList<RV>();
		factorRefs = new HashMap<Factor, Integer>();
		variableRefs = new HashMap<String, Integer>();
	}
	
	public FactorGraph( ArrayList<Factor> factors) {
		this.factors = new ArrayList<Factor>();
		this.variables = new ArrayList<RV>();
		this.factorRefs = new HashMap<Factor, Integer>();
		this.variableRefs = new HashMap<String, Integer>();
		for( Factor fac : factors ) {
			this.addFactor(fac);
		}
	}

	public FactorGraph(FactorGraph fg){
		factors = new ArrayList<Factor>();
		variables = new ArrayList<RV>();
		factorRefs = new HashMap<Factor, Integer>();
		variableRefs = new HashMap<String, Integer>();
		HashMap<RV,RV> vars = new HashMap<RV, RV>();
		for(RV v:fg.variables){
			RV v1 = new RV(v);
			vars.put(v, v1);
		}
		int i=0;
		for(Factor f:fg.factors){
			VariableSet fv = new VariableSet();
			for(RV v:f.getRVs()){
				fv.add(vars.get(v));
			}
			Factor f1 = new Factor(fv,i);
			i++;
			f1.setCondTable(new Probability(f.getCondTable()));
			addFactor(f1);
		}
	}
	public void addFactor(Factor f){
		for(RV v: f.getRVs()){
			if(!variableRefs.containsKey(v.getName())){
				v.setFgNum(variables.size());
				variables.add(v);
				//nodes.add(v);
				variableRefs.put(v.getName(), variableRefs.size());
				
			}
			GraphNode.addEdge(v, f);
		}
		f.setFgNum(factors.size());
		factors.add(f);
		//nodes.add(f);
		factorRefs.put(f, factors.size()-1);
	}

	public int numVariables(){
		return variables.size();
	}
	public int numFactors(){
		return factors.size();
	}
	public ArrayList<RV> getVariables(){
		return variables;
	}
	public ArrayList<Factor> getFactors(){
		return factors;
	}
	public RV getVariable(int i){
		return variables.get(i);
	}
//	public GraphNode getNode(int i){
//		return nodes.get(i);
//	}
	public Factor getFactor(int I){
		return factors.get(I);
	}

	public int getVariableIndex(String vj) {
		//System.out.println(variableRefs);
		return variableRefs.get(vj);
	}
	public String toDebugString(){
		String result = "variables:\n";
		for(RV v:variables){
			result += v+"\n";
			for(Edge e: v.getNeighbors()){
				result += "\t"+e.getSecond()+"\n";
			}
		}
		result+="factors:\n";
		for(Factor f:factors){
			result += f+"\n";
			for(Edge e: f.getNeighbors()){
				result += "\t"+e.getSecond()+"\n";
			}
		}
		return result;
	}
	
	public int numActiveFactors(){
		int result = 0;
		for(Factor f:factors){
			//System.out.println(f.toString()+f.getCondTable());
			boolean hidden = false;
			for(Edge<Factor,RV> e:f.getNeighbors()){
				RV v = e.getSecond();
				hidden |= !v.isInput();
			}
			if(hidden){
				Probability condTable = f.getCondTable();
				if(!condTable.sameValue()){
					//System.out.println("result+="+condTable.size());
					result+=condTable.size();
				}
			}
		}
		return result;
	}
}
