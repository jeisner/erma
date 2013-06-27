package data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;

import utils.Index;

public class Factor extends GraphNode{
	private static int nextId=0;
	private VariableSet vars;
	private Probability theta;
	private Probability gradient;
	
	VariableSet getRVs(){
		return vars;
	}
	public Factor(int id){
		this.id = id;
		vars = new VariableSet(); 
		theta = new Probability(getNumStates(vars));
	}
	public Factor(Factor f){
		id = nextId++;
		vars = f.vars;
		theta = new Probability(f.theta);
	}
	public Factor(int id, VariableSet vars){
		this.id = id;
		this.vars = vars;
		theta = new Probability(getNumStates(vars));
	}
	public Factor(VariableSet vars, double w){
		id = nextId++;
		theta = new Probability(getNumStates(vars),w);		
		this.vars = vars;
	}
	public Factor(RV variable, double w) {
		VariableSet vars = new VariableSet();
		vars.add(variable);
		id = nextId++;
		theta = new Probability(getNumStates(vars),w);		
		this.vars = vars;
	}
	public Factor(int id, VariableSet vars, double w) {
		this.id = id;
		theta = new Probability(getNumStates(vars),w);		
		this.vars = vars;
	}
	public static int getNumStates(VariableSet vars){
		int states = 1;
        for( RV v:vars )
            states *= v.numValues();
        return states;
	}
	public Probability getCondTable(){
		return theta;
	}
	public void setCondTable(Probability theta){
		this.theta=theta;
	}
	public String toString(){
		return "f"+id;
	}
	public VariableSet getVars() {
		return vars;
	}
	public int states(){
		return theta.size();
	}
	public Factor marginalize(VariableSet vars, boolean normed){
	    VariableSet res_vars = new VariableSet(vars);
	    res_vars.retainAll(this.vars);

	    Factor result = new Factor( res_vars, 0.0 );

	    Index i_res = new Index( res_vars, this.vars );
	    for( int i = 0; i < theta.size(); i++){
	        result.theta.setValue(i_res.index(), result.theta.getValue(i_res.index()).sum(theta.getValue(i)));
	        i_res.increment();
	    }

	    if( normed )
	        result.theta.normalize();

	    return result;
	}
	public void setGradient(Probability gradient) {
		this.gradient = gradient;
	}
	public Probability getGradient() {
		if(gradient==null)
			gradient = new Probability(theta.size(),0.0);
		return gradient;
	}
}
