package data;

import java.util.Arrays;

import utils.Real;
import utils.Utils;

public class Probability {
	private Real[] prob;
	
	public Probability(int size){
		prob = new Real[size];
	}
	public Probability(int size, Real val){
		prob = new Real[size];
		for (int i = 0; i < prob.length; i++) {
			prob[i]=val;
		}
	}
	public Probability(int size, double val){
		prob = new Real[size];
		for (int i = 0; i < prob.length; i++) {
			prob[i]=new Real(val);
		}
	}
	public Probability(Probability p){
		prob = new Real[p.size()];
		for (int i = 0; i < prob.length; i++) {
			prob[i]=new Real(p.getValue(i));
		}
	}
	public int size(){
		return prob.length;
	}
	public Real getValue(int i){
		//System.out.println(i);
		return prob[i];
	}
	public void setValue(int i,Real r){
		prob[i]=r;
	}
	public void setValue(int i,double r){
		prob[i]=new Real(r);
	}
	public void incrValue(int i,Real r){
		prob[i].sumEquals(r);
	}
	public void incrValue(int i,double r){
		prob[i].sumEquals(r);
	}
	public Real sum(){
		Real result = new Real(0);
		for (int i = 0; i < prob.length; i++) {
			result.sumEquals(prob[i]);
		}
		return result;
	}
	public void divide(Real div){
		for (int i = 0; i < prob.length; i++) {
			prob[i]=prob[i].divide(div);
		}
	}
	public void multiply(Real mult){
		for (int i = 0; i < prob.length; i++) {
			prob[i]=prob[i].product(mult);
		}
	}
	public void multiply(double mult){
		multiply(new Real(mult));
	}
    public Real normalize( ) {
        Real Z = new Real(0);
        Z = sum();
        divide(Z);
        return Z;
    }
    public void fill(Real r){
		for (int i = 0; i < prob.length; i++) {
			prob[i]=r;
		}
    }
    public void fill(double r){
		for (int i = 0; i < prob.length; i++) {
			prob[i]=new Real(r);
		}
    }
	public void multiply(Probability p) {
		for (int i = 0; i < prob.length; i++) {
			prob[i]=prob[i].product(p.getValue(i));
		}
	}
	public void divide(Probability p) {
		for (int i = 0; i < prob.length; i++) {
			prob[i]=prob[i].divide(p.getValue(i));
		}
	}
	public void add(Probability p) {
		for (int i = 0; i < prob.length; i++) {
			prob[i]=prob[i].sum(p.getValue(i));
		}
	}
	public void exp(double e) {
		for (int i = 0; i < prob.length; i++) {
			prob[i]=prob[i].pow(e);
		}
	}	
	public static double absDiff(double o1, double o2){
		return Math.abs(o1-o2);
	}
	public static double distL1( Probability p, Probability q){
		if(p.size()!=q.size())
			throw new RuntimeException("Incompatible lengths "+p.size()+" vs. "+q.size());
		int i;
		double sum=0;
		for (i=0; i<p.size(); i++) 
			sum += absDiff(p.getValue(i).getValue(), q.getValue(i).getValue()); 
		return sum;
	}
	public static double distLInf( Probability p, Probability q){
		if(p.size()!=q.size())
			throw new RuntimeException("Incompatible lengths "+p.size()+" vs. "+q.size());
		int i;
		double sum=0;
		for (i=0; i<p.size(); i++){ 
			double cur = absDiff(p.getValue(i).getValue(), q.getValue(i).getValue());
			sum = sum>cur?sum:cur; 
		}
		return sum;
	}
	public static double distTV( Probability p, Probability q){
		return distL1(p, q)/2.0;
	}
	public static double distKL( Probability p, Probability q){
		assert(p.size()!=q.size());
		int i;
		double sum=0;
		for (i=0; i<p.size(); i++){ 
			double cur;
			double pI = p.getValue(i).getValue();
	        if(pI==0.0)
	            cur=0.0;
	        else
	            cur = pI * (Utils.log0(pI) - Utils.log0(q.getValue(i).getValue()));
	        //System.out.println("cur="+cur+" pI="+pI+" qI="+q.getValue(i).getValue());
			sum += cur; 
		}
		return sum;	
	}
	public static double distHel( Probability p, Probability q){
		if(p.size()!=q.size())
			throw new RuntimeException("Incompatible lengths "+p.size()+" vs. "+q.size());
		int i;
		double sum=0;
		for (i=0; i<p.size(); i++){ 
			double cur = Math.sqrt(p.getValue(i).getValue())-Math.sqrt(q.getValue(i).getValue());
			sum = cur*cur; 
		}
        return sum/2.0;
	}
	/// Returns the Shannon entropy of \c *this, \f$-\sum_i p_i \log p_i\f$
	public Real entropy(){ 
		Real sum = new Real(0);
		for(int i=0; i<prob.length; i++){
			sum=sum.minus(prob[i].product(Real.log(prob[i])));
		}
		return sum; 
	}
	public boolean sameValue(){
		if(prob.length<2)
			return true;
		double val = prob[0].getValue();
		for(int i=0; i<prob.length; i++){
			if(Math.abs(val-prob[i].getValue())>Utils.epsilon){
				//System.out.println(val+" vs. "+prob[i]);
				return false;
			}
		}
		return true;
		
	}
	public boolean equals(Probability p){
		if(prob.length!=p.size())
			return false;
		for(int i=0; i<prob.length; i++){
			if(Math.abs(p.getValue(i).getValue()-prob[i].getValue())>Utils.epsilon){
				//System.out.println(val+" vs. "+prob[i]);
				return false;
			}
		}
		return true;
		
	}
	@Override
	public String toString() {
		if (prob == null)
			return "";
		if (prob.length < 1)
			return "[]";

		int iMax = prob.length-1;
		StringBuilder b = new StringBuilder();
		b.append('[');
		for (int i = 0; ; i++) {
			b.append(Utils.df.format(prob[i].getValue()));
			if (i == iMax)
				return b.append(']').toString();
			b.append(", ");
		}
	}
	public int argmax() {
		int maxI = 0;
		double max = prob[0].getValue();
		for(int i=1; i<prob.length; i++){
			if(prob[i].getValue()>max){
				maxI=i;
				max = prob[i].getValue();
			}
		}
		return maxI;
	}
    
    public double getProb() {
		int maxI = 0;
		double max = prob[0].getValue();
		for(int i=1; i<prob.length; i++){
			if(prob[i].getValue()>max){
				maxI=i;
				max = prob[i].getValue();
			}
		}
		return max;    
    
    }
    
	public Probability softArgmax(double beta){
		Probability softm = new Probability(this);
		softm.exp(beta); 
		softm.normalize(); 
		return softm;
	}
}
