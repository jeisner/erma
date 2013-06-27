package data;

import java.util.ArrayList;

import regularizer.L1;
import regularizer.Regularizer;
import regularizer.Zero;
//import utils.Profiler;
import utils.Real;

public class Feature {
	//Features have two names -- an original (external) name and another name used for internal
	//representation and consisting of the original name +"*"+concatenation of the feature values
	private String name;
	private String orig_name;
	private Type[] types;
	private int[] values;
	private int len;
	private Real w;
	private Real grad;
	private Real reg;
	private double v,eta;
	private FeatureFile ff;
	private long lastUpdate;
	private static int count=0;

	
//	public Feature(){
//		types = new Type[0];
//	}

	public Feature(String oname, String n, int size, FeatureFile ff){
		setOrigName(oname);
		name = n;
		types = new Type[size];
		values = new int[size];
		for(int i=0; i<len; i++){
			values[i]=-1;
		}
		len = size;
		w=new Real(0.0);
		grad=new Real(0.0);
		reg=new Real(0.0);
		v = 0.0;
		eta=0.0;
		count++;
		this.ff = ff;
		lastUpdate = ff.getCurrentTime();
	}
	
	public Feature(Feature f){
		//cout << "feature -- "<<f.name<<endl;
		setOrigName(f.getOrigName());
		name = f.name;
		len = f.len;
		types = new Type[len];
		values = new int[len];
		for(int i=0; i<len; i++){
			types[i]=f.types[i];
			values[i]=f.values[i];
		}
		w=f.getWeight();
		grad=f.grad;
		reg=f.reg;
		v = f.v;
		eta=f.eta;
		count++;
		lastUpdate=f.lastUpdate;
	}
	public void updateWeight(FeatureFile featFile){
		//Profiler.startProcess("update weights");
		boolean use_reg = featFile.r!=null&&!(featFile.r instanceof Zero) ;
		double rate = featFile.rate;
		double old_eta = 0.0;
		
		if(featFile.runSMD){
			old_eta = eta;
			//eta[I][j]=eta[I][j]*max(.5,1-mu*g_I_j.getValue()*v[I][j]);/*/*max(.5,dai::exp(-mu*g_I_j.getValue()*v[I][j]));
			double temp = 1+featFile.mu*(v)*grad.getValue();
			eta =old_eta*Math.min(5.0,Math.max(.5,temp));
			rate=eta;
			//System.out.println("rate="+rate);
		}
		if(use_reg){
			Real temp = w;
			ArrayList<Real> tempv = new ArrayList<Real>();
			tempv.add(temp);
			ArrayList<Real> reg = featFile.r.gradEvaluate(tempv,
					featFile.regBeta,(int)(featFile.getCurrentTime()-lastUpdate),
					featFile.batchSize,rate);
			Real regularizer = reg.get(0);
			//Real regularizer = reg.product(rate);
			grad.sumEquals(regularizer);
			this.reg = new Real(0.0);
		}
		Real change = grad.product(rate);

		Real old = w;
		if(featFile.r instanceof L1 && w.getValue()*grad.getValue()>0 &&Math.abs(w.getValue())<Math.abs(grad.getValue())){
			w = new Real(0.0);
		}else{
			w = old.minus(change);
		}
		if(w.getValue()>700.0||w.getValue()<-700.0){
			throw new RuntimeException("\nfeat="+this+"\nNumber exception old=" + old + " new = " + w+"grad=" +getGradient());
		}

		if(featFile.runSMD){
			v=featFile.lambda*v+eta*(grad.getValue()-featFile.lambda*grad.getAccum());
			w.setAccum(getV());
		}
		lastUpdate = featFile.getCurrentTime();
		grad = new Real(0.0);
		//Profiler.endProcess("update weights");
	}
	public void setValue(Type t, int val, int ind){
		types[ind]=t;
		values[ind]=val;
	}
	
	public void setWeight(Real w){
		this.w = w;
	}
	
	public Real getWeight(){
//		if(lastUpdate<ff.getCurrentTime()){
//			updateWeight(ff);
//		}
		return w;
	}
	
	public Type getType(int ind){
		return types[ind];
	}
	
	public int getValue(int ind) {
		return values[ind];
	}
	
	public void setName(String n){
		name = n; 
	}
	
	public int size() {
		return len;
	}
	public String getName() { 
		return name;
	}
	public void setOriginalName(String n){ 
		setOrigName(n);
	}
	
	public String getOriginalName() { 
		return getOrigName();
	}
	
	public void setEta(double e){
		eta = e;
	}
	
	public double getEta() {
		return eta;
	}
	
	public void setV(double v){
		this.v = v;
	}
	
	public double getV() {
		return v;
	}
	
	public void setGradient(Real g){
		grad = g;
	}
	public Real getGradient() {
		return grad;
	}
	public void accumGradient(Real g){ 
		grad=grad.sum(g); 
	}
	public void accumReg(Real g){ 
		reg=reg.sum(g); 
	}
	
	public Real getReg() {
		return reg; 
	}
	
	public void setReg(Real g){
		reg = g;
	}

	public String print(){
		String result = getOrigName() + "(";
		boolean first = true;

		for (int i = 0; i<len; i++) {
			if(!first)
				result+= ",";
			result+= types[i].getName();
			first = first && false;
		}
		result+= ")";
		return result;
	}
	
	public String toString( ){
		String result = getOrigName() + "(";
		boolean first = true;

		for (int i = 0; i<len; i++) {
			if(!first)
				result+= ",";
			result+= types[i].getName();
			first = first && false;
		}
		result += "):= [";
		first = true;
		for (int i = 0; i<len; i++) {
			if(!first)
				result+= ",";
			result+= types[i].getValName(values[i]);
			first = first && false;
		}
		result+= "]";
		return result;
	}

	public void setOrigName(String orig_name) {
		this.orig_name = orig_name;
	}

	public String getOrigName() {
		return orig_name;
	}

	public void setFeatFile(FeatureFile featFile) {
		ff = featFile;	
	}

}
