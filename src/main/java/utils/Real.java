package utils;

public class Real{
	private double value, accum;
	public double getValue() {
		return value;
	}
	public void setValue(double value) {
		this.value = value;
	}
	public double getAccum() {
		return accum;
	}
	public void setAccum(double accum) {
		this.accum = accum;
	}
	public Real(double v){
		value = v;
		accum = 0.0;
	}
	public Real(double v, double a){
		value = v;
		accum = a;
	}
	public Real(Real val) {
		this.value = val.value;
		this.accum = val.accum;
	}
	public Real product(Real x){
		return new Real(value*x.value,value * x.accum + x.value * accum);
	}
	public Real sum(Real rhs){
		return new Real(value + rhs.value, accum + rhs.accum);
	}
	public Real sumEquals(Real rhs){
		this.value= value + rhs.value;
		this.accum = accum + rhs.accum;
		return this;
	}
	public Real product(double x){
		return new Real(value*x, x * accum);
	}
	public Real sum(double rhs){
		return new Real(value + rhs, accum);
	}
	public Real divide(Real rhs){
		double inv = 1.0 / rhs.value;
		double val = value * inv;

		return new Real(val, inv * (accum - val * rhs.accum));
	}
	public static Real log(Real arg){
		return new Real(Utils.log0(arg.getValue()), arg.getAccum() / arg.getValue());
	}
	public Real minus(Real rhs){
        return new Real(value - rhs.value, accum - rhs.accum);
	}
	public static Real unMinus(Real rhs){
        return new Real( -rhs.value, - rhs.accum);
	}	
	public boolean equals(Real rhs){
		return (value == rhs.value);
	}
	// EQ
	public boolean equals(double rhs){
		return (value == rhs);
	}
	// NEQ
	public boolean	neq(Real rhs){
		return (value != rhs.value);
	}
	// NEQ
	public boolean	neq(double rhs){
		return (value != rhs);
	}
	// LT
	public boolean lt(Real rhs){
		return (value < rhs.value);
	}
	// GT
	public boolean gt(Real rhs){
		return (value > rhs.value);
	}
	// LT
	public boolean lt(double rhs){
		return (value < rhs);
	}
	// GT
	public boolean gt(double rhs){
		return (value > rhs);
	}
	// LTE
	public boolean lte(Real rhs){
		return (value <= rhs.value);
	}
	// GTE
	public boolean gte(Real rhs){
		return (value >= rhs.value);
	}
	// LTE
	public boolean lte(double rhs){
		return (value <= rhs);
	}
	// GTE
	public boolean gte(double rhs){
		return (value >= rhs);
	}
	public String toString(){
		return Double.toString(value);
	}
	public Real pow(double e) {
		return pow(new Real(e));
	}
	public Real pow(Real rhs) {
		return new Real(Math.pow(value,rhs.value),
				(rhs.value * Math.pow(value,(rhs.value-1.0)))* accum + value==0.0?0.0:Math.log(value)*(Math.pow(value,rhs.value)) * rhs.accum);
	}
	public static Real exp(Real arg){
		double value = Math.exp(arg.getValue());

		return new Real(value, value * arg.getAccum());
	}
//	public Real log(Real arg){
//		return new Real(log(arg.getValue()), arg.getAcc() / arg.getValue());
//	}
	public Real divide(double div) {
		return divide(new Real(div));
	}
	public void sumEquals(double rhs) {
		sumEquals(new Real(rhs));		
	}
	public Real minus(double delta) {
		return minus(new Real(delta));
	}
}
