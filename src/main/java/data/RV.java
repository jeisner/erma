package data;

import java.util.HashSet;

public class RV extends GraphNode{
	private static int nextId = 0;
	private int numValues;
	private int value;
	private int number; //The variable's number in the factor graph
	private Type type;
	private VisibilityType vtype = VisibilityType.OUTPUT; 
	private double weight;
	private String name;
	private HashSet<Factor> factors;
	private Probability decode;
	private Probability gradient;

	public enum VisibilityType{ INPUT,OUTPUT,HIDDEN }
	
	public RV(int id, int numValues){
		this.id = id;
		factors = new HashSet<Factor>();
		this.numValues = numValues;
	}
	
	public RV(int id, String name, Type t, int numValues) { 
		this.id = id;
		this.name = name; 
		this.type = t; 
		value=-1; 
		this.numValues = numValues;
		weight=1.0;
	}
	
	public RV(int id, String name, Type t) { 
		this.id = id;
		this.name = name; 
		this.type = t; 
		value=-1; 
		this.numValues = t.numValues();
		weight=1.0;
	}
	
	
	public RV(RV v) {
		super();
		this.numValues = v.numValues;
		this.value = v.value;
		this.number = v.number;
		this.type = v.type;
		this.vtype = v.vtype;
		this.weight = v.weight;
		this.name = v.name;
	}

	public int numValues(){
		return numValues;
	}
	
	public void addFactor(Factor f){
		factors.add(f);
	}
	
	public String toString(){
		return name+(isInput()?"in":isHidden()?"h":"o");
	}
	
	public void setValue(int value) {
		this.value = value;
	}
	
	public int getValue() {
		return value;
	}
	
	public void setName(String name) {
		this.name = name;
	}
	
	public String getName() {
		return name;
	}
	public double getWeight() {
		return weight;
	}
	public void setWeight(double weight) {
		this.weight = weight;
	}

	public void setVisType(VisibilityType vt){
		this.vtype = vt;
	}
	public boolean isInput(){
		return vtype==VisibilityType.INPUT;
	}
	public boolean isOutput(){
		return vtype==VisibilityType.OUTPUT;
	}
	public boolean isHidden(){
		return vtype==VisibilityType.HIDDEN;
	}

	public Type getType() {
		return type;
	}

	public void setType(Type type) {
		this.type = type;
	}
	public String getValueName(){
		return type.getValName(value);
	}

	public int getNumber() {
		return number;
	}

	public void setNumber(int number) {
		this.number = number;
	}

	public void setDecode(Probability decode) {
		this.decode = decode;
	}

	public Probability getDecode() {
		return decode;
	}

	public void setGradient(Probability grad) {
		this.gradient = grad;
		
	}

	public Probability getGadient() {
		return gradient;
	}
}
