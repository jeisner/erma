package data;

import java.util.ArrayList;

public class Type {
	private ArrayList<String> value_map;
	private String name;
	private boolean isNil = false;
	private static Type nilType;
	public Type(){
		value_map = new ArrayList<String>();
	}
	public Type(String name){
		//System.out.println("Creating type "+name);
		this.name = name;
		value_map = new ArrayList<String>();
	}
	public static Type getNilType(){
		if(nilType==null){
			nilType = new Type("NIL");
			nilType.isNil = true;
		}
		return nilType;
	}
	public boolean addValue(String s){
		//System.out.println("Adding "+s);
		if(value_map.contains(s))
			return false;
		value_map.add(0,s);
		return true;
	}
	public String getName(){
		return name;
	}
	public String getValName(int val){
		if(isNil||val<0)
			return "$";
		return value_map.get(val);
	}
	public int getValue(String n){
		return value_map.indexOf(n);
	}
	public int numValues(){
		return value_map.size();
	}
	public String toString(){
		String result = name + ":= [";
		boolean first = true;
		for (String t : value_map) {
			if(!first)
				result+= ",";
			result+=t ;
			first = first && false;
		}
		return result+"]";
	}
}
