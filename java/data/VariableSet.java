package data;

import java.util.TreeSet;

public class VariableSet extends TreeSet<RV> {
	public VariableSet(VariableSet vs){
		super();
		addAll(vs);
	}
	public VariableSet() {
		super();
	}
	/**
	 * 
	 */
	private static final long serialVersionUID = -8004071724025324223L;

}
