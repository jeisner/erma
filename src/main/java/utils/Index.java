package utils;


import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import data.RV;
import data.VariableSet;

public class Index {
	/// The current linear index corresponding to the state of indexVars
	private int index;
	/// For each variable in forVars, the amount of change in _index
	private ArrayList<Integer> sum;

	/// For each variable in forVars, the current state
	private ArrayList<Integer> state;

	/// For each variable in forVars, its number of possible values
	private ArrayList<Integer> ranges;

	/// Default constructor
	public Index() {
		index = -1;
	}

	/// Construct Index for object from \a indexVars and \a forVars
	private void init(VariableSet indexVars, VariableSet forVars ) {
		state = new ArrayList<Integer>(forVars.size());
		for (int i = 0; i < forVars.size(); i++) {
			state.add(0);			
		}
		int s = 1;

		ranges =  new ArrayList<Integer>(forVars.size());
		sum = new ArrayList<Integer>(forVars.size());
	    Iterator<RV> forIter = forVars.iterator();
	    RV vj = null;
	    for( RV vi: indexVars ) {
	    	do{
	    		vj=forIter.next();
	    		ranges.add( vj.numValues() );
	    		sum.add( (vi.equals(vj)) ? s : 0 );
	    	}while(forIter.hasNext()&&vj.compareTo(vi)<0);
	    	s *= vi.numValues();
	    }

		while(forIter.hasNext()) {
			ranges.add( forIter.next().numValues() );
			sum.add( 0 );
		}
		index = 0;
//		System.out.println("index indvars "+indexVars+" forVars "+forVars);
//		System.out.println("state "+state);
//		System.out.println("ranges "+ranges);
//		System.out.println("sum "+sum);
	}
	public Index(RV indexVar, VariableSet forVars) {
		//System.out.println("Index for "+indexVar+" Set: "+forVars);
		VariableSet indexVars = new VariableSet();
		indexVars.add(indexVar);
		init(indexVars,forVars);
	}
	
	public Index( VariableSet indexVars, VariableSet forVars ) {
		init(indexVars,forVars);
	}
	/// Resets the state
	public Index reset() {
		for (int i = 0; i < state.size(); i++) {
			state.set(i, 0);
		}
		index = 0;
		return this;
	}

	/// Returns linear index of the current state of indexVars
	public int index(){
		return index;
	}

	/// Increments the current state of \a forVars (prefix)
	public Index increment() {
		//System.out.println("State "+state);
		if( index >= 0 ) {
			int i = 0;

			while( i < state.size() ) {
				index += sum.get(i);
				state.set(i, state.get(i)+1);
				if( state.get(i) < ranges.get(i) )
					break;				
				index -= sum.get(i) * ranges.get(i);
				state.set(i, 0);
				i++;
			}

			if( i == state.size() )
				index = -1;
		}
		return this;
	}

	/// Returns \c true if the current state is valid
	public boolean isValid() {
		return index >= 0;
	}
	
	public int current(){
		return index;
	}
}
