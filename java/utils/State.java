package utils;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;

import javax.management.RuntimeErrorException;

import data.RV;
import data.VariableSet;

public class State {
	/// Current state (represented linearly)
	private int state;

	/// Current state (represented as a map)
	private TreeMap<RV, Integer> states;

	/// Default constructor
	public void State(){
		state=0;
		states = new TreeMap<RV, Integer>();
	}

	/// Construct from VarSet \a vs and corresponding linear state \a linearState
	public State( Collection<RV> vs, int linearState ) {
		state = linearState;
		states = new TreeMap<RV, Integer>();
		if( linearState == 0 )
			for( RV v :vs )
				states.put(v,0);
		else {
			for( RV v : vs ) {
				states.put(v, linearState % v.numValues());
				linearState /= v.numValues();
			}
			if( linearState != 0 )
				throw new RuntimeException("linearstate != 0");
		}
	}

	/// Construct from a std::map<Var, size_t>
	public State( TreeMap<RV, Integer> s ) {
		VariableSet vars = new VariableSet();
		vars.addAll(s.keySet());
		state = 0;
		states = s;
        //int vs_state = 0;
        int prod = 1;
        for( RV v : vars) {
            if(states.containsKey(v))
                state += states.get(v) * prod;
            prod *= v.numValues();
        }
	}

	/// Return current linear state
	public int getState() {
		if(!validate())
			throw new RuntimeException("Invalid state");
		return state;
	}

	/// Return current state represented as a map
	public Map<RV,Integer> get() { return states; }

	/// Return current state of variable \a v, or 0 if \a v is not in \c *this
	public int getState(RV v ) {
		if(!validate())
			throw new RuntimeException("Invalid state");
		return states.get( v );
	}

	/// Return linear state of variables in \a vs, assuming that variables that are not in \c *this are in state 0
	public int getState( VariableSet vs ){
		if(!validate())
			throw new RuntimeException("Invalid state");
		int vs_state = 0;
		int prod = 1;
		for( RV v : vs ) {
			Integer entry = states.get( v );
			if( entry != null )
				vs_state += entry * prod;
			prod *= v.numValues();
		}
		return vs_state;
	}

	/// Increments the current state (prefix)
	public void increment( ) {
		//TODO: this method may have to be checked
		if( validate() ) {
			state++;

			for( RV v: states.keySet() ) {
				states.put(v,states.get(v)+1);
				if( states.get(v) < v.numValues() )
					break;
				states.put(v, 0);

			}
		}
	}


	/// Returns \c true if the current state is valid
	public boolean validate(){
		return( state >= 0 );
	}

	/// Resets the current state (to the joint state represented by linear state 0)
	public void reset() {
		state = 0;
		for( RV v : states.keySet() )
			states.put(v, 0);
	}

}
