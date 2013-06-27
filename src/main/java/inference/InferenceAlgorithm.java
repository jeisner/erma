package inference;

import java.util.HashMap;

import utils.Real;

import data.FactorGraph;
import data.Probability;
/// A base class for all inference algorithms to extend

public abstract class InferenceAlgorithm {
	protected FactorGraph fg;

	public abstract double run();
	
	public abstract Probability beliefV( int i );

	public abstract Probability beliefF( int i );
	
	public FactorGraph getFactorGraph(){
		return fg;
	}

	public abstract HashMap<String, Object> getProperties();
	public abstract void setProperties(HashMap<String, Object> props);

	public abstract Real logZ();
}
