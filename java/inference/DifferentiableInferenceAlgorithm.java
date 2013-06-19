package inference;

/// A differentiable or "reversable" inference algorithm
public abstract class DifferentiableInferenceAlgorithm extends
		InferenceAlgorithm {

	public abstract void reverse();

}
