package optimization;

import driver.Learner;
import driver.Tester;

public class StochasticMetaDescent extends StochasticGradientDescent {
	public StochasticMetaDescent(Learner l, Tester t) {
		super(l, t);
		runSMD = true;
	}

}
