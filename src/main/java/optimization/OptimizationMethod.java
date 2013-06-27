package optimization;


import java.util.ArrayList;
import java.util.HashMap;

import regularizer.Regularizer;

import utils.Real;
import data.DataSample;
import data.ParameterStruct;
import evaluation.LossFunction;

public abstract class OptimizationMethod {

	public abstract ParameterStruct optimize(ParameterStruct currentFg,
			ArrayList<DataSample> trainData, String outputFilename,
			int learnIters, Real learnRate, boolean rndmzFg, boolean outIters,
			LossFunction lfn, HashMap<String, Object> infProperties, 
			Regularizer regFunction, double regBeta);
	public abstract void setTestTraining(boolean testTraining);
}
