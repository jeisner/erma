package driver;

import inference.BeliefPropagation;
import inference.BeliefPropagation.UpdateType;

import java.util.ArrayList;
import java.util.HashMap;

import utils.Real;
import utils.Utils;

import data.DataSample;
import data.FactorGraph;
import data.FeatureFile;
import data.Probability;
import data.RV;
import data.SpeedFeatureFile;
import dataParser.DataParser;
import decoder.*;
import evaluation.FilteringLoss;
import evaluation.L1;
import evaluation.LogL;
import evaluation.LossFunction;
import featParser.FeatureFileParser;


public class BackPropTester {
	public static double beta = 5;
	public static double delta = 1e-5;
	public static void main(String[] args){
		String featureTemplate = args[0];
		String dataFile = args[1];
		LossFunction loss = new FilteringLoss();
		if(args.length>2){
			loss = Utils.getLossFunction(args[2]);
		}
		System.out.println("Reading template "+featureTemplate);
		FeatureFileParser fp;
		FeatureFile ff,origff;
		HashMap<String, Object> properties = new HashMap<String, Object>();
		properties.put("verbose", 1);
		properties.put("maxiter", 100);
		properties.put("update",UpdateType.SEQFIX);
		properties.put("tol", 1e-4);
		properties.put("recordMessages", true);
		try{
			fp = FeatureFileParser.createParser(featureTemplate);
			origff=fp.parseFile();
		}catch(Exception e){
			throw new RuntimeException(e);
		}
		if(true){
			SpeedFeatureFile sf = new SpeedFeatureFile(origff,featureTemplate,false);
			sf.setSparsityAlpha(.1);
			System.out.println("Running cost-sensitive with alpha="+Utils.df.format(.1));
			ff=sf.gateFeatureFile;
		}
		System.out.println("Reading data "+dataFile);
		DataParser dp;
		ArrayList<DataSample> examples;
		try{
			dp = DataParser.createParser(dataFile, origff); 
			examples = dp.parseFile(); 
		}catch(Exception e){ 
			throw new RuntimeException(e); 
		}
		System.out.println("Read "+examples.size()+" examples.");
		//for(DataSample ex:examples){
		DataSample ex = examples.get(0);
		FactorGraph fg = ff.toFactorGraph(ex);
		//System.out.println(fg.toString());
		BeliefPropagation bp = new BeliefPropagation(fg, properties);
		bp.run();

		System.out.println("Variable marginals:");
		for(int i = 0; i < fg.numVariables(); i++ ) // iterate over all variables in fg
			System.out.println(fg.getVariable(i)+": "+bp.beliefV(i)); // display the belief of bp for that variable
//		System.out.println("Factor marginals:");
//		for(int i = 0; i < fg.numFactors(); i++ ) // iterate over all variables in fg
//			System.out.println(fg.getFactor(i)+": "+bp.beliefF(i)); // display the belief of bp for that variable

		//Decoder dec = new Indentity();
		Decoder dec = loss.getDecoder();
		dec.softDecode(bp,beta);
		System.out.println("Decode:");
		for(RV v:fg.getVariables()) // iterate over all variables in fg
			System.out.println(v+": "+v.getDecode()+" val="+v.getValue()+(v.isOutput()?"*":"")); // display the belief of bp for that variable
		System.out.println("#############"+loss.printLoss(loss.evaluate(bp)));
		loss.revEvaluate(bp);
		if(bp.isRunReverse()){
			System.out.println("Decode gradients:");
//			for(RV v:fg.getVariables()) // iterate over all variables in fg
//				System.out.println(v+": "+(v.getGadient()==null?"nil":v.getGadient().toString()));
			double diff = loss.testBackProp(bp, beta, delta, true);
			System.out.println("Decoder diff="+diff);
			dec.reverseDecode(bp,beta);
			System.out.println("Belief gradients:");
			//			for(RV v:fg.getVariables()) // iterate over all variables in fg
			//				System.out.println(v+": "+(v.getGadient()==null?"nil":v.getGadient().toString()));

			diff = dec.testBackProp(bp, loss, beta, delta, true);
			System.out.println("Decoder diff="+diff);
		}
		double diff = bp.testBackPropInitialization(loss, dec, beta, delta, true);
		System.out.println("BP init diff="+diff);

		diff = bp.testBackProp(loss, dec, beta, delta, true);
//		diff = BeliefPropagation.testBackProp(ff,ex,properties,loss, dec, beta, delta, true);
		System.out.println("BP diff="+diff);
		//}
	}
}
