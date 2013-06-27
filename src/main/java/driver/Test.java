package driver;

import inference.BeliefPropagation;
import inference.BeliefPropagation.UpdateType;

import java.util.ArrayList;
import java.util.HashMap;

import utils.Utils;

import data.DataSample;
import data.FactorGraph;
import data.FeatureFile;
import data.RV;
import dataParser.DataParser;
import decoder.Decoder;
import evaluation.FactoredLogL;
import evaluation.L1;
import evaluation.LossFunction;
import featParser.FeatureFileParser;


public class Test {
	public static double beta = 4.0;
	public static void main(String[] args){
		String featureTemplate = args[0];
		String dataFile = args[1];
		System.out.println("Reading template "+featureTemplate);
		FeatureFileParser fp;
		FeatureFile ff;
		HashMap<String, Object> properties = new HashMap<String, Object>();
		properties.put("verbose", 5);
		properties.put("maxiter", 100);
		properties.put("update",UpdateType.SEQMAX);
		properties.put("tol", 1e-4);

		try{
			fp = FeatureFileParser.createParser(featureTemplate);
			ff=fp.parseFile();
		}catch(Exception e){
			throw new RuntimeException(e);
		}
		System.out.println("Reading data "+dataFile);
		DataParser dp;
		ArrayList<DataSample> examples;
		try{
			dp = DataParser.createParser(dataFile, ff); 
			examples = dp.parseFile(); 
		}catch(Exception e){ 
			throw new RuntimeException(e); 
		}
		System.out.println("Read "+examples.size()+" examples.");
		for(DataSample ex:examples){
			FactorGraph fg = ff.toFactorGraph(ex);
			System.out.println(ff+"\n"+fg.toString());
			BeliefPropagation bp = new BeliefPropagation(fg, properties);
			bp.run();

			System.out.println("Variable marginals:");
			for(int i = 0; i < fg.numVariables(); i++ ) // iterate over all variables in fg
				System.out.println(fg.getVariable(i)+": "+bp.beliefV(i)); // display the belief of bp for that variable
			LossFunction loss = new FactoredLogL();
			Decoder dec = loss.getDecoder();
			dec.softDecode(bp,beta);
			System.out.println("Decode:");
			for(RV v:fg.getVariables()) // iterate over all variables in fg
				System.out.println(v+": "+v.getDecode()+" val="+v.getValue()+(v.isOutput()?"*":"")); // display the belief of bp for that variable
			System.out.println(loss.printLoss(loss.evaluate(bp)));
			loss.revEvaluate(bp);
			System.out.println("Gradients:");
			for(RV v:fg.getVariables()) // iterate over all variables in fg
				System.out.println(v+": "+(v.getGadient()==null?"nil":v.getGadient().toString()));
			dec.reverseDecode(bp,beta);
			System.out.println("Gradients:");
			for(RV v:fg.getVariables()) // iterate over all variables in fg
				System.out.println(v+": "+(v.getGadient()==null?"nil":v.getGadient().toString()));

		}
	}
}
