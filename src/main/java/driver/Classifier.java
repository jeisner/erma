package driver;

import inference.BeliefPropagation;
import inference.BeliefPropagation.UpdateType;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import regularizer.L1;
import regularizer.Regularizer;

import utils.Real;
import utils.Utils;

import data.DataSample;
import data.FactorGraph;
import data.FeatureFile;
import data.ParameterStruct;
import data.Probability;
import data.RV;
import data.SpeedFeatureFile;
import dataParser.DataParser;
import decoder.Decoder;
import evaluation.LossFunction;
import featParser.FeatureFileParser;

public class Classifier extends Tester{
	private String outfilename;
	public Classifier(){
		
	}
	public Classifier(Learner l){
		use_micro_ave=l.use_micro_ave;
		bp_guidance = l.bp_guidance;
		cost_sensitive = l.cost_sensitive;
		r = l.r;
		beta = l.beta;
		reg_beta = l.reg_beta;
		verbose = l.verbose;
		maxiter = l.maxiter;
		tol = l.tol;
		use_soft_decoder=true;
	}
	public double test(ParameterStruct param, ArrayList<DataSample> testData, LossFunction lfn, HashMap<String, Object> iaProps, double beta, Real alpha) {
		Decoder decoder = lfn.getDecoder();
		//System.out.println("Using decoder "+decoder.toString());
		decoder.setThreshold(0.5);
		
		//Open the output file
		PrintWriter outp;
		try {
			outp = new PrintWriter(outfilename);
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		}

		// Store the constants in a PropertySet object
		double score = 0.0;
		long tic = System.currentTimeMillis();
		int totalIters=0;
		int active=0;
//		System.out.println(param.printWeights());
		for(int i=0; i<testData.size(); i++){
			outp.println("//example "+i);
			outp.println("example:");
			if(testData.size()<outputTics||i%(testData.size()/outputTics)==0){
				System.out.print(i + ".");
				System.out.flush();
			}
			DataSample samp = testData.get(i);
			FactorGraph fg = param.toFactorGraph(samp);

			//System.out.println(i+": fg with "+ fg.nrVars() + " vars and " + fg.nrFactors() + " factors." + endl;
			BeliefPropagation bp = new BeliefPropagation(fg, iaProps);

			Real samp_score;
			//			if(cost_sensitive){
			//				samp_score = bp.run_cost_sensitive(lfn,samp,beta,decoder,(Real).5,(Real)r);
			//				totalIters+=bp.Iterations();
			//			}else{
			bp.run();
//			System.out.println("Variable marginals:");

			if(use_soft_decoder )
				decoder.softDecode(bp, beta);
			else
				decoder.decode(bp);
			for(int v = 0; v < bp.getFactorGraph().numVariables(); v++ ){
				RV var = bp.getFactorGraph().getVariable(v);
				outp.println(var+"="+var.getType().getValName(var.getDecode().argmax()) + " " + var.getDecode().getProb()); 
			}

			samp_score = lfn.evaluate(bp);
			//System.out.println("samp score="+samp_score+endl;
			totalIters+=bp.numIterations();
			if(cost_sensitive)
				active+=bp.getFactorGraph().numActiveFactors();
			//			}
			if(output_doc_scores){
				System.out.println("Document "+i+": "+samp_score);
			}
			score+=samp_score.getValue();
		}
		//		}
		System.out.println("{test in " + Utils.formatTime(System.currentTimeMillis() - tic) +" sec.} (totalBPiters "+totalIters+")");
		if(cost_sensitive)
			System.out.println("Total of "+active+" active edges");
		
		System.out.flush();
		double totalWeight=0.0;
		if(use_micro_ave){
			for(int i=0; i<testData.size(); i++){
				DataSample samp = testData.get(i);
				for(RV vj:samp.getVariables()){
					if(vj.isOutput()){
						//Needed to get the weights right
						totalWeight+=vj.getWeight();
					}
				}
			}
		}
		if(!use_micro_ave)
			score = score/(double)testData.size();
		else
			score = score/totalWeight;
		//Read in the data file
		outp.flush();
		return score;
	}

	
	public static void main(String[] args){
		// create Options object
		Options options = Utils.createOptions();
		CommandLineParser parser = new GnuParser();
		CommandLine cmd = null;
		try {
			// parse the command line arguments
			cmd = parser.parse( options, args );
		}
		catch( ParseException exp ) {
			// oops, something went wrong
			System.err.println( "Option parsing failed.  Reason: " + exp.getMessage() );
		}
		Classifier tester = new Classifier();
		HashMap<String,String> lConfig = Utils.configureTester(tester,cmd);

		tester.outfilename = lConfig.get("pred_fname");
		LossFunction lfn = Utils.getLossFunction(lConfig); 
		tester.test(lConfig.get("features"), lConfig.get("data"), lfn, lConfig);


	}
}
