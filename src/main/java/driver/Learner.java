package driver;

import inference.BeliefPropagation.UpdateType;

import java.util.ArrayList;
import java.util.HashMap;

import optimization.OptimizationMethod;
import optimization.StochasticGradientDescent;
import optimization.StochasticMetaDescent;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import regularizer.L1;
import regularizer.Regularizer;
import regularizer.Zero;
//import utils.Profiler;
import utils.Real;
import utils.Utils;
import data.DataSample;
import data.FeatureFile;
import data.ParameterStruct;
import data.SpeedFeatureFile;
import dataParser.DataParser;
import evaluation.LossFunction;
import featParser.FeatureFileParser;

public class Learner {
	public static int seed = 1;
	public boolean use_micro_ave = false;
	public int num_restarts = 1;
	public double beta = 0.0;
	public boolean runSMD = false;
	public boolean softmax = false;
	public boolean anneal_max = false;
	public boolean bp_guidance = false;
	public boolean cost_sensitive = false;
	public double cost_alpha = 0;
	public double r;
	public double start = 0.0;
	public double end = 0.0;
	public double mu;
	public int batch_size = 5;
	public double lambda;
	public double reg_beta = 0.0;
	public Regularizer regFunction = new Zero();
	public int verbose=0;
	public int maxiter=100;
	public UpdateType update=UpdateType.SEQFIX;
	public double tol=1e-6;
	public boolean rndmz_ff=false;
	public boolean out_iters=false;
	public boolean test_training = true;
	
	//	ParameterStruct train( ParameterStruct current_fg, ArrayList<DataSample> trainData, String out_fg, int learn_iters, Real learn_rate, boolean rndmz_fg, boolean out_iters, LossFunction lfn ) {
	//		FactorGraph best_fg;
	//		feature_file best_ff;
	//		ParameterStruct best_params=null;
	//		double best_score = Double.POSITIVE_INFINITY;
	//		int best_restart = 0, best_iter=0;
	//		double totalWeight=0.0;
	//		if(use_micro_ave){
	//			current_fg.initializeSMDstructures(learn_rate);
	//			for(int i=0; i<trainData.size(); i++){
	//				DataSample samp = trainData.get(i);
	//				//FactorGraph* bp_fg = current_fg.toFactorGraph(samp);
	//				//if(use_features)
	//				//	delete bp_fg;
	//				for(int j=0; j<samp.size(); j++){
	//					if(samp.isOutput(j)){
	//						//Needed to get the weights right
	//						totalWeight+=samp.getWeight(j);
	//					}
	//				}
	//			}
	//			System.out.println("total w="+totalWeight);
	//			totalWeight = trainData.size()/totalWeight;
	//		}
	//		System.out.println("Total of "+current_fg.getParams().size()+" features.");
	//		for(int restart=1; restart<=num_restarts; restart++){
	//			try{
	//				if(rndmz_fg){
	//					current_fg.initializeRandom();
	//				}
	//
	//				//Create a GBP object
	//				// Store the constants in a PropertySet object
	//				HashMap<String,Object> properties;
	//				properties.Set("verbose",verb);  // Verbosity (amount of output generated)
	//				properties.Set("tol",tol);          // Tolerance for convergence
	//				properties.Set("maxiter",maxiter);  // Maximum number of iterations
	//				properties.Set("damping",damping);  // Amount of damping applied
	//				properties.Set("updates",bp_updates);
	////				if (anneal_max){
	////					GBP::Properties::InfType inference_type = GBP::Properties::InfType::SMAXPROD;
	////					opts.Set("inference", inference_type);
	////				}else{
	////					GBP::Properties::InfType inference_type = GBP::Properties::InfType::SUMPROD;
	////					opts.Set("inference", inference_type);
	////				}
	//
	//		return best_params;
	//
	//	}

	public ParameterStruct train(String featureTemplate, String dataFile, String out_ff, int learn_iters, Real learn_rate, LossFunction lfn, HashMap<String, String> lConfig ) {
		Utils.setSeed(seed);
		if(beta>1.001){
			start = beta;
			end = beta;
		}
		//The datastructures -- the current parameters and the data
		FeatureFileParser fp;
		FeatureFile ff;
		
		//Profiler.startProcess("reading data");
		System.out.println("Reading features from "+featureTemplate);
		try{
			fp = FeatureFileParser.createParser(featureTemplate);
			ff=fp.parseFile();
		}catch(Exception e){
			throw new RuntimeException(e);
		}
		//System.out.println("cost-sensitive="+cost_sensitive);
		if(cost_sensitive){
			SpeedFeatureFile sf = new SpeedFeatureFile(ff,lConfig.get("features"),false);
			sf.setSparsityAlpha(cost_alpha);
			System.out.println("Running cost-sensitive with alpha="+Utils.df.format(cost_alpha));
			ff=sf;
		}

		System.out.println("Read total of "+ff.getFeatures().size()+" features.");
		System.out.println("Reading data from "+dataFile);
		DataParser dp;
		ArrayList<DataSample> examples;
		try{
			dp = DataParser.createParser(dataFile, ff); 
			examples = dp.parseFile(); 
		}catch(Exception e){ 
			throw new RuntimeException(e); 
		}		
		//Profiler.endProcess("reading data");
		return train(ff,examples, out_ff, learn_iters, learn_rate, lfn);
	}
	public ParameterStruct train(FeatureFile ff, ArrayList<DataSample> examples, String out_ff, int learn_iters, Real learn_rate, LossFunction lfn ) {
		Tester test = new Tester(this);
		
		//test.use_micro_ave = use_micro_ave;

		//learn_rate/=(double)batch_size;
		System.out.println("Running "+(runSMD?"SMD":"SGD")+". ");
		System.out.print("Training for " + lfn +". Parameters: eta_0="+learn_rate+" lambda="+lambda);
		System.out.println(" mu="+mu+" batch_size="+batch_size+ " regularizer "
				+ (regFunction instanceof Zero?"none":(regFunction instanceof L1 ?"L1":"L2"))+ " reg_beta="+ reg_beta+" beta="+ beta);
		if(cost_sensitive){
			System.out.println("Cost sensitive training with r="+r);
		}
		if(bp_guidance){
			System.out.println("Guided BP for training.");
		}
		if(anneal_max){
			System.out.println("Annealing for max product.");
		}
		if(softmax){
			System.out.println("Using softmax training.");
		}
		System.out.println("read " + examples.size() + " train examples.");

		HashMap<String,Object> properties=new HashMap<String, Object>();
		properties.put("verbose",verbose);  // Verbosity (amount of output generated)
		properties.put("tol",tol);          // Tolerance for convergence
		properties.put("maxiter",maxiter);  // Maximum number of iterations
		properties.put("update",update);
		properties.put("recordMessages", true);

		System.out.println(properties);

		OptimizationMethod optimizer = runSMD?new StochasticMetaDescent(this, test):new StochasticGradientDescent(this,test);
		//System.out.println("test training="+test_training);
		optimizer.setTestTraining(test_training);
		ParameterStruct bestParams = optimizer.optimize(ff,examples,out_ff,learn_iters,learn_rate,rndmz_ff,out_iters,lfn,properties,regFunction,reg_beta);

		String ofilename = out_ff+"-best.ff";
		
		bestParams.print(ofilename);
		//System.out.println("beta="+beta);
		return bestParams;
	}

	public static void main(String[] args){
		//Profiler.start();
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
		Learner learner = new Learner();
		HashMap<String,String> lConfig = Utils.configureLearner(learner,cmd);
		int learn_iters = Integer.parseInt(lConfig.get("iter"));
		Real learn_rate = new Real(Double.parseDouble(lConfig.get("learn_rate")));
		LossFunction lfn = Utils.getLossFunction(lConfig); 
		//Profiler.startProcess("learning");
		learner.train(lConfig.get("features"), lConfig.get("data"), lConfig.get("out_ff"),learn_iters, learn_rate, lfn, lConfig);
		//Profiler.endProcess("learning");
		//Profiler.printProcessTimes();
	}


}
