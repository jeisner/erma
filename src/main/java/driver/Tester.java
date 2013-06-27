package driver;

import inference.BeliefPropagation;
import inference.BeliefPropagation.UpdateType;

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

public class Tester {
	public static final int outputTics = 10;
	public boolean exact_inf_eval;
	public boolean output_doc_scores;
	public boolean use_micro_ave;
	public boolean bp_guidance;
	public boolean cost_sensitive;
	public double r;
	public double beta;
	public double reg_beta;
	public int verbose;
	public int maxiter;
	public double tol;
	public boolean use_soft_decoder = false;

	public Tester(){
		
	}
	public Tester(Learner l){
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
		// Read FactorGraph from the file specified by the first command line argument
		Decoder decoder = lfn.getDecoder();
		//System.out.println("Using decoder "+decoder.toString());
		decoder.setThreshold(0.5);

		// Store the constants in a PropertySet object
		double score = 0.0;
		long tic = System.currentTimeMillis();
		int totalIters=0;
		int active=0;
//		System.out.println(param.printWeights());
		for(int i=0; i<testData.size(); i++){
			//System.out.println(" test" +i + endl;

			if(testData.size()<outputTics||i%(testData.size()/outputTics)==0){
				System.out.print(i + ".");
				System.out.flush();
			}
			DataSample samp = testData.get(i);
			FactorGraph fg = param.toFactorGraph(samp);

			//Show about 10 random gates
			if(param instanceof SpeedFeatureFile && i==0 && ((SpeedFeatureFile)param).gbp!=null){
				FactorGraph gfg = ((SpeedFeatureFile)param).gbp.getFactorGraph();
				System.out.println("\nt="+((SpeedFeatureFile)param).getT());
				System.out.println("Gate values:");
				ArrayList<RV> gfgVars = gfg.getVariables();
				for(RV v:gfgVars){
					if(Utils.rand.nextDouble()<10.0/(double)gfgVars.size())
					System.out.println("\t"+v.getName()+" "+Utils.df.format(v.getDecode().getValue(v.getType().getValue("on")).getValue()));
				}
			}
			//System.out.println(i+": fg with "+ fg.nrVars() + " vars and " + fg.nrFactors() + " factors." + endl;
			BeliefPropagation bp = new BeliefPropagation(fg, iaProps);

			Real samp_score;
			//			if(cost_sensitive){
			//				samp_score = bp.run_cost_sensitive(lfn,samp,beta,decoder,(Real).5,(Real)r);
			//				totalIters+=bp.Iterations();
			//			}else{
			bp.run();
//			System.out.println("Variable marginals:");
//			for(int i1 = 0; i1 < bp.getFactorGraph().numVariables(); i1++ ) // iterate over all variables in fg
//				System.out.println(bp.getFactorGraph().getVariable(i1)+": "+bp.beliefV(i1)); // display the belief of bp for that variable

			if(use_soft_decoder )
				decoder.softDecode(bp, beta);
			else
				decoder.decode(bp);
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
		return score;
	}
	int test(String featureTemplate, String dataFile, LossFunction lfn,HashMap<String, String> lConfig){
		System.out.println("Reading template "+featureTemplate);
		FeatureFileParser fp;
		FeatureFile ff;

		try{
			fp = FeatureFileParser.createParser(featureTemplate);
			ff=fp.parseFile();
		}catch(Exception e){
			throw new RuntimeException(e);
		}
		if(cost_sensitive){
			SpeedFeatureFile sf = new SpeedFeatureFile(ff,lConfig.get("features"),true);
			ff=sf;
		}
		DataParser dp;
		ArrayList<DataSample> examples;
		try{
			dp = DataParser.createParser(dataFile, ff); 
			examples = dp.parseFile(); 
		}catch(Exception e){ 
			throw new RuntimeException(e); 
		}		
		HashMap<String, Object> properties = new HashMap<String, Object>();
		properties.put("verbose", verbose);
		properties.put("maxiter", maxiter);
		properties.put("update",UpdateType.SEQFIX);
		properties.put("tol", tol);
		properties.put("recordMessages", false);
		
		System.out.println("Reading data "+dataFile);

		System.out.println("Read "+examples.size()+" examples.");

		double score = test(ff, examples, lfn, properties, beta, new Real(0.0));
		System.out.println(lfn.printLoss(score));

		return 0;
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
		Tester tester = new Tester();
		HashMap<String,String> lConfig = Utils.configureTester(tester,cmd);

		int learn_iters = Integer.parseInt(lConfig.get("iter"));
		Real learn_rate = new Real(Double.parseDouble(lConfig.get("learn_rate")));
		LossFunction lfn = Utils.getLossFunction(lConfig); 
		tester.test(lConfig.get("features"), lConfig.get("data"), lfn, lConfig);


	}
}
