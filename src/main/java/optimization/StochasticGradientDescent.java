package optimization;

import inference.BeliefPropagation;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import regularizer.Regularizer;
import utils.Profiler;
import utils.Real;
import utils.Utils;
import data.DataSample;
import data.FactorGraph;
import data.ParameterStruct;
import data.RV;
import data.SpeedFeatureFile;
import decoder.Decoder;
import driver.Learner;
import driver.Tester;
import evaluation.LossFunction;

public class StochasticGradientDescent extends OptimizationMethod {
	private boolean use_micro_ave;
	private int num_restarts=1;
	private boolean bp_guidance;
	private double guidance_param;
	private boolean softmax;
	private boolean cost_sensitive;
	private int batch_size;
	private double beta;
	private boolean anneal_max;
	private double start;
	private double end;
	private int num_buckets;
	private boolean rate_restarts;
	private double epsilon;
	protected boolean runSMD;
	private double lambda;
	private double mu;
	private Tester tester;
	private boolean test_training = true;


	private static int outputTics=20;
	
	public StochasticGradientDescent(Learner l, Tester t){
		tester = t;
		use_micro_ave = l.use_micro_ave;
		num_restarts = l.num_restarts;
		bp_guidance = l.bp_guidance;
		guidance_param = 0.0;
		softmax = l.softmax;
		cost_sensitive = l.cost_sensitive;
		batch_size = l.batch_size;
		beta = l.beta;
		anneal_max = l.anneal_max;
		start = l.start;
		end = l.end;
		runSMD = l.runSMD;
		lambda = l.lambda;
		mu = l.mu;
	}

	public double computeLearnRate(LossFunction lfn, Real learn_rate, int iter, int learn_iters){
		double rate;
		Real step = new Real((end-start)/num_buckets);
		if(lfn.isAnnealed()){
			int bucket = (iter-1)*num_buckets/learn_iters;
			beta = step.product(bucket).sum(start).getValue();
			rate = learn_rate.getValue()/Math.sqrt(iter);//(1+bucket)*learn_rate;
		}else{
			rate = learn_rate.getValue()/Math.sqrt(iter);//learn_rate/(iter + 1.25)^0.602;
		}
		return rate;
	}

	@Override
	public ParameterStruct optimize(ParameterStruct currentTheta, ArrayList<DataSample> trainData, String outputFilename, int numLearnIters, Real learnRate, boolean rndmzFg, boolean outputInter, LossFunction lfn, HashMap<String,Object> infProperties, Regularizer regFunction, double regBeta) {
		Decoder dec = lfn.getDecoder();
		System.out.println("Decoder is "+dec);
		System.out.println("Regularizer is "+regFunction);
		ParameterStruct best_params=null;
		double best_score = Double.MAX_VALUE;
		int best_restart = 0, best_iter=0;
		Real totalWeight=new Real(0.0);
		if(use_micro_ave){
			currentTheta.initializeSMDstructures(learnRate);
			for(int i=0; i<trainData.size(); i++){
				DataSample samp = trainData.get(i);

				for(RV vj:samp.getVariables()){
					if(vj.isOutput()){
						//Needed to get the weights right
						totalWeight.sumEquals(vj.getWeight());
					}
				}
			}
			System.out.println("total w="+totalWeight);
			totalWeight = (new Real(trainData.size())).divide(totalWeight);
		}

		for(int restart=1; restart<=num_restarts; restart++){
			try{
				if(rndmzFg){
					currentTheta.initializeRandom();
				}


				long tic = System.currentTimeMillis();
				//Real divider = 1.0;

				//new FactorGraph(bp_fg);
				currentTheta.initializeSMDstructures(learnRate);

				double older_score = Double.MAX_VALUE;
				double old_score = Double.MAX_VALUE;
				Real alpha = new Real(1.0);
				double step_size = 0.5;
				int iter=1;
				boolean done = false;
				for(; iter<=numLearnIters && !done; iter++){

					double rate;
					//for(int i=0;i<eta.size()&&i<20;i++){
					//	cerr + eta[i] );
					//}
					//Real alpha = (Real)1.0-((Real)iter/(Real)(learn_iters-10));
					alpha = alpha.getValue()<0.0?new Real(0.0):alpha;

					//Compute learn rate for the iteration
					rate = computeLearnRate(lfn, learnRate, iter, numLearnIters);

					System.out.print("-------------- Iteration "+iter+ "(mu="+Utils.df.format(rate)+")(t="+Utils.df.format(beta)+")");
					if (bp_guidance) 
						System.out.print("(a=" + Utils.df.format(guidance_param) + ")");
					System.out.println(" ---------------");
					int totalIters = 0;
					double score = 0.0;
					Collections.shuffle(trainData);
					double loss = 0;
					if(cost_sensitive){
						((SpeedFeatureFile)currentTheta).setT(iter>3?(iter>7?iter>10?16.0:8.0:4.0):2.0);
						System.out.println("t="+((SpeedFeatureFile)currentTheta).getT());
					}
					for(int sample_num=0; sample_num<trainData.size() && !done; sample_num++){
						if(trainData.size()<outputTics || sample_num%(trainData.size()/outputTics)==0){
							System.out.print(sample_num + ".");
							System.out.flush();
						}
						//System.out.println("to fg "+ current_fg + " best "+ best_params+ endl;
						DataSample samp = trainData.get(sample_num);
						Profiler.startProcess("unrolling factor graph");
						FactorGraph bp_fg = currentTheta.toFactorGraph(samp);
						Profiler.endProcess("unrolling factor graph");
						
						Profiler.startProcess("inference");
						BeliefPropagation bp = new BeliefPropagation(bp_fg, infProperties);
						bp.setRunReverse(true);

						//System.out.println(bp.printProperties());
						//Run inference
						bp.run();
						Profiler.endProcess("inference");
						//decode
//						System.out.println("Variable marginals:");
//						for(int i = 0; i < bp.getFactorGraph().numVariables(); i++ ) // iterate over all variables in fg
//							System.out.println(bp.getFactorGraph().getVariable(i)+": "+bp.beliefV(i)); // display the belief of bp for that variable

						//dec = new Max();
						//System.out.println("beta="+beta);
						dec.softDecode(bp,beta);
						//Compute loss
						loss+=lfn.evaluate(bp).getValue();
//						for(RV v:bp.getFactorGraph().getVariables()) // iterate over all variables in fg
//							System.out.println(v+": "+v.getDecode()+" val="+v.getValue()+(v.isOutput()?"*":"")); // display the belief of bp for that variable
						//System.out.println("#############"+lfn.printLoss(lfn.evaluate(bp)));
						//Begin the reverse pass
						lfn.revEvaluate(bp);
						if(bp.isRunReverse()){
							dec.reverseDecode(bp,beta);
						}
						bp.reverse();

						totalIters+=bp.numIterations();
						//Add up the gradients for the batch
						Profiler.startProcess("update weights");
						currentTheta.accumulateGradient(bp,regFunction,regBeta);   
						//If the batch is complete, update the parameters
						if((sample_num+1)%batch_size==0||sample_num==trainData.size()-1){
							currentTheta.updateWeights(rate,score,runSMD,lambda,mu,batch_size,regFunction, regBeta);
						}
						Profiler.endProcess("update weights");
					}
					//score = score/(Real)trainData.nrSamples();
					//System.out.print(".");
					//System.out.flush();
					BeliefPropagation bp = new BeliefPropagation(currentTheta.toFactorGraph(trainData.get(0)), infProperties);
					bp.run();
//					System.out.println(currentTheta.printWeights());
//					System.out.println("Variable marginals:");
//					for(int i1 = 0; i1 < bp.getFactorGraph().numVariables(); i1++ ) // iterate over all variables in fg
//						System.out.println(bp.getFactorGraph().getVariable(i1)+": "+bp.beliefV(i1)); // display the belief of bp for that variable

					if(test_training )
						loss = tester.test(currentTheta, trainData, lfn, infProperties, beta, alpha);
					else
						loss/=trainData.size();
					double reg = currentTheta.evaluateRegularizer(regFunction,regBeta);
					score=reg+loss;
					//System.out.println("\nScore="+score);
					//double score1 = test(*test_fg,trainData,lfn,beta,alpha);
					//System.out.println("-----bp-----\n"+bp_fg.factor(0) + "\n-----test_bp-----\n"+test_fg.factor(0));
					if(!lfn.isInterpolated()||alpha.getValue()<0.01){
						if(score<best_score||cost_sensitive){
							best_score = score;
							best_params = currentTheta.copy();
							//best_fg = in_fg;
							best_restart = restart;
							best_iter = iter;
						}
					}
					System.out.println("done in " + Utils.formatTime(System.currentTimeMillis()-tic) +" seconds ("+Utils.df.format((double)totalIters/(double)trainData.size())+" ave bp iter) ["
							+lfn.printLoss(loss)+" reg="+Utils.df.format(reg)+"].");
					System.err.print("i"+ iter +".["+Utils.df.format(score)+"].");
					if(outputInter){
						String ofilename = outputFilename+"-r"+restart+"-i"+iter+".ff";
						//PrintWriter out = new PrintWriter(new FileWriter(ofilename));
						currentTheta.print(ofilename); 
					}
					tic = System.currentTimeMillis();
					if(lfn.isInterpolated()&&(old_score-score)<(epsilon*score) 
							&& (older_score-score)<(epsilon*score)){
						if(alpha.getValue()>0.0){
							alpha.sumEquals(-step_size);
							alpha = alpha.getValue()>0.0?alpha:new Real(0.0);
							//learn_rate /= (Real)10.0;
							System.out.println("*******************Reducing alpha to "+alpha+" lr="+learnRate+"******************");
							currentTheta.initializeSMDstructures(learnRate);
							older_score = Double.MAX_VALUE;
							old_score = Double.MAX_VALUE;
						}else{
							done = true;
						}
					}else{
						older_score = old_score;
						old_score = score;
					}

				}
			}catch(Exception e){ 
				throw new RuntimeException(e);
			}
			if(rate_restarts)
				learnRate=learnRate.divide(5.0);
		}
		String ofilename=outputFilename+"-best.ff";
		System.out.println("Final performance on restart "+best_restart+", iteration "+best_iter+ ": "+ofilename);
		//System.out.println("beta="+beta);
		System.out.println(lfn +"= "+best_score);
		return best_params;

	}

	public void setTestTraining(boolean testTraining) {
		test_training = testTraining;
	}

}
