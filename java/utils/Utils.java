package utils;


import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.configuration.Configuration;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.ConfigurationFactory;
import org.apache.commons.configuration.PropertiesConfiguration;
import org.apache.commons.configuration.ConfigurationFactory.PropertiesConfigurationFactory;

import regularizer.L1;
import regularizer.L2;

import driver.Learner;
import driver.Tester;
import evaluation.FactoredLogL;
import evaluation.LogL;
import evaluation.LossFunction;
import evaluation.MSE;

public class Utils {
	public static final DecimalFormat df = new DecimalFormat(".####");
	public static final double epsilon = 1e-7; 
	public static Random rand = new Random();
	public static Object getProperty(HashMap<String, Object> properties, String string, Object def) {
		if(properties.containsKey(string))
			return properties.get(string);
		return def;
	}

	public static Object getProperty(HashMap<String, Object> properties, String string) {
		if(properties.containsKey(string))
			return properties.get(string);
		throw new RuntimeException("Required property "+string+" not specified");
	}

	public static double getDouble(HashMap<String, Object> properties, String string, double def) {
		if(properties.containsKey(string))
			return (Double)properties.get(string);
		return def;
	}

	public static int getInt(HashMap<String, Object> properties, String string, int def) {
		if(properties.containsKey(string))
			return (Integer)properties.get(string);
		return def;
	}

	public static String formatTime(long timeMillis){
		long time = timeMillis / 1000;  
		String seconds = Integer.toString((int)(time % 60));  
		String minutes = Integer.toString((int)((time % 3600) / 60));  
		String hours = Integer.toString((int)(time / 3600));  
		for (int i = 0; i < 2; i++) {  
			if (seconds.length() < 2) {  
				seconds = "0" + seconds;  
			}  
			if (minutes.length() < 2) {  
				minutes = "0" + minutes;  
			}  
			if (hours.length() < 2) {  
				hours = "0" + hours;  
			}  
		}
		return hours+":"+minutes+":"+seconds;
	}

	public static String printProperties(HashMap<String, Object> properties) {
		String result = "";
		for(String key:properties.keySet())
			result+=key+"="+properties.get(key);
		return null;
	}

	public static double log0(double arg){
		if(arg==0.0)
			return 0.0;
		else
			return Math.log(arg);
	}

	public static void setSeed(int seed) {
		rand=new Random(seed);
	}

	public static Options createOptions(){
		Options options = new Options();

		// add t option
		options.addOption("config", true, "Name of the configuration file.");
		options.addOption("help", false, "Display help message.");
		options.addOption("features", true, "The input factor graph.");
		options.addOption("pred_fname", true, "Name of file containing predictions (for the classify mode).");
		options.addOption("seed", true, "Random seed for data generation.\nOnly for <gen_data>.");
		options.addOption("data", true, "The output data file for <gen_data> or input data for <learn> and <test>.");
		options.addOption("out_ff", true, "Base filename to which the output factor graph is to be written (the ff will be written to '<out_ff>-i<iter>.ff'.\nAlso used for <classify> to specify the output file.");
		options.addOption("out_iters", false, "Output factor graphs for intermediate iterations. ffs will be written to '<out_ff>-i<iter>.ff').\nOnly for <learn>.");
		options.addOption("notest_train", false, "Do not run full testing during traing.");
		options.addOption("use_soft_decoder", false,"Use the soft version of the decoder.");
		options.addOption("use_micro_ave", false, "Use micro averaging (default is to use macro averaging).");
		options.addOption("cost_sensitive", false, "Cost-sensitive training/evaluation.");
		options.addOption("cost_alpha", true, "Parameter for accuracy/speed tradeoff");
		options.addOption("guided", false, "Use guidance in BP while training.");
		options.addOption("anneal", false, "Anneal sum into max.");
		options.addOption("softmax", false, "Use the softmax learning of Gimpel and Smith.");
		options.addOption("C", true, "Margin for the softmax method");
		options.addOption("rate_restart", false, "Reduce the learning rate when you restart.");
		options.addOption("output_doc_scores", false, "Output individual document scores during testing.");
		options.addOption("lfn", true, "Loss function to be used. \nOnly for <learn> and <test>.");
		options.addOption("rnd_ff", false, "When this flag is set <learn>ing starts with a randomized factor graph. Otherwise, it starts with the factor graph that is read.");
		options.addOption("learn_rate", true, "Learning rate. \nOnly for <learn>.");
		options.addOption("opt_alg", true, "Optimization algorithm to be used -- supported: SGD (stochastic gradient descent) and SMD (stochastic meta descent).");
		options.addOption("iter", true, "Number of training iterations. \nOnly for <learn>.");
		options.addOption("maxiter", true, "Maximum number of iterations for BP.");
		options.addOption("tol", true, "Stopping condition for BP -- messages are considered converged if the maximum difference is <tol.");
		options.addOption("beta", true, "Beta parameter for loss functions that require it. Only used in <test>.");
		options.addOption("reg_beta", true, "Beta parameter controlling the strength of regularizarion.");
		options.addOption("reg_func", true, "Function used for regularization to be used during training.");
		options.addOption("start_temp", true, "Starting temperature when using annealing.");
		options.addOption("end_temp", true, "End temperature when using annealing.");
		options.addOption("batch_size", true, "Batch size for the optimization algorithm.");
		options.addOption("num_restarts", true, "Number of random restarts.");
		options.addOption("lambda", true, "Lambda parameter for SMD.");
		options.addOption("mu", true, "Mu parameter for SMD.");
		options.addOption("r", true, "weight for speed vs. accuracy trade-off.");
		options.addOption("max_prod", false, "When this flag is set testing uses max-product BP. Otherwise, it uses sum-product BP.");
		options.addOption("verbose", true, "Verbosity level.");

		return options;
	}
	public static void configureLearner(Learner learner, HashMap<String, String> options){
		System.out.println(options);
		if(options.containsKey("seed"))
			Learner.seed = Integer.parseInt(options.get("seed"));
		if(options.containsKey("use_micro_ave"))
			learner.use_micro_ave = Boolean.parseBoolean(options.get("use_micro_ave"));
		if(options.containsKey("opt_alg")){
			String opt = options.get("opt_alg");
			if(opt.equals("SMD"))
				learner.runSMD=true;
		}
		if(options.containsKey("softmax"))
			learner.softmax = Boolean.parseBoolean(options.get("softmax"));
		if(options.containsKey("anneal"))
			learner.anneal_max = Boolean.parseBoolean(options.get("anneal"));
		if(options.containsKey("bp_guidance"))
			learner.bp_guidance = Boolean.parseBoolean(options.get("bp_guidance"));
		if(options.containsKey("cost_sensitive"))
			learner.cost_sensitive = Boolean.parseBoolean(options.get("cost_sensitive"));
		if(options.containsKey("cost_alpha"))
			learner.cost_alpha = Double.parseDouble(options.get("cost_alpha"));
		if(options.containsKey("r"))
			learner.r = Double.parseDouble(options.get("r"));
		if(options.containsKey("start_temp"))
			learner.start = Double.parseDouble(options.get("start_temp"));
		if(options.containsKey("end_temp"))
			learner.end = Double.parseDouble(options.get("end_temp"));
		if(options.containsKey("beta")){
			//The beta option overrides start and end
			learner.beta = Double.parseDouble(options.get("beta"));
			learner.start = learner.beta;
			learner.end = learner.beta;
		}
		if(options.containsKey("mu"))
			learner.mu = Double.parseDouble(options.get("mu"));
		if(options.containsKey("batch_size"))
			learner.batch_size = Integer.parseInt(options.get("batch_size"));
		if(options.containsKey("lambda"))
			learner.lambda = Double.parseDouble(options.get("lambda"));
		if(options.containsKey("reg_beta"))
			learner.reg_beta = Double.parseDouble(options.get("reg_beta"));
		if(options.containsKey("reg_func")){
			String rf = options.get("reg_func");
			if(rf.equals("L1"))
				learner.regFunction = new L1();
			else if(rf.equals("L2"))
				learner.regFunction = new L2();
		}
		if(options.containsKey("verbose"))
			learner.verbose = Integer.parseInt(options.get("verbose"));
		if(options.containsKey("maxiter"))
			learner.maxiter = Integer.parseInt(options.get("maxiter"));
		if(options.containsKey("num_restarts"))
			learner.num_restarts = Integer.parseInt(options.get("num_restarts"));
		if(options.containsKey("tol"))
			learner.tol = Double.parseDouble(options.get("tol"));
		if(options.containsKey("rndmz_ff"))
			learner.rndmz_ff = Boolean.parseBoolean(options.get("rndmz_ff"));
		if(options.containsKey("out_iters"))
			learner.out_iters = Boolean.parseBoolean(options.get("out_iters"));
		if(options.containsKey("notest_train"))
			learner.test_training = !Boolean.parseBoolean(options.get("notest_train"));
	}	 
	public static void configureTester(Tester tester, HashMap<String, String> options){
		System.out.println(options);
		if(options.containsKey("seed"))
			Learner.seed = Integer.parseInt(options.get("seed"));
		if(options.containsKey("use_micro_ave"))
			tester.use_micro_ave = Boolean.parseBoolean(options.get("use_micro_ave"));
		if(options.containsKey("bp_guidance"))
			tester.bp_guidance = Boolean.parseBoolean(options.get("bp_guidance"));
		if(options.containsKey("cost_sensitive"))
			tester.cost_sensitive = Boolean.parseBoolean(options.get("cost_sensitive"));
		if(options.containsKey("r"))
			tester.r = Double.parseDouble(options.get("r"));
		if(options.containsKey("beta")){
			//The beta option overrides start and end
			tester.beta = Double.parseDouble(options.get("beta"));
		}
		if(options.containsKey("verbose"))
			tester.verbose = Integer.parseInt(options.get("verbose"));
		if(options.containsKey("use_soft_decoder"))
			tester.use_soft_decoder = true;
		if(options.containsKey("maxiter"))
			tester.maxiter = Integer.parseInt(options.get("maxiter"));
		if(options.containsKey("tol"))
			tester.tol = Double.parseDouble(options.get("tol"));
	}	

	public static HashMap<String,String> optionsToMap(CommandLine cmd, HashMap<String,String> map){
		for(Option o:cmd.getOptions()){
			if(o.hasArg()){
				System.out.println(o.getOpt()+"="+o.getValue());
				map.put(o.getOpt(), o.getValue());
			}else{
				System.out.println(o);
				map.put(o.getOpt(), "true");
			}
		}
		return map;
	}
	public static HashMap<String,String> optionsToMap(Configuration config, CommandLine cmd, HashMap<String,String> map){
		Iterator<String> keys = config.getKeys();
		while(keys.hasNext()){
			String k=keys.next();
			map.put(k, config.getString(k));
		}
		return map;
	}
	public static HashMap<String, String> configureLearner(Learner learner, CommandLine cmd){
		///Configure a learner
		///The learner can be configured either through the command line or through a config
		///file with the command line taking precedence
		HashMap<String,String> options = new HashMap<String, String>();
		if(cmd.hasOption("config")){
			String configName = cmd.getOptionValue("config");
			System.out.println("Reading configuration from "+configName);
			PropertiesConfiguration config;
			try {
				config = new PropertiesConfiguration(configName);
			} catch (ConfigurationException e) {
				throw new RuntimeException(e);
			}
			optionsToMap(config, cmd, options);
		}else{
			System.out.println("Configuration file not specified");
		}
		optionsToMap(cmd, options);
		configureLearner(learner, options);
		return options;
	}
	public static HashMap<String, String> configureTester(Tester tester, CommandLine cmd){
		///Configure a learner
		///The learner can be configured either through the command line or through a config
		///file with the command line taking precedence
		HashMap<String,String> options = new HashMap<String, String>();
		if(cmd.hasOption("config")){
			String configName = cmd.getOptionValue("config");
			System.out.println("Reading configuration from "+configName);
			PropertiesConfiguration config;
			try {
				config = new PropertiesConfiguration(configName);
			} catch (ConfigurationException e) {
				throw new RuntimeException(e);
			}
			optionsToMap(config, cmd, options);
		}
		optionsToMap(cmd, options);
		configureTester(tester, options);
		return options;
	}
	public static LossFunction getLossFunction(HashMap<String, String> lConfig) {
		String loss = lConfig.get("lfn");
		return getLossFunction(loss);
	}
	public static LossFunction getLossFunction(String loss) {
		if("L1".equalsIgnoreCase(loss))
			return new evaluation.L1();
		if("LogL".equalsIgnoreCase(loss))
			return new LogL();
		if("MSE".equalsIgnoreCase(loss))
			return new MSE();
		if("FactoredLogL".equalsIgnoreCase(loss))
			return new FactoredLogL();
		throw new RuntimeException("Loss function "+loss+" not recognized.");
	}

}
