package data;

import evaluation.LossFunction;
import evaluation.MSE;
import featParser.FeatureFileParser;
import inference.BeliefPropagation;
import inference.InferenceAlgorithm;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import decoder.Decoder;
import decoder.Identity;
import decoder.Max;

import regularizer.L1;
import regularizer.Regularizer;
import regularizer.Zero;

import utils.Real;
import utils.Utils;

public class SpeedFeatureFile extends FeatureFile{
	private static final boolean DEBUG = false;
	private static final String[] vals = new String[]{"in","out","hid"};
	public double t = 16.0;
	public double threshold = .05;
	public double gateRate = 5.0;
	private boolean test = false;
	//Do the gates go in the exponent (the alternative computes expected factor weights summing on and off weight)
	private final boolean EXPONENT = true;
	public double sparsityAlpha;
	public FeatureFile gateFeatureFile;
	private HashMap<String, Object> infProperties;
	private FactorGraph origFg;
	public BeliefPropagation gbp;
	private Decoder dec;
	//private boolean train = false;

	public double getT() {
		return t;
	}
	public void setT(double t) {
		this.t = t;
	}
	public double getSparsityAlpha() {
		return sparsityAlpha;
	}
	public void setSparsityAlpha(double sparsityAlpha) {
		this.sparsityAlpha = sparsityAlpha;
	}


	public SpeedFeatureFile(){
		super();
	}
	public SpeedFeatureFile(FeatureFile f, String featureFilename, boolean test){
		String gateFeatureFilename = featureFilename+".gates";
		this.test = test;
		this.featureGroups = f.featureGroups;
		this.features = f.features;
		this.relations = f.relations;
		this.str = f.str;
		this.types = f.types;
		File file=new File(gateFeatureFilename);
		if(file.exists()){
			System.out.println("Loading feature file from "+gateFeatureFilename);
			try{
				FeatureFileParser fp = FeatureFileParser.createParser(gateFeatureFilename);
				gateFeatureFile=fp.parseFile();
			}catch(Exception e){
				throw new RuntimeException(e);
			}
		}else{
			if(test){
				gateFeatureFile=null;
			}else{
				System.out.println("File "+gateFeatureFilename+" not found. Creating feature file.");
				gateFeatureFile=new FeatureFile();
				Type t = new Type("bool");
				t.addValue("off");
				t.addValue("on");
				gateFeatureFile.addType("bool", t);
				ArrayList<String> values = new ArrayList<String>();
				values.add("*");
				ArrayList<String> onValues = new ArrayList<String>();
				onValues.add("on");
				ArrayList<Type> types = new ArrayList<Type>();
				types.add(t);
				HashSet<String> factorFeatures = new HashSet<String>();
				for(Feature feat:f.getFeatures()){
					String name = feat.getName();
					name = name.substring(0,name.indexOf('*'));
					factorFeatures.add(name);
				}
				for(String name:factorFeatures){
					for(int i=0; i<vals.length; i++){
						String v1=vals[i];
						for(int j=i; j<vals.length; j++){
							String v2=vals[j];
							String featName = name+"_"+v1+"_"+v2;
							gateFeatureFile.addFeature(featName, types, values);
							//gateFeatureFile.setWeight(featName, onValues, new Real(1.0));
						}
					}
				}
				//String gateName = 
				//			gateFeatureFile.addFeature("all_input", types, values);
				//			gateFeatureFile.addFeature("all_output", types, values);
				//			gateFeatureFile.addFeature("input_output", types, values);
				//			values.remove(0);
				//			values.add("on");
				//			gateFeatureFile.addFeature("on_prior", types, values);
				//			gateFeatureFile.setWeight("on_prior", values, new Real(1.0));
				//			values.remove(0);
				//			values.add("off");
				//			gateFeatureFile.addFeature("off_prior", types, values);
			}
		}
		//System.out.println(gateFeatureFile.printWeights());

		infProperties=new HashMap<String, Object>();
		infProperties.put("verbose",0);  // Verbosity (amount of output generated)
		infProperties.put("tol",1e-6);          // Tolerance for convergence
		infProperties.put("maxiter",2);  // Maximum number of iterations
		infProperties.put("update",BeliefPropagation.UpdateType.SEQFIX);
		infProperties.put("recordMessages", true);

		dec = new Identity();// Max();
	}



	@Override
	public FactorGraph toFactorGraph(DataSample samp) {
		FeatureFactorGraph result = (FeatureFactorGraph)samp.toFactorGraph();
		//System.out.println(result);
		//System.out.println(samp);
		FactorGraph gfg =null;
		ArrayList<Factor> facs_vec=new ArrayList<Factor>();
		ArrayList<ArrayList<HashMap<Feature,Double> > > feature_ref_vec=new ArrayList<ArrayList<HashMap<Feature,Double>>>();
		VariableSet vars = new VariableSet();
		HashMap<String,RV> varCoresp = new HashMap<String, RV>();
		for(RV v:result.getVariables()){
			varCoresp.put(v.getName(),new RV(v));
		}
		HashMap<String,RV> gateVars = new HashMap<String, RV>();

		if(gateFeatureFile!=null){
			DataSample gates = new DataSample(gateFeatureFile);

			Type boolType = gateFeatureFile.getType("bool");
			for( int I = 0; I < result.numFactors(); I++ ) {
				Factor fI = result.getFactor(I);
				if(fI.nrNeighbors()>1){
					String rvName = "g"+fI.toString();
					RV vr = gates.addVariable(rvName, boolType, "on");
					gateVars.put(rvName, vr);
					//gates.addFeatureInst(rvName, "", "on_prior");
					//gates.addFeatureInst(rvName, "", "off_prior");
					RV v1 = (RV)fI.getNeighbor(0).getSecond(), v2=(RV)fI.getNeighbor(1).getSecond();
//					String str1 = v1.isInput()?"v"+v1.getValue():(v1.isHidden()?"hid":"out");
//					String str2 = v2.isInput()?"v"+v2.getValue():(v2.isHidden()?"hid":"out");
					String str1 = v1.isInput()?"in":(v1.isHidden()?"hid":"out");
					String str2 = v2.isInput()?"in":(v2.isHidden()?"hid":"out");
					String str3 = makeString(str1,str2);
					
					String facName = result.getFactorFeatures(I).get(0).keySet().iterator().next().getName();
					facName = facName.substring(0,facName.indexOf('*'));
					//System.out.println("Adding feature "+facName+"_"+str1+"_"+str2);
					gates.addFeatureInst(rvName, "", facName+str3,1.0);
					//				boolean input=false, output = false;
					//				for(Edge<Factor,RV> e:fI.getNeighbors()){
					//					input |=e.getSecond().isInput();
					//					output |=(!e.getSecond().isInput());
					//				}
					//				if(input&&output){
					//					gates.addFeatureInst(rvName, "", "input_output");
					//				}else if(input){
					//					gates.addFeatureInst(rvName, "", "all_input");
					//				}else{
					//					gates.addFeatureInst(rvName, "", "all_output");
					//				}
					//				ArrayList<HashSet<Feature>> fFeatures = result.getFactorFeatures(I);
					//				for(HashSet<Feature> feats:fFeatures){
					//					for(Feature f:feats){
					//						gates.addFeatureInst(rvName, "", f.getName());
					//					}
					//				}
				}

			}
			gfg = gates.toFactorGraph();
			if(DEBUG)
				System.out.println(gfg);
			gbp = new BeliefPropagation(gfg, infProperties);
			//bp.setRunReverse(true);
			//System.out.println(bp.printProperties());
			//Run inference
			gbp.run();
			dec.softDecode(gbp,t);
			if(DEBUG) 
				for(RV v:gbp.getFactorGraph().getVariables()) // iterate over all variables in fg
					System.out.println(v+": "+v.getDecode()+" val="+v.getValue()+(v.isOutput()?"*":"")); // display the belief of bp for that variable



		}
		int next=0;
		for( int I = 0; I < result.numFactors(); I++ ) {
			Factor fI = result.getFactor(I);
			if(DEBUG)
				System.out.println("Orig "+fI+fI.getCondTable().toString());
			Probability cond = fI.getCondTable();

			//Real sum = new Real(0.0);
			//for( int s = 0; s < fI.states(); s++ )
			//	sum.sumEquals(cond.getValue(s).product(cond.getValue(s)));
			Real damp = new Real(1.0);
			if(gfg!=null&&fI.nrNeighbors()>1){
				String rvName = "g"+fI.toString();
				if(DEBUG)
					System.out.println(rvName+" = "+gateVars.get(rvName));
				RV v = gfg.getVariable(gfg.getVariableIndex(rvName));
				damp = v.getDecode().getValue(v.getType().getValue("on"));
			}
			//			Real thr = threshold(sum);
			if(DEBUG)
				System.out.println("thr="+damp);
			//System.out.println("Adding");
			Probability newCond = new Probability(cond.size());
			for(int s =0; s<cond.size(); s++){
				if(EXPONENT)
					newCond.setValue(s, cond.getValue(s).pow(damp));
				else
					newCond.setValue(s, cond.getValue(s).product(damp).sum(new Real(1.0).minus(damp)));
			}
			if(strength(newCond).gt(threshold)||(!test)){

				VariableSet rvs = new VariableSet();
				for(RV v:fI.getVars())
					rvs.add(varCoresp.get(v.getName()));
				int id = fI.id;
				Factor fNew = new Factor(id,rvs);
				fNew.setCondTable(newCond);
				facs_vec.add(fNew);
				vars.addAll(fI.getVars());
				feature_ref_vec.add(result.getFactorFeatures(I));
			}
		}

		for (int j=0; j<result.numVariables(); j++){
			if(!vars.contains(result.getVariable(j))){
				Factor fnew = new Factor(varCoresp.get(result.getVariable(j).getName()),1.0);
				//System.out.println(fnew.toString());
				facs_vec.add(fnew);
				feature_ref_vec.add(new ArrayList<HashMap<Feature,Double>>());
			}
		}
		if(DEBUG)
			for(Factor f:facs_vec)
				System.out.println(f+f.getCondTable().toString());
		FeatureFactorGraph ffg = new FeatureFactorGraph(facs_vec,feature_ref_vec);
		if(DEBUG)
			System.out.println(result+"\n*************************************\n"+ffg);
		origFg = result;
		return ffg;
	}
	private String makeString(String str1, String str2) {
		String first=str1,second=str2;
		int i1 =0, i2=0;
		for(int i=0; i<vals.length; i++){
			if(str1.equals(vals[i]))
				i1=i;
			if(str2.equals(vals[i]))
				i2=i;
		}
		if(i1>i2){
			first = str2;
			second = str1;
		}
		return "_"+first+"_"+second;
	}
	@Override
	public void accumulateGradient(InferenceAlgorithm ia, Regularizer r, double reg_beta){
		FeatureFactorGraph bpFg = (FeatureFactorGraph)ia.getFactorGraph();
		//cout + *bp_fg + endl;
		FactorGraph gfg = gbp.getFactorGraph();
		//Probability[] orig = new Probability[bpFg.numFactors()];
		for( int I = 0; I < bpFg.numFactors(); I++ ) {
			Factor fI = bpFg.getFactor(I);
			String fName = fI.toString();
			Real strength = strength(fI.getCondTable());		
//			Real denum = Real.exp(strength.product(-t)).sum(1.0);
//			Real g = new Real(-sparsityAlpha).product(Real.exp(strength.product(-t)).product(2.0).divide(denum.product(denum)));
			
			Real g = gradSigmoid(new Real(-sparsityAlpha), strength);
			Probability factorGrad = gradStrength(g,fI.getCondTable());
//			if(fName.equals("f5")){
//				System.out.println(fI.getCondTable()+" -- "+strength);
//				System.out.println(g+" -- "+denum+"=="+new Real(-sparsityAlpha).product(Real.exp(strength.product(-t)).product(2.0))+"--"+sparsityAlpha);
//				System.out.println(fName+": "+factorGrad);
//			}
			Real gateVal;
			Probability gI = fI.getGradient();
			gI.add(factorGrad);
			Probability table = fI.getCondTable();
			//orig[I] = new Probability(gI);
			Probability origTable = origFg.getFactor(I).getCondTable();
//			if(!origTable.equals(table)){
//				System.out.println(origTable+" vs. "+table);
//			}
			int onVal = -1,offVal=-1;
			boolean incl = fI.getNeighbors().size()>1;
			RV gateVar=null;
			if(incl){
				//System.out.println(gI);
				String rvName = "g"+fI.toString();
				
				gateVar = gfg.getVariable(gfg.getVariableIndex(rvName));
				
				onVal = gateVar.getType().getValue("on");
				offVal = gateVar.getType().getValue("off");
				gateVal= gateVar.getDecode().getValue(onVal);
			}else{
				gateVal = new Real(1.0);
			}
			Probability gateGrad = new Probability(2,0.0);
			for(int s=0; s<fI.states();s++){
				//dt += dpsi'*psi^t*log(psi)
				//System.out.println(gI.getValue(s)+"  "+origTable.getValue(s)+"  "+gateVal);
				//System.out.println("g+= "+gI.getValue(s).product(table.getValue(s).product(Real.log(origTable.getValue(s)))));
				if(incl){
					if(EXPONENT)
						gateGrad.incrValue(onVal, gI.getValue(s).product(table.getValue(s).product(Real.log(origTable.getValue(s)))));
					else{
						gateGrad.incrValue(onVal, gI.getValue(s).product(origTable.getValue(s)));
						gateGrad.incrValue(offVal, gI.getValue(s));
					}
				}
				//dpsi += dpsi'*t*psi^(t-1)
				if(EXPONENT){
					gI.setValue(s,gI.getValue(s).product(gateVal.product(origTable.getValue(s).pow(gateVal.minus(1.0)))));
				}else{
					gI.setValue(s,gI.getValue(s).product(gateVal));
				}
			}
			//Add the speed gradient
			//int offVal = gateVar.getType().getValue("off");
			if(incl){
//				Real denum = Real.exp(gateVal.product(-t)).sum(1.0);
//				Real g = Real.exp(gateVal.product(-t)).product(2.0).divide(denum.product(denum)).product(sparsityAlpha);
//				gateGrad.incrValue(onVal, g);
//				gateGrad.incrValue(onVal, sparsityAlpha);
				gateVar.setGradient(gateGrad);
				//if(Utils.rand.nextDouble()<10.0/(double)bpFg.numVariables())
				//	System.out.println(gateVar+" "+gateVar.getGadient()+" "+sparsityAlpha);
			}
			//System.out.println(gateVar+" "+gateGrad);
			fI.setCondTable(origTable);
		}
		//testGradient(bpFg, gfg, ia.getProperties());
		dec.reverseDecode(gbp,t);
		//		for(RV v:gfg.getVariables()) // iterate over all variables in fg
		//			System.out.println(v+": "+(v.getGadient()==null?"nil":v.getGadient().toString()));

		gbp.reverse();
		gateFeatureFile.accumulateGradient(gbp, r, reg_beta);

		super.accumulateGradient(ia, r, reg_beta);
//		for( int I = 0; I < ia.getFactorGraph().numFactors(); I++ ){
//			if(!orig[I].equals(ia.getFactorGraph().getFactor(I).getGradient())){
//				System.out.println(orig[I]+" vs. "+ia.getFactorGraph().getFactor(I).getGradient());
//			}
//		}
		/*		FeatureFactorGraph bpFg = (FeatureFactorGraph)ia.getFactorGraph();
			//cout + *bp_fg + endl;
			for( int I = 0; I < bpFg.numFactors(); I++ ) {
				Factor fI = bpFg.getFactor(I);
				Probability cond = fI.getCondTable();
				Real sum = new Real(0.0);
				for( int s = 0; s < fI.states(); s++ )
					sum.sumEquals(cond.getValue(s).product(cond.getValue(s)));
				Real thr = threshold(sum);
				//dthr = (2*t*exp(pow(sum,-t))*pow(sum,-t-1))/(((Real)1+exp(pow(sum,-t)))*((Real)1+exp(pow(sum,-t))));
				Real dthr = Real.exp(sum.pow(-t)).product(sum.pow(-t-1)).product(t*2.0).divide(Real.exp(sum.pow(-t)).sum(1)).product(Real.exp(sum.pow(-t)).sum(1));
				System.out.println("dthr="+dthr);
				for( int j = 0; j < fI.states(); j++){
					//Propagate the adjoint to the factor's elements
					Real lam = cond.getValue(j);
					Real value = Real.exp(lam.product(thr));
					Probability gI = fI.getGradient();
					Real gIj = new Real(gI.getValue(j));
					//gI[j]+=gIj*value*thr;
					gI.setValue(j, gI.getValue(j).sum(gIj.product(value).product(thr)));
					for( int t = 0; t < fI.states(); t++ ){
						//gI[t] += gIj*value*((Real)2*lam*fac[t]*dthr);
						gI.setValue(t, gI.getValue(t).sum(gIj.product(value).product(lam.product(2).product(cond.getValue(t)).product(dthr))));
					}
					//gI[j]+=sparsity_alpha*(2*lam*dthr)/numVars();
					gI.setValue(j, gI.getValue(j).sum(lam.product(2).product(dthr).product(sparsityAlpha).divide(bpFg.numVariables())));
				}
			}
			for( int I = 0; I < bpFg.numFactors(); I++ ) {
				ArrayList<HashSet<Feature>> featureRefs = bpFg.getFactorFeatures(I);
				Factor fI = bpFg.getFactor(I);
				//System.out.println(fI.getGradient());
				for( int j = 0; j < fI.getCondTable().size(); j++){
					HashSet<Feature> features = featureRefs.get(j);
					for(Feature f : features){
						f.accumGradient(fI.getGradient().getValue(j).product(fI.getCondTable().getValue(j)));
						if(Double.isNaN(f.getGradient().getValue())){
							throw new RuntimeException("grad+="+bpFg.getFactor(I).getGradient().getValue(j)+"*"+bpFg.getFactor(I).getCondTable().getValue(j));
						}
					} 
				}
			}
		 */
	}
	private Real strength(Probability condTable) {
		Real result = new Real(0.0);
		for(int i=0; i<condTable.size(); i++){
			Real ln = Real.log(condTable.getValue(i));
			result.sumEquals(ln.product(ln));
		}
		return result;
	}
	private Probability gradStrength(Real grad,Probability condTable){
		Probability result = new Probability(condTable.size(),0.0);
		for(int i=0; i<condTable.size(); i++){
			result.setValue(i, grad.divide(condTable.getValue(i)));
		}
		return result;
	}
	private Real sigmoid(Real x){
		if(x.lt(1e-7))
			return new Real(0.0);
		return new Real(2).divide(new Real(1.0).sum(Real.exp(x.product(-t)))).minus(1.0);
	}
	private Real gradSigmoid(Real grad, Real strength){
		Real denum = Real.exp(strength.product(-t)).sum(1.0);
		Real g = grad.product(Real.exp(strength.product(-t)).product(2.0).divide(denum.product(denum)));
		return g;
	}
	@Override
	public boolean updateWeights(double rate, double score, boolean runSMD, double lambda,
			double mu, int batchSize, Regularizer r, double regBeta) {
		//Update weights in the original model
		//System.out.println(curTime+" vs. "+super.getCurrentTime());
		super.updateWeights(rate, score, runSMD, lambda, mu, batchSize, r, regBeta);

		//Update weights in the gate model

		gateFeatureFile.updateWeights(gateRate*rate, score, runSMD, lambda, mu, batchSize, new Zero(), 0.0);

		return true;
	}

	public void initializeRandom(){
		for(Feature f:features.values()){
			f.setWeight(new Real(2.0*(rndm.nextDouble()-.5)));
		}
	}
	@Override
	public SpeedFeatureFile copy() {
		SpeedFeatureFile result = new SpeedFeatureFile();
		result.types = types;
		result.curTime = curTime;
		result.features = new HashMap<String, Feature>(features.size());
		for(Entry<String, Feature> feat: features.entrySet()){
			Feature newF = new Feature(feat.getValue());
			newF.setFeatFile(result);
			result.features.put(feat.getKey(), newF);
		}
		result.featureGroups = new HashMap<String, ArrayList<Feature>>();
		for(Entry<String, ArrayList<Feature>> featG: featureGroups.entrySet()){
			ArrayList<Feature> feats = new ArrayList<Feature>();
			for(Feature f:featG.getValue()){
				feats.add(result.features.get(f.getName()));
			}
			result.featureGroups.put(featG.getKey(), feats);	
		}
		result.relations = new HashMap<String, ArrayList<Feature>>();
		for(Entry<String, ArrayList<Feature>> featG: relations.entrySet()){
			ArrayList<Feature> feats = new ArrayList<Feature>();
			for(Feature f:featG.getValue()){
				feats.add(result.features.get(f.getName()));
			}
			result.relations.put(featG.getKey(), feats);	
		}
		result.str = str;
		result.gateFeatureFile = gateFeatureFile.copy();
		result.infProperties = infProperties;
		return result;
	}
	@Override
	public void print(PrintWriter out) {
		super.print(out);
		System.out.println(gateFeatureFile.toString());
	}
	public void print(String ofilename){
		PrintWriter ofile;
		try {
			ofile = new PrintWriter(new FileWriter(ofilename));
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		super.print(ofile);

		String gfile = ofilename+".gates";
		gateFeatureFile.print(gfile);

	}
	@Override
	public void initializeSMDstructures(Real learn_rate){
		super.initializeSMDstructures(learn_rate);
		gateFeatureFile.initializeSMDstructures(learn_rate.product(gateRate));
	}

	public double testGradient(FactorGraph fg, FactorGraph gfg, HashMap<String,Object> infProperties){
		double diff = 0;
		//		for(Factor f:fg.getFactors()){
		//			f.getGradient().fill(0.0);
		//		}
		double delta = 1e-7;
		LossFunction loss = new evaluation.L1();
		Decoder dec = loss.getDecoder();
		//infProperties.put("maxiter",2);
		BeliefPropagation obp = new BeliefPropagation(fg, infProperties);

		obp.setRunReverse(true);
		//infProperties.put("verbose",0);
		obp.run();
		dec.softDecode(obp,t);
		Real value = loss.evaluate(obp);
		//		loss.revEvaluate(obp);
		//		if(obp.isRunReverse()){
		//			dec.reverseDecode(obp,t);
		//			obp.reverse();
		//		}
		System.out.println(loss.printLoss(value));
		System.out.println("Gate: Theta Gradients");
		ArrayList<Factor> facs = fg.getFactors();
		for(int i=0;i<facs.size(); i++){
			Factor fI = facs.get(i);
			Probability g = fI.getCondTable();
			Probability grad = new Probability(g.size(), 0.0);
			String rvName = "g"+fI.toString();
			RV gateVar = gfg.getVariable(gfg.getVariableIndex(rvName));
			int onVal = gateVar.getType().getValue("on");
			Real gateVal = gateVar.getDecode().getValue(onVal);

			for(int k=0; k<g.size(); k++){
				Real oldVal = g.getValue(k);
				System.out.println(g.getValue(k));
				g.setValue(k, origFg.getFactor(i).getCondTable().getValue(k).pow(gateVal.sum(delta)));
				//g.setValue(k, g.getValue(k).sum(delta));
				BeliefPropagation bp = new BeliefPropagation(fg, infProperties);
				bp.run();
				dec.softDecode(bp,t);
				Real newValue = loss.evaluate(bp);
				//System.out.println("New value = "+newValue + "\t("+value+")");
				grad.setValue(k, newValue.minus(value).divide(delta));
				g.setValue(k, oldVal);
			}
			System.out.println(" "+ rvName + " vs. "+grad.sum());
			System.out.println(" "+fI+": "+grad + " vs. "+fI.getGradient()/*+ "diff="+Math.abs(Probability.distL1(grad, fI.getGradient()))*/);
			diff+=Math.abs(Probability.distL1(grad, fI.getGradient()));

		}
		return diff;
	}
}
