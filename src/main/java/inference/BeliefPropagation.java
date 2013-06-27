package inference;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;

import utils.Index;
import utils.Real;
import utils.Utils;
import data.DataSample;
import data.Edge;
import data.Factor;
import data.FactorGraph;
import data.FeatureFile;
import data.Message;
import data.Probability;
import data.RV;
import data.SpeedFeatureFile;
import data.VariableSet;
import data.Message.MessageType;
import decoder.Decoder;
import evaluation.LossFunction;

public class BeliefPropagation extends DifferentiableInferenceAlgorithm{
	//	private class WeightEdge implements Comparable<WeightEdge>{
	//		double weight = 0.0;
	//		Edge e;
	//		public WeightEdge(double weight, Edge e){
	//			this.weight=weight;
	//			this.e = e;
	//		}
	//		@Override
	//		public int compareTo(WeightEdge o) {
	//			if(weight<o.weight)
	//				return -1;
	//			if(weight>o.weight)
	//				return +1;
	//			int comp = e.getFirst().compareTo(o.e.getFirst());
	//			return comp==0?e.getSecond().compareTo(o.e.getSecond()):comp;
	//		}
	//	}
	public enum UpdateType {SEQMAX,SEQRND,SEQFIX,PARALL}
	private static final String Name = "BP";
	///Variable to factor messages
	private Probability[][] nMessages;
	///Factor to variable messages
	private Probability[][] mMessages;
	///Variable to factor messages
	private Probability[][] gradNMessages;
	///Factor to variable messages
	private Probability[][] gradMMessages;
	private Real[][] Zm;
	private Real[][] Zn;
	///Intermediate products 
	private Probability[][] product;
	//Factor beliefs
	private Probability[] fBeliefs;
	//Variable beliefs
	private Probability[] vBeliefs;
	private Real[] Zbf;
	private Real[] Zbv;

	//	private PriorityQueue<WeightEdge> residuals;
	/// Does the gradient computation algorithm need to run the reverse pass
	private boolean runReverse=false;
	private int iterations = 0;
	private boolean guided = false;
	private double guidanceAlpha = 0.0;
	//	private double gamma;
	public int verbose;
	private HashMap<String, Object> properties;
	private int maxiter;
	private UpdateType update;
	private double tol;
	private double _maxdiff;
	private boolean useGuidance;
	private boolean recordSentMessages;
	private ArrayList<Message> sentMessages;
	private ArrayList<ArrayList<ArrayList<Integer>>> indices;

	public BeliefPropagation(FactorGraph f,HashMap<String, Object> properties){
		fg=f;
		this.properties = properties;
		readProperties();
		//		this.residuals = new PriorityQueue<WeightEdge>();
		initialize();
	}

	private void readProperties(){
		verbose = Utils.getInt(properties, "verbose", 0);
		maxiter = Utils.getInt(properties, "maxiter", 100);
		update = (UpdateType)Utils.getProperty(properties, "update");
		tol = Utils.getDouble(properties, "tol", 0.0);
		recordSentMessages = (Boolean)Utils.getProperty(properties, "recordMessages");
	}

	public String printProperties(){
		String result = "verbose="+verbose+"\n";
		result+="maxiter="+maxiter+"\n";
		result+="update="+update+"\n";
		result+="tol="+tol+"\n";
		result+="recordSentMessages="+recordSentMessages;
		return result;
	}
	private void initialize() {
		runReverse=true;
		vBeliefs = new Probability[fg.numVariables()];
		fBeliefs = new Probability[fg.numFactors()];
		//		for(Factor f:fg.getFactors()){
		//			System.out.println(f+"("+f.getFgNum()+")");
		//		}
		//		for(RV v:fg.getVariables()){
		//			System.out.println(v+"("+v.getFgNum()+")");
		//		}
		Zbv = new Real[fg.numVariables()];
		Zbf = new Real[fg.numFactors()];
		nMessages = new Probability[fg.numVariables()][];
		mMessages = new Probability[fg.numFactors()][];
		Zm = new Real[fg.numFactors()][];
		Zn = new Real[fg.numVariables()][];
		for(int i=0; i<fg.numVariables(); i++){
			RV v = fg.getVariable(i);
			nMessages[i]=new Probability[v.nrNeighbors()];
			Zn[i]=new Real[v.nrNeighbors()];
			for(int j=0; j<v.getNumNeighbors(); j++){
				nMessages[i][j]= new Probability(v.numValues(),new Real(1.0));
			}
		}
		for(int i=0; i<fg.numFactors(); i++){
			Factor f = fg.getFactor(i);
			mMessages[i]=new Probability[f.nrNeighbors()];
			Zm[i] = new Real[f.nrNeighbors()];
			for(int j=0; j<f.getNumNeighbors(); j++){
				Edge<Factor,RV> gn = (Edge<Factor,RV>)f.getNeighbor(j);
				RV v = gn.getSecond();
				mMessages[i][j]= new Probability(v.numValues(),new Real(1.0));
			}
		}
		sentMessages = new ArrayList<Message>();
	}
	private void initializeReverse() {
		product = new Probability[fg.numVariables()][];
		for(int i=0; i<fg.numVariables(); i++){
			RV v = fg.getVariable(i);
			product[i]=new Probability[v.nrNeighbors()];
			for(int j=0; j<v.getNumNeighbors(); j++){
				product[i][j]= new Probability(v.numValues(),-1.0);
			}
		}

	}
	String identify(){
		return Name+Utils.printProperties(properties);
	}
	public ArrayList<RV> topoSort(FactorGraph fg){
		ArrayList<RV> result = new ArrayList<RV>();
		LinkedList<RV> agenda = new LinkedList<RV>();
		HashSet<RV> unused = new HashSet<RV>(fg.getVariables());
		HashSet<RV> used = new HashSet<RV>();
		RV first = fg.getVariable(0);
		agenda.add(first);
		do{
			if(agenda.size()<1){
				agenda.add(unused.iterator().next());
				used.add(agenda.peek());
			}
			RV current = agenda.pop();
			while(!unused.contains(current)){
				if(agenda.size()<1){
					agenda.add(unused.iterator().next());
					used.add(agenda.peek());
				}
				current = agenda.poll();
			}
			unused.remove(current);
			result.add(current);

			for(Edge<RV,Factor> e :current.getNeighbors()){
				Factor f = e.getSecond();
				for(Edge<Factor,RV> e1:f.getNeighbors()){
					RV v = e1.getSecond();
					if(!used.contains(v)){
						agenda.add(v);
						used.add(v);
					}
				}
			}
		}while(unused.size()>0);
		return result;
 	}
	public double run(){
		if( verbose >= 1 )
			System.out.print( "Starting " + identify() + "...");
		if( verbose >= 3)
			System.out.println();

		long tic = System.currentTimeMillis();
		//ArrayList<Double> diffs = new ArrayList<Double>();
		double newMaxDiff = Double.MAX_VALUE;

		//int nredges = fg.numFactors();
		ArrayList<Edge<RV,Factor>> updateSeq = new ArrayList<Edge<RV,Factor>>();
		//RV[] vars= fg.getVariables().toArray(new RV[0]);
		ArrayList<RV> vars = topoSort(fg);
		HashSet<RV> visited = new HashSet<RV>();
		LinkedList<Edge<RV,Factor>> returnSeq = new LinkedList<Edge<RV,Factor>>();
		for(RV varI:vars){
//			RV varI = fg.getVariable(i);
			//System.out.println(v+"("+v.getFgNum()+")");
			visited.add(varI);
			if(!varI.isInput()){
				for( int j = varI.nrNeighbors()-1; j>=0; j-- ){
					Edge<RV, Factor> ej = varI.getNeighbor(j);
					boolean vis = true;
					Factor f = (Factor)ej.getSecond();
					for(Edge<Factor,RV> e:f.getNeighbors()){
						if(!visited.contains(e.getSecond())){
							vis = false;
							break;
						}
					}
					if(!vis)
						returnSeq.addFirst(ej);
					else
						updateSeq.add(ej);
				}
			}
		}
		updateSeq.addAll(returnSeq);
		//int numUpdates = updateSeq.size();
		if(verbose>=5)
			System.out.println( "Update sequence: " + updateSeq );

		Probability[] oldBeliefs = new Probability[fg.numVariables()];
		for(int i=0; i<fg.numVariables(); i++)
			oldBeliefs[i] = calcBeliefV(i);
		// do several passes over the network until maximum number of iterations has
		// been reached or until the maximum belief difference is smaller than tolerance
		for( iterations=0; iterations < maxiter && newMaxDiff > tol; ++iterations ) {
			if( update == UpdateType.PARALL ) {
				// Parallel updates
				for( int i = 0; i < fg.numVariables(); ++i )
					for( Edge<RV,Factor> e: fg.getVariable(i).getNeighbors() ){
						sendMMessage(e);
						//calcNewN(e);
					}


				for( int I = 0; I < fg.numFactors(); ++I )
					for( Edge<Factor,RV> e: fg.getFactor(I).getNeighbors() )
						sendNMessage(e);
			} else {
				// Sequential updates
				if( update == UpdateType.SEQRND )
					Collections.shuffle(updateSeq);

				for( Edge<RV,Factor> e: updateSeq ) {
					//Updating message factor I --> variable j
					//First, update all messages incoming into factor I
					for( Edge<Factor,RV> ve:e.getSecond().getNeighbors()){
						//System.out.println("$$ "+e+" * "+ve);
						if(!ve.getSecond().equals(e.getFirst()))
							sendNMessage(ve);
					}
					sendMMessage(e);
				}
			}

			// calculate new beliefs and compare with old ones
			newMaxDiff = 0.0;
			for( int i = 0; i < fg.numVariables(); ++i ) {
				Probability bel = new Probability(calcBeliefV(i));
				if(verbose>=5)
					System.out.println(fg.getVariable(i)+ " bel: "+bel);
				double curDiff = Probability.distLInf(bel, oldBeliefs[i]);
				newMaxDiff = curDiff>newMaxDiff?curDiff:newMaxDiff;
				oldBeliefs[i] = bel;
			}

			if( verbose >= 3 )
				System.out.println( Name+".run:  maxdiff "+newMaxDiff+" after "+(iterations+1)+" passes");
		}

		if( newMaxDiff > _maxdiff )
			_maxdiff = newMaxDiff;

		if( verbose >= 1 ) {
			if( newMaxDiff > tol ) {
				if( verbose == 1 )
					System.out.println();
				System.out.println(Name+".run:  WARNING: not converged within "+maxiter+" passes ("+Utils.formatTime(System.currentTimeMillis()- tic)+" seconds)...final maxdiff:"+newMaxDiff);
			} else {
				if( verbose >= 3 )
					System.out.print(Name+".run:  ");
				System.out.println("converged in "+iterations+" passes ("+Utils.formatTime(System.currentTimeMillis()-tic) +").");
			}
		}

		calcBeliefs();
		return newMaxDiff;
	}


	//	private double max(ArrayList<Real> diffs) {
	//		double max = Double.MIN_VALUE;
	//		for(Real r:diffs){
	//			if(r.getValue()>max)
	//				max=r.getValue();
	//		}
	//		return max;
	//	}

	//	void updateMessageM( Edge e ) {
	//		int i = e.getFirst().getFgNum(), _I = e.getIndSecond();
	//	    if(recordSentMessages || verbose>=5)
	//	        sentMessages.add(new Message(fg.getVariable(i),_I,mMessages[i][_I],Zm[i][_I]));
	//	    if(verbose >= 5){
	//	       	System.out.println("Message "+sentMessages.size()+" ("+i+","+fg.getVariable(i).getNeighbor(_I)+"):" + mMessages[i][_I]);
	//	    }
	//
	//	    mMessages[i][_I] = newMessage[i][_I];
	//	    if( update == UpdateType.SEQMAX )
	//	    	updateResidual( i, _I, new Real(0.0) );
	//	    if( verbose >= 5){
	//	    	System.out.println(" --> " + mMessages[i][_I]);
	//	    }
	//	}
	public double max(ArrayList<Double> diffs) {
		double max = Double.MIN_VALUE;
		for(Double r:diffs){
			if(r>max)
				max=r;
		}
		return max;
	}
	@SuppressWarnings("unchecked")
	private void sendMMessage(Edge<RV,Factor> nI) {
		// calculate updated message factor I --> variable i
		if(verbose>=6){
			System.out.println("m"+nI.getSecond()+"_"+nI.getFirst());
		}

		RV vi = nI.getFirst();
		int i = vi.getFgNum();
		Factor fI = nI.getSecond();
		int I = fI.getFgNum();

		Probability p = new Probability( fI.getCondTable() );
		if(verbose>=6)
			System.out.println("prod init="+p);
		for (int j = 0; j < fI.getNumNeighbors(); j++) {
			Edge<Factor,RV> vj = fI.getNeighbor(j);
			if( vj.getSecond() != vi ) { // for all j in I \ i
				Probability n = nMessages[vj.getSecond().getFgNum()][vj.getIndFirst()]; 
				if(verbose>=6)
					System.out.println("Mult "+n);
				Index ind = new Index( vj.getSecond(), fI.getVars() );
				for( int x = 0; ind.isValid(); ind.increment() ){
					int cur  = ind.current();
					if(verbose>=6)
						System.out.println(n+"   "+cur+" x="+x);
					p.setValue(x, p.getValue(x).product(n.getValue(cur)));
					x++;
				}
			}
			if(verbose>=6)
				System.out.println("prod="+p);
		}
		// Marginalize onto i
		Probability marg =  new Probability( fg.getVariable(i).numValues(), new Real(0.0) );
		// ind is the pre-calculated Index(i,I) i.e. to x_I == k corresponds x_i == ind[k]
		Index ind = new Index( fg.getVariable(i), fI.getVars() );
		for( int x = 0; ind.isValid(); x++, ind.increment() )
			marg.setValue(ind.current(), marg.getValue(ind.current()).sum(p.getValue(x)));

		Zm[I][nI.getIndFirst()] = marg.normalize();
		if(recordSentMessages){
			sentMessages.add(new Message(nI,mMessages[I][nI.getIndFirst()],Zm[I][nI.getIndFirst()],MessageType.FACTOR_VAR));
		}
		mMessages[I][nI.getIndFirst()] = marg;

		if(verbose>=5){
			System.out.println("m: "+nI.getSecond()+"->"+nI.getFirst()+"="+mMessages[I][nI.getIndFirst()]);
		}

	}


	private void unsendMMessage(Message me){
		//Reverse message fI->vj.
		//Compute the appropriate adjoints
		if(verbose>=5)
			System.out.println("undo "+me);
		Edge<RV,Factor> nI=me.getEdge();
		Probability message=me.getMessage();
		Real normalizer = me.getNormalizer(); 
		Factor fI = nI.getSecond();
		int I = fI.getFgNum();
		RV vj = nI.getFirst();
		int j = vj.getFgNum();
		int _I = nI.getIndSecond();
		//System.out.println(j+"  "+_I);
		int _j = nI.getIndFirst();
		Zm[I][_j]=normalizer;
	
		//First compute d(psi_F[I]) += sum_{i \in I}(prod_{j \in I; j!=i}N_{j->I}(x_j)) d(unnorm_M_{I->i}(x_i)
		Probability _adj_m_unnorm_jI = gradMMessages[I][_j];
	
		if(verbose>=5){
			System.out.println( "Message M["+I+","+_j+"]="+mMessages[I][_j]+"("+Zm[I][_j]+")");
			System.out.print( "adj_m["+I+","+_j+"]="+_adj_m_unnorm_jI);
		}
		//System.out.println( "_adj_m_jI"+_adj_m_unnorm_jI+endl;
		_adj_m_unnorm_jI = unnormalize(mMessages[I][_j],normalizer,_adj_m_unnorm_jI);
	
		if(verbose>=5)
			System.out.println("(" +_adj_m_unnorm_jI+")["+Zm[I][_j]+"]");
		//compute adj_psi_F[I]
		//_adj_m_unnorm[j][_I] = _adj_m_unnorm_jI;
		Probability um = new Probability(U(fI.getNeighbor(_j)));
		//System.out.println( "--b"+um *fg.factor(I).p()+endl;
		//System.out.println( "_adj_m_unnorm_jI"+_adj_m_unnorm_jI+endl;
		ArrayList<Integer> ind = indices.get(j).get(_I);
		//Probability sum = new Probability( vj.numValues(), 0.0 );
		for( int x_I = 0; x_I < um.size(); x_I++ ){
			um.setValue(x_I,um.getValue(x_I).product(_adj_m_unnorm_jI.getValue(ind.get(x_I))));
			//	    	}
		}
		//System.out.println( "um"+I+"=" +um+endl;
		//um *= 1 - props.damping;
		//	    if(inference == SMAXPROD){
		//	    	//System.out.println( props.gamma;
		//	    	for (int l = 0; l < um.size(); ++l) {
		//	    		um[l] = um[l]*props.gamma*(fg.factor(I)[l]^(props.gamma-(Real)1.0));
		//	    	}
		//	    }
		if(verbose>=5)
			System.out.println("gradient["+fI +"]+=" +um);
		fI.getGradient().add(um);
	
		//Now compute d(n_{j->I}(x_j))+= sum_{x_i}(sum_{x/{x_i,x_j}}psi_{I}(x_I) prod_{k!=i,j}(n_{k->I}(x_k)) d(unnormM_{I_i}(x_i)
		//System.out.println( "for";
		for(Edge<Factor,RV> ni: fI.getNeighbors()) {
			RV vi = ni.getSecond();
			if( !vi.equals(vj) ) {
				//Probability &S = _Smsg[i][i.dual][_j];
				Probability Sp = S(vi.getFgNum(),ni.getIndFirst(),_j);
				Probability msg = new Probability(vi.numValues(), 0.0 );
	
	
				//System.out.println( "Sp="+Sp);
				//Prob& n = nMessages( i.node, i.dual );
				int i_states = vi.numValues();         
				int j_states = vj.numValues();         
				int xij=0;    
				if(vi.compareTo(vj)>0){
					for(int xi=0; xi<i_states; xi++) {       
						for(int xj=0; xj<j_states; xj++) {
							//System.out.println("msg[+"+xi+"]+="+Sp.getValue(xij)+"*"+_adj_m_unnorm_jI.getValue(xj));
							msg.setValue(xi,msg.getValue(xi).sum(Sp.getValue(xij).product(_adj_m_unnorm_jI.getValue(xj))));
							xij++;                              
						}                                       
					}                      
				}else{
					for(int xj=0; xj<j_states; xj++) {
						for(int xi=0; xi<i_states; xi++) {       
							//System.out.println("msg[+"+xi+"]+="+Sp.getValue(xij)+"*"+_adj_m_unnorm_jI.getValue(xj));
							msg.setValue(xi,msg.getValue(xi).sum(Sp.getValue(xij).product(_adj_m_unnorm_jI.getValue(xj))));
							xij++;                              
						}                                       
					}
				}
				assert( ni.getSecond().getNeighbor(ni.getIndFirst()).getSecond().equals(fI));
				//System.out.println("gradNMessages["+vi.getFgNum()+"]["+ni.getIndFirst()+"]+="+msg);
				gradNMessages[vi.getFgNum()][ni.getIndFirst()].add(msg);
			}
		}
		//System.out.println( "done unsend message";
		gradMMessages[I][_j].fill(0.0);//(fg.var(j).states(),0.0);
	
		//Unroll the message that was sent to arrive at the previous state
		mMessages[I][_j]=message;
	}


	private void sendNMessage( Edge<Factor,RV> nI ) {
		// calculate updated message i->I

		RV vi = nI.getSecond();
		int val = vi.getValue();
		//System.out.println(val);
		int i = vi.getFgNum();
		if(!vi.isInput()){  
			Factor fI = nI.getFirst();
			Probability prod = new Probability( fg.getVariable(i).numValues(), new Real(1.0) );
			for(int j=0; j<vi.getNumNeighbors();j++){
				Edge<RV,Factor> nfJ = vi.getNeighbor(j);
				Factor fJ = nfJ.getSecond();

				if( !fJ.equals(fI) ) // for all J in i \ I
					prod.multiply(mMessages[fJ.getFgNum()][nfJ.getIndFirst()]);
			}
			if (guided && useGuidance && !vi.isHidden()){
				val = -(val+2);
				//Real marginalizer = prod.sum();
				prod.multiply(1.0-guidanceAlpha);
				prod.setValue(val,prod.getValue(val).sum(guidanceAlpha));
			}
			Zn[i][nI.getIndFirst()] = new Real(1);
			if(recordSentMessages){
				sentMessages.add(new Message(nI,nMessages[i][nI.getIndFirst()],Zn[i][nI.getIndFirst()],MessageType.VAR_FACTOR));
			}
			nMessages[i][nI.getIndFirst()] = prod;

		}else{
			Probability p = new Probability( fg.getVariable(i).numValues(), new Real(0.0) );
			//variable var_j is observed. The message should put all the probability on the observed value
			if((int)val>=fg.getVariable(i).numValues())
				throw new RuntimeException();
			p.setValue(val,new Real(1.0));
			Zn[i][nI.getIndFirst()] = new Real(1.0);
			if(recordSentMessages){
				sentMessages.add(new Message(nI,nMessages[i][nI.getIndFirst()],Zn[i][nI.getIndFirst()],MessageType.VAR_FACTOR));
			}
			nMessages[i][nI.getIndFirst()] = p;
		}
		if(verbose>=5){
			System.out.print("n: "+nI.getSecond()+"->"+nI.getFirst());
			System.out.println("="+nMessages[i][nI.getIndFirst()]);
		}
	}

	private void unsendNMessage(Message me){
		///Reverse n message from variable to factor
		///Differentiation of private void sendNMessage( Edge<Factor,RV> nI )
		Edge<Factor,RV> nI = me.getEdge();
		RV vi = nI.getSecond();
		if(vi.isInput())
			return;
		int i = vi.getFgNum();
		if(verbose>=5){
			System.out.println("undo "+me);
			System.out.println("        grad="+gradNMessages[i][nI.getIndFirst()]);
		}
		//System.out.println( "adj_n("+i+","+fg.nbV(i,_I)+")+="+f+endl;

		Probability totalProd = new Probability(calcProduct(i,nI.getIndFirst()));
		Real guidance_discount = new Real((guided && vi.getValue()<-1)?1.0-guidanceAlpha:1.0);
		for(Edge B: vi.getNeighbors()){
			if(!nI.getFirst().equals(B.getSecond())){
				Probability r = new Probability(totalProd);
				r.divide((mMessages[B.getSecond().getFgNum()][B.getIndFirst()]));
				Probability p = new Probability(gradNMessages[i][nI.getIndFirst()]);
				p.multiply(r);
				//System.out.println("product="+totalProd+"  message="+mMessages[B.getSecond().getFgNum()][B.getIndFirst()]);
				p.multiply(guidance_discount);
				if(verbose>=5)
					System.out.println("adj_m["+B.getSecond().getFgNum()+","+B.getIndFirst()+"]+="+p);
				gradMMessages[B.getSecond().getFgNum()][B.getIndFirst()].add(p);
			}
		}
		gradNMessages[i][nI.getIndFirst()].fill(0.0);
		nMessages[i][nI.getIndFirst()]=me.getMessage();
		Zn[i][nI.getIndFirst()]=me.getNormalizer();

		//		RV vj = nI.getSecond();
		//		//Invalidate the corresponding caches
		//	    nMessages[vj.getFgNum()][nI.getIndSecond()].setValue(i, r)[0]=-1.0;
		//	    product(var_j,_I)[0]=(Real)-1.0;
	}


	//	private Probability calcMsgN( Edge<Factor,RV> nI ){
	//	       	if(nMessages[nI.getFirst().getFgNum()][nI.getIndSecond()].getValue(0).lt(0.0))
	//	       		calcNewN(nI);
	//	       	return nMessages[nI.getFirst().getFgNum()][nI.getIndSecond()];
	//	}

	private Probability calcProduct( int i, int _I ){
		if( product[i][_I].getValue(0).lt(0.0)){
			// calculate updated message i->I
			RV vi = fg.getVariable(i);
			Probability prod = new Probability(vi.numValues(), new Real(1.0));
			//Factor fI = (Factor)fg.getVariable(i).getNeighbor(_I).getSecond();
			for(Edge<RV,Factor> nJ:vi.getNeighbors()){
				Factor fJ = nJ.getSecond();
				if(_I!=nJ.getIndSecond()){
					//System.out.println("prod*=mMessages["+fJ+"]["+nJ.getFirst()+"]="+mMessages[fJ.getFgNum()][nJ.getIndFirst()]);
					prod.multiply(mMessages[fJ.getFgNum()][nJ.getIndFirst()]);
				}
			}
			product[i][_I] = prod;
		}
		return product[i][_I];
	}

	public void calcBeliefs() {
		for( int i = 0; i < fg.numVariables(); i++ )
			calcBeliefV(i);  // calculate b_i
		for( int I = 0; I < fg.numFactors(); I++ )
			calcBeliefF(I);  // calculate b_I
	}

	private void reverseBeliefs() {

		int nv = fg.numVariables();
		gradMMessages = new Probability[fg.numFactors()][];
		gradNMessages = new Probability[nv][];
		for(int i=0; i<fg.numFactors(); i++){
			gradMMessages[i] = new Probability[fg.getFactor(i).getNumNeighbors()];
		}
		for( int i = 0; i < fg.numVariables(); i++ ) {
			int n_i = fg.getVariable(i).nrNeighbors();        
			gradNMessages[i] = new Probability[n_i];
			RV vI = fg.getVariable(i);
			for( Edge nI:vI.getNeighbors()) {
				// calculate adj_m
				Probability prod = unnormalize(vBeliefs[i], Zbv[i], vI.getGadient());
				assert(prod.size() == vI.numValues());
				for(Edge nJ: vI.getNeighbors() )
					if( !nI.getSecond().equals(nJ.getSecond()))
						prod.multiply(mMessages[nJ.getSecond().getFgNum()][nJ.getIndFirst()]);// *= mMessages( i, J.iter );
				gradMMessages[nI.getSecond().getFgNum()][nI.getIndFirst()] = prod;
			}
		}

		for( int i = 0; i < fg.numVariables(); i++ ){
			RV vI = fg.getVariable(i);
			for( Edge nI:vI.getNeighbors() ) {
				if(!vI.isInput()){
					//Real guidance_discount = new Real((guided && vI.getValue()<-1)?1.0-guidanceAlpha:1.0);
					// calculate adj_n
					Factor fI = (Factor)nI.getSecond();
					Probability prod = new Probability(fI.getCondTable());
					//System.out.println(fBeliefs[fI.getFgNum()]+" 2 "+Zbf[fI.getFgNum()]+" 3 "+fI.getGradient());
					prod.multiply(unnormalize(fBeliefs[fI.getFgNum()],Zbf[fI.getFgNum()], fI.getGradient()));
					//System.out.println( "*="+_adj_b_F_unnorm[I]+endl;
					for(Edge nj: fI.getNeighbors() ){
						if( i != nj.getSecond().getFgNum()) {
							RV vj = (RV)nj.getSecond();
							Probability n_jI = new Probability(vj.numValues(), 1.0);
							if(vj.getValue()<0){
								for(Edge J:vj.getNeighbors()){
									System.out.println(J);
									if( J.getSecond() != nI.getSecond() ) { // for all J in nb(j) \ I
										n_jI.multiply(mMessages[vj.getFgNum()][J.getSecond().getFgNum()]);
									}
								}
							}else{
								n_jI.fill(0.0);
								n_jI.setValue(vj.getValue(),1.0);
							}
							//System.out.println(nj.getSecond().getFgNum()+" ind "+indices);//+indices.get(nj.getSecond().getFgNum()));
							ArrayList<Integer> ind = indices.get(nj.getSecond().getFgNum()).get(nj.getIndFirst());
							// multiply prod with n_jI
							for( int x_I = 0; x_I < prod.size(); x_I++ ){
								prod.setValue(x_I,prod.getValue(x_I).product(n_jI.getValue(ind.get(x_I))));
								//System.out.println( "   (x_"+x_I+"*="+n_jI[ind[x_I]]+endl;
							}
						}
					}
					Probability marg = new Probability(fg.getVariable(i).numValues(), 0.0);
					ArrayList<Integer> ind = indices.get(i).get(nI.getIndSecond());
					for( int r = 0; r < prod.size(); r++ )
						marg.getValue(ind.get(r)).sumEquals(prod.getValue(r));
					gradNMessages[i][nI.getIndSecond()]=marg;
					for(Edge B:fg.getVariable(i).getNeighbors()){
						if(nI.getFirst()!=B.getFirst()){
							marg.multiply(R(nI.getFirst().getFgNum(),i,B.getFirst().getFgNum()));
							gradMMessages[i][B.getIndSecond()].add(marg);
							//System.out.println( "adj_m["+i+","+B.node+"]+="+adj_n(i,I.iter)+"*R" +R(I,i,B.iter)+"="+marg*R(I,i,B.iter)+endl;
						}
					}

				}else{

					Probability marg = new Probability( fg.getVariable(i).numValues(), 0.0);
					gradNMessages[i][nI.getIndSecond()]=marg ;
				}
			}
		}
		//Compute gradients for the input factors
		//Real guidance_discount = (Real)((props.guided && fg.getVarValue(i)<-1)?1.0-props.guidance_alpha:1.0);
		for( int I = 0; I < fg.numFactors(); I++ ) {
			Factor fI = fg.getFactor(I);
			Probability p = new Probability(unnormalize(fBeliefs[fI.getFgNum()],Zbf[fI.getFgNum()], fI.getGradient()));
			Probability bel_I = new Probability(fI.getCondTable());
			assert( p.size() == fI.states() );
			for(Edge ni: fI.getNeighbors()) {
				RV vi = (RV)ni.getSecond();
				Probability n_iI = new Probability(vi.numValues(), 1.0);// nMessages( i, i.dual ) );
				if(!vi.isInput()){
					for(Edge nJ: vi.getNeighbors() )
						if(!nJ.getSecond().equals(fI)) { // for all J in nb(j) \ I
							n_iI.multiply(mMessages[nJ.getSecond().getFgNum()][nJ.getIndFirst()]);
						}
				}else{
					n_iI.fill(0.0);
					n_iI.setValue(vi.getValue(),1.0);
				}
				ArrayList<Integer> ind = indices.get(vi.getFgNum()).get(ni.getIndFirst());
				// multiply prod with n_jI
				for( int x_I = 0; x_I < p.size(); x_I++ ){
					bel_I.setValue(x_I, bel_I.getValue(x_I).product(n_iI.getValue(ind.get(x_I))));
					//System.out.println( "n("+x_I+")="+n_iI[ind[x_I]]+endl;
					p.setValue(x_I, p.getValue(x_I).product(n_iI.getValue(ind.get(x_I))));
				}
			}
			bel_I.normalize();
			fI.setGradient(p);
		}

	}

	private Probability calcBeliefV( int i ) {
		//		if(verbose>=5){
		//			System.out.println("CalcBelief "+i);
		//		}
		RV vi = fg.getVariable(i);
		int val = vi.getValue();
		if(!vi.isInput()){	       		
			Probability prod = new Probability(vi.numValues(), new Real(1.0));
			for(int j=0; j<vi.getNumNeighbors();j++){
				Edge<RV,Factor> e = vi.getNeighbor(j);
				//				if(verbose>=5){
				//					System.out.println("Edge "+e+" "+e.getSecond()+"("+e.getSecond().getFgNum()+") *= mMes["+e.getSecond().getFgNum()+"]["+e.getIndFirst()+"]"+mMessages[e.getSecond().getFgNum()][e.getIndFirst()]);
				//				}
				prod.multiply(mMessages[e.getSecond().getFgNum()][e.getIndFirst()]);
			}
			Zbv[i] = prod.normalize();
			vBeliefs[i] = prod;
		}else{
			if((int)val>=fg.getVariable(i).numValues())
				throw new RuntimeException("Value is "+val+" max is "+fg.getVariable(i).numValues());
			if((int)val<0)
				throw new RuntimeException("Variable "+vi.toString()+"Value is "+val);
			Probability p = new Probability( fg.getVariable(i).numValues(), new Real(0.0));
			p.setValue(val,new Real(1.0));
			Zbv[i] = new Real(1.0);
			vBeliefs[i] = p;
		}
		//System.out.println(vBeliefs[i]);
		return vBeliefs[i];
	}

	public Probability beliefV( int i ) {
		return vBeliefs[i];
	}
	public void setBeliefV(int i, Probability b){
		vBeliefs[i]=b;
	}
	public Probability beliefF( int i ) {
		return fBeliefs[i];
	}

	private Probability calcBeliefF( int I ) { 

		Probability prod= new Probability(fg.getFactor(I).getCondTable());
		//System.out.println("prod="+prod);
		Factor fI = (Factor)fg.getFactor(I);
		for(int j=0; j<fI.getNumNeighbors();j++){
			Edge<Factor,RV> nj=fI.getNeighbor(j);
			RV vj = (RV)nj.getSecond();
			Probability prod_j;
			if(!vj.isInput()){
				sendNMessage(nj);
				prod_j = nMessages[vj.getFgNum()][nj.getIndFirst()];
			}else{
				prod_j = new Probability(vj.numValues(),0.0);
				prod_j.setValue(vj.getValue(),1.0);
			}
			//System.out.println("prod_j="+prod_j);
			Index ind=new Index( vj, fg.getFactor(I).getVars() );
			for( int x = 0; ind.isValid(); x++){
				//System.out.println("prod["+x+"]*="+prod_j.getValue(ind.current()));
				prod.setValue(x,prod.getValue(x).product(prod_j.getValue(ind.current())));
				ind.increment();
			}
			//System.out.println("prod="+prod);
		}

		//System.out.println("bel="+prod);
		Zbf[I] = prod.normalize();
		fBeliefs[I] = prod;
		return prod;
	}

	public void printMessages(){
		System.out.println(" N messages ");
		printVector(nMessages);
		System.out.println(" M messages ");
		printVector(mMessages);

	}

	public void printVector(Probability[][] vec){
		for(int i=0; i<vec.length; i++){
			for(int j=0; j<vec[i].length; j++)
				System.out.print("["+vec[i][j]+"]");
			System.out.println();
		}
	}

	Probability unnormalize(Probability w, Real Z_w, Probability gradW ) {
		//System.out.println(w);
		//System.out.println(gradW);
		assert(w.size()== gradW.size());
		Probability unnormGrad = new Probability(w.size(), 0.0);
		Real s = new Real(0.0);
		for( int i = 0; i < w.size(); i++ )
			s.sumEquals(w.getValue(i).product(gradW.getValue(i)));
		for( int i = 0; i < w.size(); i++ )
			unnormGrad.setValue(i, (gradW.getValue(i).minus(s)).divide(Z_w));
		return unnormGrad;
	}

	@Override
	public void reverse() {
		if(runReverse){
			initializeReverse();
			createIndices();
			reverseBeliefs();
			//printMessages();
			for(int i = sentMessages.size()-1; i>=0; i--){
				Message me = sentMessages.get(i);
				if(me.getType().equals(MessageType.FACTOR_VAR)){
					unsendMMessage(me);
				}else{
					unsendNMessage(me);
				}
			}
		}
	}
	public void createIndices() {
		// initialise _indices
		//     typedef std::vector<int>        _ind_t;
		//     std::vector<std::vector<_ind_t> >  _indices;
		indices = new ArrayList<ArrayList<ArrayList<Integer>>>(fg.numVariables());
		for( int i=0; i < fg.numVariables(); i++ ) {
			indices.add(new ArrayList<ArrayList<Integer>>());
			for(Edge I: fg.getVariable(i).getNeighbors() ) {
				ArrayList<Integer> index = new ArrayList<Integer>();
				for(Index k = new Index(fg.getVariable(i), ((Factor)I.getSecond()).getVars()); k.isValid(); k.increment() )
					index.add(k.index());
				indices.get(i).add(index);
			}
		}
	}

	private Probability R(int I, int i, int J) {
		// _Rmsg[I][_i][_J]
		//Neighbor i = fg.var(i);
		//Neighbor J = fg.nbV(i,_J);
		assert(I!=J);
		Probability prod = new Probability(fg.getVariable(i).numValues(), 1.0);
		for(Edge K: fg.getVariable(i).getNeighbors()){
			if( K.getFirst().getFgNum() != I && K.getIndSecond() != J )//J.node )
				prod.multiply(mMessages[i][K.getIndSecond()] );
		}
		return prod;
	}
	private Probability S(int i, int _I, int _j) {
		// _Smsg[i][_I][_j]
		Edge<RV,Factor> nI = fg.getVariable(i).getNeighbor(_I);
		Factor fI = nI.getSecond();
		Edge<Factor,RV> nj = fI.getNeighbor(_j);
		Factor prod =  new Factor(fI);
		//	    if (inference == GBP::Properties::InfType::SMAXPROD ){
		//	    	//cout << "g="<<props.gamma << endl;
		//	    	for(int i =0; i<prod.states(); i++){
		//	    		prod[i]=prod[i]^props.gamma;
		//	    	}
		//	    }
		for(Edge<Factor,RV> k: fI.getNeighbors()) {
			if( k.getSecond().getFgNum() != i && !k.getSecond().equals(nj.getSecond())) {
				ArrayList<Integer> ind = indices.get(k.getSecond().getFgNum()).get(k.getIndFirst() );
				Probability p = new Probability(nMessages[k.getSecond().getFgNum()][ k.getIndFirst()]);
				Probability prodp = prod.getCondTable();
				for( int x_I = 0; x_I < prodp.size(); x_I++ )
					prodp.setValue(x_I, prodp.getValue(x_I).product(p.getValue(ind.get(x_I))));
			}
		}
		// "Marginalize" onto i|j (unnormalized)
		//cout << "prod="<<prod <<endl;
		VariableSet vs = new VariableSet();
		vs.add(fg.getVariable(i));
		vs.add(nj.getSecond());
		Factor marg = prod.marginalize(vs, false );
		//cout << "marg="<<marg <<endl;
		return marg.getCondTable();
	}
	private Probability U(Edge<Factor,RV> nI) {
		// _Umsg[I][_i]
		Factor fI = nI.getFirst();
		Probability prod = new Probability( fI.states(), 1.0 );
		for(Edge nj: fI.getNeighbors() ){
			if( !nj.getSecond().equals(nI.getSecond())) {
				Probability n_jI = new Probability( nMessages[nj.getSecond().getFgNum()][nj.getIndFirst()]);
				ArrayList<Integer> ind = indices.get(nj.getSecond().getFgNum()).get(nj.getIndFirst());
				// multiply prod by n_jI
				//cout << "nn" << n_jI<<endl;
				for( int x_I = 0; x_I < prod.size(); x_I++ )
					prod.setValue(x_I, prod.getValue(x_I).product(n_jI.getValue(ind.get(x_I))));
			}
		}
		return prod;
	}

	public double testBackPropInitialization(LossFunction loss, Decoder dec, double beta, double delta, boolean print){
		Real value = loss.evaluate(this);
		boolean oldRecordMessages = recordSentMessages;
		recordSentMessages = false;
		int oldVerbose = verbose;
		verbose = 0;
		if(runReverse){
			createIndices();
			reverseBeliefs();
			double diff = 0;
			//Compute finite-differences gradients. First for loss
			//calcBeliefs();

			if(print)
				System.out.println("M Gradients");
			for(int i=0;i<mMessages.length; i++){
				Probability[] gmm = mMessages[i];
				for(int j=0; j<gmm.length; j++){
					Probability g = gmm[j];
					Probability grad = new Probability(g.size(), 0.0);
					for(int k=0; k<g.size(); k++){
						g.setValue(k, g.getValue(k).sum(delta));
						calcBeliefs();
						dec.softDecode(this,beta);
						Real newValue = loss.evaluate(this);
						//System.out.println(value+"-"+newValue+"/"+delta);
						grad.setValue(k, newValue.minus(value).divide(delta));
						g.setValue(k, g.getValue(k).minus(delta));
					}
					if(print)
						System.out.println("\t: "+grad + " vs. "+gradMMessages[i][j]);
					diff+=Math.abs(Probability.distL1(grad, gradMMessages[i][j]));
				}
			}
			if(print)
				System.out.println("N Gradients");
			for(int i=0;i<nMessages.length; i++){
				Probability[] gmm = nMessages[i];
				for(int j=0; j<gmm.length; j++){
					Probability g = gmm[j];
					Probability grad = new Probability(g.size(), 0.0);
					for(int k=0; k<g.size(); k++){
						g.setValue(k, g.getValue(k).sum(delta));
						calcBeliefs();
						dec.softDecode(this,beta);
						Real newValue = loss.evaluate(this);
						grad.setValue(k, newValue.minus(value).divide(delta));
						g.setValue(k, g.getValue(k).minus(delta));
					}

					if(print)
						System.out.println("\t: "+grad + " vs. "+gradNMessages[i][j]);
					diff+=Math.abs(Probability.distL1(grad, gradNMessages[i][j]));
				}
			}
		}
		double diff = 0.0;
		if(print)
			System.out.println("Theta-Gradients");
		ArrayList<Factor> facs = fg.getFactors();
		for(int i=0;i<facs.size(); i++){
			Factor fI = facs.get(i);
			Probability g = fI.getCondTable();
			Probability grad = new Probability(g.size(), 0.0);
			for(int k=0; k<g.size(); k++){
				g.setValue(k, g.getValue(k).sum(delta));
				calcBeliefs();
				dec.softDecode(this,beta);
				Real newValue = loss.evaluate(this);
				grad.setValue(k, newValue.minus(value).divide(delta));
				g.setValue(k, g.getValue(k).minus(delta));
			}

			if(print)
				System.out.println("  "+fI+": "+grad + " vs. "+fI.getGradient());
			diff+=Math.abs(Probability.distL1(grad, fI.getGradient()));

		}
		calcBeliefs();
		dec.softDecode(this,beta);
		recordSentMessages = oldRecordMessages;
		verbose = oldVerbose;
		return diff;
	}

	public double testBackProp(LossFunction loss, Decoder dec, double beta, double delta, boolean print){
		reverse();
		double diff = 0;
		//		printMessages();
		//		for(int i = 0; i<fg.numVariables(); i++){
		//			RV vi = fg.getVariable(i);
		//			System.out.println(beliefV(i)+" --> "+vi.getDecode());
		//		}
		//		initialize();
		//		run();
		//		dec.softDecode(this,beta);
		Real value = loss.evaluate(this);
		//		System.out.println("=========================");
		//		printMessages();
		//		for(int i = 0; i<fg.numVariables(); i++){
		//			RV vi = fg.getVariable(i);
		//			System.out.println(beliefV(i)+" --> "+vi.getDecode());
		//		}
		verbose = 0;
		if(print)
			System.out.println("BP: Theta Gradients");
		ArrayList<Factor> facs = fg.getFactors();
		for(int i=0;i<facs.size(); i++){
			Factor fI = facs.get(i);
			Probability g = fI.getCondTable();
			Probability grad = new Probability(g.size(), 0.0);
			for(int k=0; k<g.size(); k++){
				g.setValue(k, g.getValue(k).sum(delta));
				initialize();
				run();
				dec.softDecode(this,beta);
				Real newValue = loss.evaluate(this);
				//System.out.println("New value = "+newValue + "\t("+value+")");
				grad.setValue(k, newValue.minus(value).divide(delta));
				g.setValue(k, g.getValue(k).minus(delta));
			}

			if(print)
				System.out.println(" "+fI+": "+grad + " vs. "+fI.getGradient()/*+ "diff="+Math.abs(Probability.distL1(grad, fI.getGradient()))*/);
			diff+=Math.abs(Probability.distL1(grad, fI.getGradient()));

		}
		return diff;
	}

	public static double testBackProp(SpeedFeatureFile ff, DataSample ex, HashMap<String, Object> properties,LossFunction loss, Decoder dec, double beta, double delta, boolean print){
		FactorGraph fg = ff.toFactorGraph(ex);
		//System.out.println(fg.toString());
		BeliefPropagation bp = new BeliefPropagation(fg, properties);
		bp.initialize();
		bp.run();
		dec.softDecode(bp,beta);
		Real value = loss.evaluate(bp);
		loss.revEvaluate(bp);
		dec.reverseDecode(bp, beta);
		bp.reverse();
		double diff = 0;
		//		printMessages();
		//		for(int i = 0; i<fg.numVariables(); i++){
		//			RV vi = fg.getVariable(i);
		//			System.out.println(beliefV(i)+" --> "+vi.getDecode());
		//		}
		//		initialize();
		//		run();
		//		dec.softDecode(this,beta);
		//		System.out.println("=========================");
		//		printMessages();
		//		for(int i = 0; i<fg.numVariables(); i++){
		//			RV vi = fg.getVariable(i);
		//			System.out.println(beliefV(i)+" --> "+vi.getDecode());
		//		}
		bp.verbose = 0;
		if(print)
			System.out.println("BP: Theta Gradients");
		ArrayList<Factor> facs = fg.getFactors();
		for(int i=0;i<facs.size(); i++){
			Factor fI = facs.get(i);
			Probability g = fI.getCondTable();
			Probability grad = new Probability(g.size(), 0.0);
			for(int k=0; k<g.size(); k++){
				g.setValue(k, g.getValue(k).sum(delta));
				bp.initialize();
				bp.run();
				dec.softDecode(bp,beta);
				Real newValue = loss.evaluate(bp);
				//System.out.println("New value = "+newValue + "\t("+value+")");
				grad.setValue(k, newValue.minus(value).divide(delta));
				g.setValue(k, g.getValue(k).minus(delta));
			}

			if(print)
				System.out.println(" "+fI+": "+grad + " vs. "+fI.getGradient()/*+ "diff="+Math.abs(Probability.distL1(grad, fI.getGradient()))*/);
			diff+=Math.abs(Probability.distL1(grad, fI.getGradient()));

		}
		return diff;
	}

	@Override
	public HashMap<String, Object> getProperties() {
		return this.properties;
	}

	@Override
	public Real logZ() {
	    Real sum = new Real(0.0);
	    for(int i = 0; i < fg.numVariables(); i++ ){
	    	//System.out.println(beliefV(i).entropy());
	        sum= sum.sum(beliefV(i).entropy().product((1.0 - fg.getVariable(i).getNumNeighbors())));
	    }
    	//System.out.println("sum="+sum);
	    for( int I = 0; I < fg.numFactors(); ++I ){
	    	//System.out.println(Probability.distKL(beliefF(I), fg.getFactor(I).getCondTable()));
	    	sum = sum.minus(Probability.distKL(beliefF(I), fg.getFactor(I).getCondTable()));
	    }
	    return sum;
	}

	@Override
	public void setProperties(HashMap<String, Object> props) {
		this.properties = props;
		readProperties();
	}

	public void setRunReverse(boolean runReverse) {
		this.runReverse = runReverse;
	}

	public boolean isRunReverse() {
		return runReverse;
	}

	public int numIterations() {
		return iterations;
	}
}
