package data;

import utils.Real;

public class Message {
	public enum MessageType{FACTOR_VAR,VAR_FACTOR};
	Edge e;
	MessageType type;

	public Probability mes;
	public Real normalizer;
	public Edge getEdge() {
		return e;
	}
	public void setEedge(Edge e) {
		this.e = e;
	}	
	public MessageType getType(){
		return type;
	}
	public Message(Edge e, Probability mes, Real normalizer, MessageType type) {
		super();
		this.e = e;
		this.mes = mes;
		this.normalizer = normalizer;
		this.type = type;
	}
	public Probability getMessage() {
		return mes;
	}
	public Real getNormalizer() {
		return normalizer;
	}
	
	public String toString(){
		return (type==MessageType.FACTOR_VAR?"m: ":"n: ")+e.getSecond()+"->"+e.getFirst()+" "+mes;
	}
	
}
