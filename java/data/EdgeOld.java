package data;

public class EdgeOld implements Comparable<EdgeOld>{
	GraphNode first;
	GraphNode second;

	public EdgeOld(GraphNode n1, GraphNode n2) {
		this.first = n1;
		this.second = n2;
	}

	public GraphNode getFirst() {
		return first;
	}

	public void setFirst(GraphNode first) {
		this.first = first;
	}

	public GraphNode getSecond() {
		return second;
	}

	public void setSecond(GraphNode second) {
		this.second = second;
	}

	@Override
	public int compareTo(EdgeOld o) {
		// An arbitrary order over edges
		if(o==null)
			return -1;
		if(this.equals(o))
			return 0;
		if(this.first.compareTo(o.first)!=0)
			return this.first.compareTo(o.first);
		else
			return this.second.compareTo(o.second);
	}
	
	
}
