package data;

import java.util.ArrayList;

public class Edge<F extends GraphNode,T extends GraphNode> {
	/// The reference node
	F first;
	/// The neighbor node
	T second;
    /// Contains the index for the second entry in the neighbours of the first
    int indSecond;
    /// Contains the index for the first entry in the neighbours of the second
    int indFirst;
	
	public Edge(F first, T second, int indFirst, int indSecond) {
		this.first = first;
		this.second = second;
		this.indSecond = indSecond;
		this.indFirst = indFirst;
	}

	public T getSecond() {
		return second;
	}

	public void setSecond(T node) {
		this.second = node;
	}

	public int getIndSecond() {
		return indSecond;
	}

	public void setIndSecond(int iter) {
		this.indSecond = iter;
	}

	public int getIndFirst() {
		return indFirst;
	}

	public void setIndFirst(int dual) {
		this.indFirst = dual;
	}

	public F getFirst() {
		return first;
	}
	public String toString(){
		return first.toString()+" --> "+second.toString();
	}
}
