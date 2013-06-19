package data;

import java.util.ArrayList;
import java.util.HashMap;

public abstract class GraphNode implements Comparable<GraphNode>{
	protected int id;
	protected int fgNum;
	private ArrayList<Edge> neighbors = new ArrayList<Edge>();
	private ArrayList<Integer> dualIndex = new ArrayList<Integer>();
	private HashMap<GraphNode, Integer> refs = new HashMap<GraphNode, Integer>();

	public int nrNeighbors(){
		return neighbors.size();
	}
	@SuppressWarnings("unchecked")
	public ArrayList<Edge> getNeighbors(){
		return neighbors;
	}
	@SuppressWarnings("unchecked")
	public Edge getNeighbor(int _I){
		return neighbors.get(_I);
	}
	@SuppressWarnings("unchecked")
	public static void addEdge(GraphNode g1, GraphNode g2){
		if(g1.refs.containsKey(g2)||g2.refs.containsKey(g1))
			throw new RuntimeException("Nodes already connected");
		Edge n1 = new Edge(g2,g1,g1.nrNeighbors(),g2.nrNeighbors());
		Edge n2 = new Edge(g1,g2,g2.nrNeighbors(),g1.nrNeighbors());
		g1.refs.put(g2,g1.nrNeighbors());
		g2.refs.put(g1,g2.nrNeighbors());		
		g1.dualIndex.add(g2.nrNeighbors());
		g2.dualIndex.add(g1.nrNeighbors());
		g1.neighbors.add(n2);
		g2.neighbors.add(n1);
	}
	@Override
	public int compareTo(GraphNode o) {
		return this.toString().compareTo(o.toString());
	}

	public Integer getDualIndex(int i) {
		return dualIndex.get(i);
	}
	public int getNumNeighbors(){
		return neighbors.size();
	}
	public int getNeighborIndex(GraphNode g) {
		return refs.get(g);
	}
	public int getFgNum() {
		return fgNum;
	}
	public void setFgNum(int fgNum) {
		this.fgNum = fgNum;
	}
}
