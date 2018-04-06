import java.util.HashMap;
import java.util.List;


public class MyBNInferencer implements Inferencer{
	public Distribution ask( BayesianNetwork bn,RandomVariable x, Assignment e){
		System.out.println("Distribution of " + x.name + " given " + e.toString());

		Distribution q = new Distribution (x);
		for (Object xi: x.getDomain()){
			Assignment exi = e.copy();
			exi.set(x, xi);
			q.put(xi,enumerateAll(bn, bn.getVariableListTopologicallySorted(), exi));
		}
		q.normalize();
		return q;
	}
	
	public double enumerateAll(BayesianNetwork bn, List <RandomVariable> vars, Assignment e){
		if (vars.isEmpty()) return 1;
		RandomVariable Y = vars.get(0);
		if(e.containsKey(Y)) return bn.getProb(Y, e)*enumerateAll(bn, vars.subList(1, vars.size()), e);
		else{
			double p = 0;
			for (Object yi: Y.getDomain()){
				Assignment exi = e.copy();
				exi.set(Y, yi);
				p+=bn.getProb(Y,exi) * enumerateAll(bn, vars.subList(1, vars.size()), exi);
			}
			return p;
		}	
	}
	
	
	public Assignment prior_sample(BayesianNetwork bn){
		Assignment x = new Assignment();
		List<RandomVariable> vars = bn.getVariableListTopologicallySorted();
		for(RandomVariable rvar : vars) {				//assign a value to every random variable based on its prior probability
			double probT = 0;
			double numba = Math.random();
			for (int i = 0; i<rvar.getDomain().size()-1; i++) {
				x.put(rvar, rvar.getDomain().get(i));
				probT += bn.getProb(rvar, x);
				if(numba > probT) {
					x.set(rvar, rvar.getDomain().get(i+1));
				}else {
					x.set(rvar, rvar.getDomain().get(i));
				}
			}
		}
		return x;
	}
	
	public Distribution rejection_sampling(RandomVariable X, Assignment e, BayesianNetwork bn, int n) {
		System.out.println("Distribution of " + X.name + " given " + e.toString());
		Distribution d = new Distribution(X.domain.size());
		for(Object i: X.domain) {
			d.put(i, 0);			 						//initialize distribution to be 0 for each value
		}
		for(int i = 1; i<n ; i++) {
			boolean consistent = true;
			Assignment a = prior_sample(bn);				//get a prior sample and check if it is consistent
			for(Object o : e.variableSet()) {
				if(a.get(o).equals(e.get(o)) != true){
					consistent = false;					//if it is not consistent, throw it out
					break;
				}
			}
			if(consistent) {								// if it is consistent, add 1 to the distribution 
				d.put(a.get(X), d.get(a.get(X))+1);		//for the corresponding value of X in the assignment
			}
		}
		for(Object i : d.keySet()) {						
			if(d.get(i) != 0) {							//if there is a non-zero value in distribution, normalize and return
				d.normalize();
				return d;
			}
		}
		return d;
	}
	
	
	public WeightedAssignment weighted_sample(BayesianNetwork bn, Assignment e){
		double w = 1.0;
		Assignment x = e.copy();
		for(RandomVariable var : bn.getVariableListTopologicallySorted()) {
			//Assignment temp = e.copy();
			if(e.containsKey(var)) {
				double p = bn.getProb(var, x);
				w = w*p;
				
			}else {
				double probT = 0;
				double numba = Math.random();
				for (int i = 0; i < var.getDomain().size()-1; i++) {
					x.put(var, var.getDomain().get(i));
					probT += bn.getProb(var, x);
					if(numba > probT) {
						x.put(var, var.getDomain().get(i+1));
					}else {
						x.put(var, var.getDomain().get(i));
					}
				}
			}
		}
		WeightedAssignment wx = new WeightedAssignment(w, x);
		return wx;
	}
	
	public Distribution likelihood_weighting(RandomVariable X, Assignment e, BayesianNetwork bn, int n) {
		System.out.println("Distribution of " + X.name + " given " + e.toString());
		Distribution d = new Distribution(X.domain.size());
		for(Object i: X.domain) {
			d.put(i, 0);			 						//initialize distribution to be 0 for each value
		}
		for(int i = 1; i<n;i++) {
			WeightedAssignment wx = weighted_sample(bn, e);
			d.put(wx.x.get(X), d.get(wx.x.get(X))+ wx.weight);
		}
		for(Object i : d.keySet()) {						
			if(d.get(i) != 0) {							//if there is a non-zero value in distribution, normalize and return
				d.normalize();
				return d;
			}
		}
		return d;
	}
	

	
}