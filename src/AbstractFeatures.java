/**
 * This is an abstract feature class, to be extended by problem dependent
 * feature classes.
 * 
 * @author Bikramjit Banerjee
 *
 */

abstract public class AbstractFeatures 
{
    int stateSpace;
	int actionSpace;
	int numfeatures;
    
    public AbstractFeatures(int stateSpace, int actionSpace, int numfeatures){
		
		this.actionSpace = actionSpace;
		this.stateSpace = stateSpace;
		this.numfeatures = numfeatures;
		
	}
    
    abstract public double[] featureVector(int s, int a);
}
