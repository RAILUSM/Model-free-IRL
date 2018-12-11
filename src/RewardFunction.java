/**
 * This class manages the reward function parameters for the IRL. Reward function 
 * is expressed as weighted sum of the features. 
 * 
 * @author Vinamra Jain
 *
 */

public class RewardFunction {

	AbstractFeatures SAFeatures;
	int numParameters;
	double[] parameters;
	double[][] rewardMatrix;
	int stateSpace, actionSpace;
	
	public RewardFunction(AbstractFeatures features){
		
		this.SAFeatures = features;
		this.stateSpace = features.stateSpace;
		this.actionSpace = features.actionSpace;
		this.numParameters = features.numfeatures;
		this.parameters = new double[numParameters];
		this.rewardMatrix = new double[features.stateSpace][features.actionSpace];
	}
	
	public void setParameters(int index, double value){
		
		this.parameters[index] = value; 
		
	}
	
	/**
	 * 
	 * Generates the reward matrix for all state-action pairs.
	 * 
	 */
	
	public void generateRewardMatrix(){
		double[] fv = new double[numParameters];
		double sum;
		for (int i = 0; i < this.stateSpace; i++){
			for (int j = 0; j < this.actionSpace; j++){
				
				fv = SAFeatures.featureVector(i, j);
				sum = 0;
				for(int k = 0; k < numParameters; k++){
				sum += parameters[k]*fv[k];  
				}
				this.rewardMatrix[i][j] = sum;
			}
		}
		
	}
	
	public double[][] getRewardMatrix(){
		return this.rewardMatrix;
	}
	
	
}
