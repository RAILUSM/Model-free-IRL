import java.util.List;

/**
 * Implements Q-Averaging. Calculates the optimal Q-values iteratively after each gradient update. 
 * Here we calculate Q-values by replacing the max operator with average operator in canocical 
 * Q-learning equation. 
 * 
 * @authors Vinamra Jain, Bikramjit Banerjee
 *
 */

public class QAveraging extends Agent {
    
	public QAveraging(List<Trajectories> AllTJs, RewardFunction rf, double gamma, double learningRate, double boltzmannBeta){
        super(AllTJs, rf, gamma, learningRate, boltzmannBeta);
	}
    
	/**
	 * Method to generate optimal Q- and QGradient-values for all the state-action pairs.
	 * 
	 * Does not use policyGradient
	 */
	public void updateQnQGradValues(double threshold, double[][][] policyGradient){
		int count = 0;
		double maxchange = 2*threshold;
		while ((count < 10000 ) && (maxchange > threshold)){
			count++;
			maxchange = updateQvalues();
		}
		//if (count==10000)
		//	System.out.println("Q maxed out");
        count = 0; maxchange = 2*threshold;
        while ((count < 10000 ) && (maxchange > threshold)){
			count++;
			maxchange = 0.0;
            updateUnseenQgradients();
            int state, action, sprime;
            double oldQGrad, newQGrad;
            for(int i = 0; i < AllTJs.size(); i++){
				Trajectories t1 = AllTJs.get(i);
				for(int k = 0; k < t1.stateSequence.size() - 1; k++){
					
					state = t1.stateSequence.get(k);
					action = t1.actionSequence.get(k);
					sprime = t1.stateSequence.get(k+1);
					
					double[] fv = rf.SAFeatures.featureVector(state, action);
					
					for (int j = 0; j < rf.numParameters;j++){
						oldQGrad = qGradient[j][state][action];			
						newQGrad = oldQGrad + this.learningRate*(fv[j] + this.gamma * avgQGrad(sprime, j) - oldQGrad); 
						qGradient[j][state][action] = newQGrad;
						double diff = Math.abs( oldQGrad - newQGrad);
						if (diff > maxchange)
							maxchange = diff;
							
					}
				}
			}
		}
		//if (count==10000)
		//	System.out.println("Q-grads maxed out");
	}
	
	
	/**
	 * Calculate averaged Q-value for state sprime. 
	 * 
	 * @param sprime 
	 * @return averaged Q-value.
	 */
	
	public double getNextQvalue(int sprime){
		
		double average = 0;
		double sum = 0;
		for(int i = 0; i < actionSpace ; i++){
			
			sum += qValues[sprime][i];
			
		}
		
		average =  sum / actionSpace;
		
		return average;
	}
	
	/**
	 * calculating average Q-gradient values.
	 * @param sprime state sprime
	 * @param parameter	gradient parameter
	 * @return average Q-gradient value.
	 */
	public double avgQGrad(int sprime,int parameter){
		double average = 0;
		double sum = 0;
		for(int i = 0; i < actionSpace ; i++){
			
			sum += qGradient[parameter][sprime][i];
			
		}
		
		average =  sum / actionSpace;
		
		return average;
		
	}

}
