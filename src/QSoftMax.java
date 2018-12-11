import java.util.List;

/**
 * Implements Q-softmax. Calculates the optimal Q-values iteratively along with gradient update. 
 * Here we calculate Q-values by replacing the max operator with softmax operator in canocical 
 * Q-learning equation. 
 * 
 * @author Bikramjit Banerjee
 *
 */

public class QSoftMax extends Agent{
	
	public QSoftMax(List<Trajectories> AllTJs, RewardFunction rf, double gamma, double learningRate, double boltzmannBeta){
        super(AllTJs, rf, gamma, learningRate, boltzmannBeta);
	}
	
	/**
	 * Method to generate optimal Q- and QGradient-values for all the state-action pairs.
	 * 
	 */
	public void updateQnQGradValues(double threshold, double[][][] policyGradient){
		
		int state,action,sprime,count = 0;
		double newQGradient, oldQGradient, maxchange = 2*threshold;
		while ((count < 10000 ) && (maxchange > threshold)){
			count++;
            updateUnseenQgradients();
			maxchange = updateQvalues();
            
            for(int i = 0; i < AllTJs.size(); i++){
				Trajectories t1 = AllTJs.get(i);
				for(int k = 0; k < t1.stateSequence.size() - 1; k++){
                    state = t1.stateSequence.get(k);
                    action = t1.actionSequence.get(k);
                    sprime = t1.stateSequence.get(k+1);
                    double[] fv = rf.SAFeatures.featureVector(state, action);
                    for (int j = 0; j < rf.numParameters;j++){
                        oldQGradient = qGradient[j][state][action];
                        double lastSum = 0.0;
                        for (int act = 0; act < actionSpace; act++){
                            lastSum += (getActionProb(sprime, act) * qGradient[j][sprime][act]);
                            lastSum += ( qValues[sprime][act] * policyGradient[j][sprime][act]);
                        }
                        newQGradient = oldQGradient + this.learningRate*(fv[j] + this.gamma * lastSum - oldQGradient); 
                        qGradient[j][state][action] = newQGradient;
                        double diff = Math.abs( oldQGradient - newQGradient);
						if (diff > maxchange)
							maxchange = diff;
                    }
				}
			}
		}
		//System.out.println("maxed count");
	}
	
	/**
	 * Calculate softmax Q-value for state sprime. 
	 * 
	 * @param sprime 
	 * @return softmax Q-value.
	 */
	
	public double getNextQvalue(int sprime){
		double sum = 0;
		for(int i = 0; i < actionSpace ; i++){
			
			sum += getActionProb(sprime, i) * qValues[sprime][i];
			
		}
		return sum;
	}
}

