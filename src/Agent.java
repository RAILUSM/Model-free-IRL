/**
 * This is an abstract agent class, to be extended by Q-averaging and 
 * Q-softmax agents. These agents only differ in how they calculate the Q-value of 
 * the next state, and how they update Q-values and Q-gradients. The remaining 
 * functionalities are common and are implemented here.
 * 
 * @authors Vinamra Jain, Bikramjit Banerjee
 *
 */

import java.util.*;

abstract public class Agent 
{
    double[][] qValues;
	double[][][] qGradient;
	List<Trajectories> AllTJs;
	RewardFunction rf;
	int stateSpace, actionSpace;
	double gamma;
	double learningRate;
	double boltzmannBeta;
	Set<SAPair> unseenSA;
	
    abstract public double getNextQvalue(int sprime);
    abstract public void updateQnQGradValues(double d, double[][][] x);
    
    public Agent(List<Trajectories> AllTJs, RewardFunction rf, double gamma, double learningRate, double boltzmannBeta) {
        this.AllTJs = AllTJs;
		this.rf = rf;
		this.actionSpace =rf.actionSpace;
		this.stateSpace =rf.stateSpace;
		this.gamma = gamma;
		this.learningRate = learningRate;
		this.qValues = new double[stateSpace][actionSpace];
		this.qGradient = new double[rf.numParameters][stateSpace][actionSpace];
		this.boltzmannBeta = boltzmannBeta;
		this.unseenSA = unseenStateActions();
    }
    
    /**
	 * Set the reward function from externally supplied
	 * 
	 */
    public void setRf(RewardFunction rf){
		
		this.rf = rf;
	}

    /**
	 * initializing Q-values 
	 * 
	 */
	
	public void initQValues(){
		for (int i = 0; i < this.stateSpace; i++){
			for (int j = 0; j < this.actionSpace; j++){
				this.qValues[i][j] = 0.0;
			}
		}
	}

    /**
	 * Initializing Q-gradients
	 */
	
	public void initQGrad(){
        for (int i = 0; i < this.stateSpace; i++){
            for (int j = 0; j < this.actionSpace; j++){
                double[] fv = rf.SAFeatures.featureVector(i, j);
                for(int k = 0; k < rf.numParameters;k++){
					this.qGradient[k][i][j] = 0.0;
				}	
			}
		}
	}
	
    /**
     * Calculate pi_theta(s,a)
     */
    public double getActionProb(int s, int a){
		double sum = 0;
		double actProb = 0;
		for(int i = 0; i < actionSpace ; i++){
			
			sum += Math.exp(this.boltzmannBeta * qValues[s][i]);
			
		}
		
		actProb= Math.exp(this.boltzmannBeta * qValues[s][a])/sum;
			
		return actProb;
	}

    /**
     * Return a set of all (s,a) that are not on any trajectory; also include the last step of any trajectory.
     */
    public Set<SAPair> unseenStateActions() {
		Set<SAPair> ret_val = new HashSet<SAPair>();
		for(int s=0; s < stateSpace; s++) {
			for(int a=0; a<actionSpace; a++) {
				SAPair sa = new SAPair(s,a);
				ret_val.add(sa);
			}
		}
		for(int i = 0; i < AllTJs.size(); i++){
			Trajectories t1 = AllTJs.get(i);
			for(int k = 0; k < t1.stateSequence.size() - 1; k++){ //leave the last step as unseen so it is set to R_theta by QUpdate 
				int state = t1.stateSequence.get(k);
				int action = t1.actionSequence.get(k);
				SAPair sa = new SAPair(state, action);
				ret_val.remove(sa);
			}
		}
		return ret_val;
	}

    public void updateUnseenQvalues() {
        for(SAPair sa : unseenSA) {
            int st = sa.s;
            int act = sa.a;
            qValues[st][act] += this.learningRate * ( rf.rewardMatrix[st][act] - qValues[st][act] );
        }
    }
    
    public void updateUnseenQgradients() {
        for(SAPair sa : unseenSA) {
        	int st = sa.s;
            int act = sa.a;
            double[] fv = rf.SAFeatures.featureVector(st, act);
            for(int p = 0; p < rf.numParameters; p++) {
                qGradient[p][st][act] += this.learningRate * ( fv[p] - qGradient[p][st][act] );
            }
        }
    }
    
    /**
     * Update Q values; return max-change.
     */
    public double updateQvalues() {
        double maxchange = -1.0, diff, oldQvalue, newQvalue;
        int state,action,sprime;
        updateUnseenQvalues();
        for(int i = 0; i < AllTJs.size(); i++){
            Trajectories t1 = AllTJs.get(i);
            for(int k = 0; k < t1.stateSequence.size() - 1; k++){
                state = t1.stateSequence.get(k);
                action = t1.actionSequence.get(k);
                sprime = t1.stateSequence.get(k+1);
                oldQvalue = qValues[state][action];
                newQvalue = oldQvalue + this.learningRate*(rf.rewardMatrix[state][action] + this.gamma * getNextQvalue(sprime) - oldQvalue); 
                qValues[state][action] = newQvalue;
                diff = Math.abs( oldQvalue - newQvalue);
                if (diff > maxchange)
                    maxchange = diff;
            }
        }
        return maxchange;
    }

    /**
	 * @return all trajectories.
	 */
	
    public List<Trajectories> getTrajectories(){
		
		return this.AllTJs;
	}

}


    
    
