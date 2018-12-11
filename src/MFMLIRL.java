import java.text.DecimalFormat;
import java.util.List;

/**
 * An implementation of model-free maximum likelihood IRL.
 * 
 * @authors Vinamra Jain, Bikramjit Banerjee
 *
 */

public class MFMLIRL {

	int runID;
	RewardFunction rf;
	
    Agent agent;
	
	double learningRate;
	
	double maxLikelihoodChange;
	
	int maxSteps;
	
	double bolzmannBeta;
	
	double[][][] policyGradient;
	
	public MFMLIRL(int runid, RewardFunction rf, Agent agent ,double learningRate, double maxLikelihoodChange, int maxSteps, double boltzmannBeta){
        
		this.runID = runid;
		this.rf=rf;
		this.agent = agent;
		this.learningRate = learningRate;
		this.maxLikelihoodChange = maxLikelihoodChange;
		this.maxSteps = maxSteps;
		this.bolzmannBeta=boltzmannBeta;
		this.policyGradient = new double[rf.numParameters][rf.stateSpace][rf.actionSpace];
	}
	
	/**
	 * Function to perform model-free maximum likelihood IRL iteratively until convergence.
	 * 
	 */
	
	public void performIRL(){
		
			double[] theta = new double[rf.numParameters];
			
			for(int k=0; k<rf.numParameters; k++) {
				theta[k] = 2*Math.random()-1; //Initial weights in [-1,1]
                rf.setParameters( k, theta[k] );
			}
			DecimalFormat nf = new DecimalFormat("#0.0000");
			int n=0, t=0;
			double L, Lprime=0, delta=1000000.0;
			rf.generateRewardMatrix();
			agent.setRf(rf);
			agent.initQValues();
			agent.initQGrad();
			int count=0;
			do {
				n += 1;
				L = Lprime;
				rf.generateRewardMatrix();
				agent.setRf(rf);
                
				agent.updateQnQGradValues(maxLikelihoodChange, policyGradient);
                
				Lprime = this.logLikelihood();
				double[] gradient = this.logLikelihoodGradient();
				for(int j=0; j<gradient.length;j++){

					double curVal = rf.parameters[j];
					double nexVal = curVal + this.learningRate*(gradient[j]);
					theta[j] = nexVal;

				}
				for(int k=0; k<rf.numParameters; k++) {
					rf.setParameters( k, theta[k] );
				}
				delta = Math.abs( L - Lprime );
                
				if (count ==0) {
					String out_str = runID + " (Initial): ";
					for (int l=0; l<rf.numParameters; l++)
						out_str += nf.format(rf.parameters[l]) + ",";
					out_str += " LL=" + nf.format(Lprime);
					System.out.println(out_str);
				}

				count++;
			} while ((delta > maxLikelihoodChange) && (count < maxSteps));
			String out_str = runID + " (Final): ";
			for (int l=0; l<rf.numParameters; l++)
				out_str += nf.format(rf.parameters[l]) + ",";
			out_str += n+" iterations, LL="+nf.format(Lprime);
			System.out.println(out_str);
	}
    
	
	/**
	 * Method to calculate log-likelihood of expert's trajectories
	 * @return log-likelihood of trajectories
	 */
	
	public double logLikelihood(){

		List<Trajectories> exampleTrajectories = this.agent.getTrajectories();

		double sum = 0.;
		for(int i = 0; i < exampleTrajectories.size(); i++){
			sum += this.logLikelihoodOfTrajectory(exampleTrajectories.get(i));
		}
		
		return sum;

	}
	
	/**
	 * This method calculates the log-likelihood of state-action pairs in a single trajectory.
	 * @param ea is each trajectory.
	 * @return log-likelihood of single trajectory.
	 */
	
	public double logLikelihoodOfTrajectory(Trajectories ea){
		double logLike = 0.;
		int state,action;
		for(int i = 0; i < ea.actionSequence.size(); i++){
			state = ea.stateSequence.get(i);
			action = ea.actionSequence.get(i);
					
			double actProb = agent.getActionProb(state,action);
			
			logLike += Math.log(actProb);
		}
		return logLike;
	}
	
	/**
	 * Calculates log-likelihood gradient.
	 *  
	 * @return gradient value vector.
	 */
	
	public double[] logLikelihoodGradient(){
	
		double[] logGrad = new double[rf.numParameters];
		int state,action;
		
		List<Trajectories> exampleTrajectories = this.agent.getTrajectories();
		
		for(int j =0;j<rf.numParameters;j++){
			double gradValue =0;
			for(int i = 0; i < exampleTrajectories.size(); i++){
				Trajectories t1 = exampleTrajectories.get(i);
				for(int k = 0; k < t1.stateSequence.size(); k++){
					
					state = t1.stateSequence.get(k);
					action = t1.actionSequence.get(k);	
					
					gradValue += getGradValue(j,state,action);					
				}
			}
			logGrad[j]=gradValue;
		}
		return getNormalizedGradient( logGrad );
	}
	
	/**
	 * Calculating gradient for each state-action pair with respect to corresponding gradient parameter.
	 * @param p gradient parameter
	 * @param s state
	 * @param a action
	 * @return gradient value
	 */
	
	public double getGradValue(int p, int s, int a){
		
		double z = 0;
		double zGrad = 0;
		for(int i = 0; i < agent.actionSpace ; i++){
			
			z += Math.exp(this.bolzmannBeta*agent.qValues[s][i]);
		}
		double numeratorPartOne = z*this.bolzmannBeta*Math.exp(this.bolzmannBeta*agent.qValues[s][a])*agent.qGradient[p][s][a];
		
		for(int i = 0; i < agent.actionSpace ; i++){
			
			zGrad += this.bolzmannBeta*Math.exp(this.bolzmannBeta*agent.qValues[s][i])*agent.qGradient[p][s][i];
			
		}
		
		double numeratorPartTwo = Math.exp(this.bolzmannBeta*agent.qValues[s][a])*zGrad;
        
		policyGradient[p][s][a] = (numeratorPartOne - numeratorPartTwo)/z*z;
		
		double gradVal = (policyGradient[p][s][a] * z) / Math.exp(this.bolzmannBeta*agent.qValues[s][a]);
		
		return gradVal;
	}
	
	/**
	 * Calculating the normalized gradient values
	 * @param grad gradient values vector
	 * @return normalized gradient value vector.
	 */
	public double[] getNormalizedGradient(double[] grad){
		
		double[] normGrad = new double[grad.length];
		double gradientSquareSum = 0;
		
		for(int i=0;i<grad.length;i++){
			gradientSquareSum += Math.pow(grad[i], 2) ; 	
		}
		
		double gradientMagnitude = Math.sqrt(gradientSquareSum);
		
		for(int i=0;i<grad.length;i++){
			normGrad[i] = grad[i] / gradientMagnitude;	
		}
						
		return normGrad;
		
	}
	
}
