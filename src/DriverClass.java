import java.io.IOException;
import java.util.List;
import java.util.ArrayList;

/**
 * This is the driver class for model-free Maximum likelihood IRL approach.
 * Will perform multiple independent parallel runs.
 * 
 * @authors Vinamra Jain, Bikramjit Banerjee
 *
 */

public class DriverClass extends Thread {

	int stateSpace = 25; //1250 for NGSIM
	int actionSpace = 4; //5 for NGSIM
    int numFeatures = 5; //3 for NGSIM
	String trajectoryPath;
	int runID;
	
	public void setParameters(String pn, int run_id) {
		this.trajectoryPath = pn;
		this.runID = run_id;
	}
	
	/**
	 *	The following method will initialize the parameters for IRL and call the method to perform IRL.
	 *  Currently set to run QAveraging on Grid World with no-noise trajectories.
	 * 
	 */
	public void run() {
        
		AbstractFeatures f = new GWFeatures(stateSpace, actionSpace, numFeatures);
		
		RewardFunction rf = new RewardFunction(f);
		
		Trajectories tj = new Trajectories(trajectoryPath);
		List<Trajectories> AllTJs = new ArrayList<Trajectories>();
		try {
			AllTJs = tj.readAllTrajectories();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		Agent QA = new QAveraging(AllTJs,rf,0.9,0.1, 0.01);
        //Agent QA = new QSoftMax(AllTJs,rf,0.9,0.1, 0.01);
				
		MFMLIRL irl = new MFMLIRL(this.runID, rf, QA, 0.1, 0.0001, 20000, 0.01);
		
		irl.performIRL();
	}
	
	/**
	 * The following is the main function of the project.	
	 */
	public static void main(String[] args) throws IOException{

        int num_processes = 10; // number of parallel runs
		DriverClass[] dc = new DriverClass[num_processes];
		String pathname = "data/gridworld/noise_0.0_expert_trajs";

		for (int i = 0; i < num_processes; i++) {
            
			dc[i] = new DriverClass();
			dc[i].setParameters(pathname, i);
			dc[i].start();
		}
	}
}
