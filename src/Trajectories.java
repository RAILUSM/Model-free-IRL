import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * This class reads all the trajectories from a folder and stores as state-action pair in form of list.
 * 
 * @author Vinamra Jain
 *
 */


public class Trajectories {

	String pathName;	
	ArrayList<Integer> stateSequence;
	ArrayList<Integer> actionSequence;
	List<Trajectories> tj;
	public Trajectories(){
		
	}
	
	public Trajectories(String pathName){
		this.pathName = pathName;		
	}

	/**
	 * Read all trajectories from the folder.
	 * 
	 * @return list of all state-action pair lists.
	 * @throws IOException
	 */
	
	public List <Trajectories> readAllTrajectories() throws IOException{
		
		tj = new ArrayList<Trajectories>();
		File dir = new File(pathName);
		File[] listOfFiles = dir.listFiles();
		
		for (File traj : listOfFiles){

			if(traj.isFile()){
				
			Trajectories t = readTrajectory(traj);	
			
			tj.add(t);
			}			
		}
		
		return tj;
	}
	
	/**
	 * Reading one trajectory at a time.
	 * 
	 * @param trajectory1 the trajectory file to be read and stored in list.
	 * @return the state-action pair for one trajectory.
	 * @throws IOException
	 */
	public Trajectories readTrajectory(File trajectory1) throws IOException{
		
		Trajectories t1 = new Trajectories();
		
		BufferedReader br1 = new BufferedReader(new FileReader(trajectory1));
 		int lines = 0;
		String line1 = br1.readLine();
		
		while (line1 != null){
			lines++;
			line1 = br1.readLine();
		}
		br1.close();
		
		t1.stateSequence = new ArrayList<Integer>(lines);
		t1.actionSequence = new ArrayList<Integer>(lines);
		
		BufferedReader br = new BufferedReader(new FileReader(trajectory1));
		String line = br.readLine();
		while (line != null){
			String[] splitted = line.split("\t");
			
			t1.stateSequence.add(Integer.parseInt(splitted[0]));
			t1.actionSequence.add(Integer.parseInt(splitted[1]));
			
			line = br.readLine();
		}
		br.close();
		
			return t1;
			
	}
	
		
}
