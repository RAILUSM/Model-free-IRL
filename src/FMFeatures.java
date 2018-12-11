/**
 * This class implements the binary features of the FreeWay Merging domain.
 * @author Vinamra Jain, Bikramjit Banerjee
 */


public class FMFeatures extends AbstractFeatures{
	
	public FMFeatures (int stateSpace, int actionSpace, int features){
        super(stateSpace, actionSpace, features);		
	}
	
	/**
	 * Returns the feature vector for a corresponding state-action pair.
	 * 
	 * @param s state
	 * @param a action
	 * @return feature vector
	 */
	public double[] featureVector(int s, int a) {

		double [] fv = new double[this.numfeatures];

		if (((s >= 500 && s <= 624) || (s >=1125 && s<= 1249)) && (a >= 3)){
			fv[0] = 0;
		}else{
			fv[0] = 1;
		}
		
		if (a > 1 ){
			fv[1] = 1; 
		}else {
			fv[1] = 0;
		}
		
		if ((s > 624) && (a >= 3)) {
			fv[2] = 1;
		}else {
			fv[2] = 0;			
		}
		

		return fv;
	}
}
