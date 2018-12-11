/**
 * This class implements the binary features of the Grid World domain.
 * @authors Vinamra Jain, Bikramjit Banerjee
 */

public class GWFeatures extends AbstractFeatures {

	public GWFeatures (int stateSpace, int actionSpace, int features){
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

        int ns = s; //ignoring action; otherwise use ns = nextState(s,a);
        if (ns==0) { //top-left: green
            fv[1] = 1;
		}else{
			fv[1] = 0;
		}
        if (ns==4) { //top-right: yellow
            fv[0] = 1;
		}else{
			fv[0] = 0;
		}
        if (ns==20) { //bottom-left: red
            fv[2] = 1;
		}else{
			fv[2] = 0;
		}
        if (ns==24) { //bottom-right: pink
            fv[3] = 1;
		}else{
			fv[3] = 0;
		}
        if ((ns==1)||(ns==8)||(ns==11)||(ns==18)||(ns==21)) //blue
            fv[4] = 1;
        else
            fv[4] = 0;

		return fv;
	}

	/* The following functions are not needed for this project. They are used to generate data 
	 * for Babes VRoman's MLIRL code.
	 */
    public int nextState(int s, int a) {
        int x,y,nx,ny;
        x=s/5;
        y=s%5;
        nx=x;
        ny=y;
        if (a==0)
            ny=y-1;
        if (a==1)
            nx=x+1;
        if (a==2)
            ny=y+1;
        if (a==3)
            nx=x-1;
        
        if (nx<0)
            nx=0;
        if (nx>4)
            nx=4;
        if (ny<0)
            ny=0;
        if (ny>4)
            ny=4;
        
        int ns = nx*5+ny;
        return ns;
    }
    
	
	/* Output R matrix (FXS matrix) & Theta matrix (S.A X S) for Babes VRoman's 
	 * MLIRL code. 
	 */
	public static void main(String[] args) {
		GWFeatures gwf = new GWFeatures(25, 4, 5);
		for (int s = 0; s < gwf.stateSpace; s++) {
			for(int a = 0; a < gwf.actionSpace; a++) {
				int ns = gwf.nextState(s,a);
				String str=""; 
				for(int sp = 0; sp < gwf.stateSpace; sp++) {
					if (sp == ns) {
						str += "1,";
					}
					else {
						str += "0,";
					}
				}
				System.out.println(str);
			}
		}
		for (int f = 0; f < gwf.numfeatures; f++) {
			char[] str = new char[gwf.stateSpace];
			for(int s = 0; s < gwf.stateSpace; s++) {
				str[s] = '0';
			}
			for(int s = 0; s < gwf.stateSpace; s++) {
				for (int a = 0; a < gwf.actionSpace; a++) {
					int ns = gwf.nextState(s, a);
					double[] fv = gwf.featureVector(s, a);
					if (fv[f] == 1) {
						str[s] = '1';
					}
				}
			}
			String st = "";
			for (int s = 0; s < gwf.stateSpace; s++) {
				st += (str[s] + ",");
			}
			//System.out.println(st);
		}
	}
	
}
