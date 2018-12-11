/** 
 * Convenient state-action pair object. 
 * 
 * @author: Bikramjit Banerjee
 */
public class SAPair{
	public int s;
	public int a;
	public SAPair(int s, int a) {
		this.s = s;
		this.a = a;
	}
	@Override 
	public boolean equals(Object obj) {
		if (obj instanceof SAPair) {
			SAPair sa = (SAPair) obj;
			if ((this.s==sa.s) && (this.a==sa.a))
				return true;
			else
				return false;
		}
		else
			return false;
	}
	@Override
	public int hashCode() {
		int numActions = 5;
		return s*numActions + a;
	}
}
