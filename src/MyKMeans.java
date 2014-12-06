import weka.clusterers.FarthestFirst;
import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

/**
 <!-- options-end -->
 * 
 * @author Fathan Adi Pranaya 13511027
 * @author Sonny Lazuardi Hermawan 13511029
 * @author Aldi Doanta Kurnia 13511031
 */
public class MyKMeans extends RandomizableClusterer implements
NumberOfClustersRequestable, WeightedInstancesHandler {

	/** for serialization. */
	static final long serialVersionUID = -3235809600124455376L;

	protected int n_cluster = 2; //parameter
	protected Instances initial_points;
	protected Instances centroids;
	protected int[][][] n_nominal;
	protected double[] modes;
	protected int[][] full_nominal;
	protected int[] cluster_size;
	protected int max_iterations = 500; //parameter
	protected int iterations = 0;
	protected double[] mse;
	protected DistanceFunction distance_function = new EuclideanDistance();
	
	protected int method = 3;

	protected double min_density = 2;

	public MyKMeans() {
		super();
		m_SeedDefault = 10;
		setSeed(m_SeedDefault);
	}

	@Override
	public void buildClusterer(Instances data) throws Exception {
		iterations = 0;
		Instances instances = new Instances(data);

		instances.setClassIndex(-1);
		full_nominal = new int[instances.numAttributes()][0];

		modes = moveCentroid(0, instances, false, false);
		for (int i = 0; i < instances.numAttributes(); i++) {
			full_nominal[i] = instances.attributeStats(i).nominalCounts;
		}

		centroids = new Instances(instances, n_cluster);
		int[] clusterAssignments = new int[instances.numInstances()];

		distance_function.setInstances(instances);

		Instances initInstances = null;
		initInstances = instances;

		//FarthestFirst init
		farthestFirstInit(initInstances);
		initial_points = new Instances(centroids);

		n_cluster = centroids.numInstances();

		// removing reference
		initInstances = null;

		int i;
		boolean converged = false;
		int emptyClusterCount;
		Instances[] tempI = new Instances[n_cluster];
		mse = new double[n_cluster];
		n_nominal = new int[n_cluster][instances.numAttributes()][0];
		//startExecutorPool();

		while (!converged) {
			emptyClusterCount = 0;
			iterations++;
			converged = true;

			for (i = 0; i < instances.numInstances(); i++) {
				Instance toCluster = instances.instance(i);
				int newC = clusterProcessedInstance(
						toCluster,
						false,
						true,
						null);
				if (newC != clusterAssignments[i]) {
					converged = false;
				}
				clusterAssignments[i] = newC;
			}

			// update centroids
			centroids = new Instances(instances, n_cluster);
			for (i = 0; i < n_cluster; i++) {
				tempI[i] = new Instances(instances, 0);
			}
			for (i = 0; i < instances.numInstances(); i++) {
				tempI[clusterAssignments[i]].add(instances.instance(i));
			}

			for (i = 0; i < n_cluster; i++) {
				if (tempI[i].numInstances() == 0) {
					// empty cluster
					emptyClusterCount++;
				} else {
					moveCentroid(i, tempI[i], true, true);
				}
			}

			if (iterations == max_iterations) {
				converged = true;
			}

			if (emptyClusterCount > 0) {
				n_cluster -= emptyClusterCount;
				if (converged) {
					Instances[] t = new Instances[n_cluster];
					int index = 0;
					for (int k = 0; k < tempI.length; k++) {
						if (tempI[k].numInstances() > 0) {
							t[index] = tempI[k];

							for (i = 0; i < tempI[k].numAttributes(); i++) {
								n_nominal[index][i] = n_nominal[k][i];
							}
							index++;
						}
					}
					tempI = t;
				} else {
					tempI = new Instances[n_cluster];
				}
			}

			if (!converged) {
				n_nominal = new int[n_cluster][instances
				                                                .numAttributes()][0];
			}
		}

		cluster_size = new int[n_cluster];
		for (i = 0; i < n_cluster; i++) {
			cluster_size[i] = tempI[i].numInstances();
		}

		// save memory!
		distance_function.clean();
	}

	protected void farthestFirstInit(Instances data) throws Exception {
		FarthestFirst ff = new FarthestFirst();
		ff.setNumClusters(n_cluster);
		ff.buildClusterer(data);

		centroids = ff.getClusterCentroids();
	}

	protected double[] moveCentroid(int centroidIndex, Instances members,
			boolean updateClusterInfo, boolean addToCentroidInstances) {
		double[] vals = new double[members.numAttributes()];

		// used only for Manhattan Distance
		Instances sortedMembers = null;
		int middle = 0;
		boolean dataIsEven = false;

		if (distance_function instanceof ManhattanDistance) {
			middle = (members.numInstances() - 1) / 2;
			dataIsEven = ((members.numInstances() % 2) == 0);
			sortedMembers = new Instances(members);
		}

		for (int j = 0; j < members.numAttributes(); j++) {
			if (distance_function instanceof EuclideanDistance
					|| members.attribute(j).isNominal()) {
				vals[j] = members.meanOrMode(j);
			} else if (distance_function instanceof ManhattanDistance) {
				// singleton special case
				if (members.numInstances() == 1) {
					vals[j] = members.instance(0).value(j);
				} else {
					vals[j] = sortedMembers.kthSmallestValue(j, middle + 1);
					if (dataIsEven) {
						vals[j] = (vals[j] + sortedMembers.kthSmallestValue(j, middle + 2)) / 2;
					}
				}
			}

			if (updateClusterInfo) {
				n_nominal[centroidIndex][j] = members.attributeStats(j).nominalCounts;
			}
		}
		if (addToCentroidInstances) {
			centroids.add(new DenseInstance(1.0, vals));
		}
		return vals;
	}

	private int clusterProcessedInstance(Instance instance, boolean updateErrors,
			boolean useFastDistCalc, long[] instanceCanopies) {
		double minDist = Integer.MAX_VALUE;
		int bestCluster = 0;
		for (int i = 0; i < n_cluster; i++) {
			double dist;
			if (useFastDistCalc) {


				dist = distance_function.distance(instance,
						centroids.instance(i), minDist);

			} else {
				dist = distance_function.distance(instance,
						centroids.instance(i));
			}
			if (dist < minDist) {
				minDist = dist;
				bestCluster = i;
			}
		}
		if (updateErrors) {
			if (distance_function instanceof EuclideanDistance) {
				// Euclidean distance to Squared Euclidean distance
				minDist *= minDist;
			}
			mse[bestCluster] += minDist;
		}
		return bestCluster;
	}

	@Override
	public int clusterInstance(Instance instance) throws Exception {
		Instance inst = null;
		inst = instance;
		return clusterProcessedInstance(inst, false, true, null);
	}

	@Override
	public int numberOfClusters() throws Exception {
		return n_cluster;
	}

	public void setNumClusters(int n) throws Exception {
		if (n <= 0) {
			throw new Exception("Number of clusters must be > 0");
		}
		n_cluster = n;
	}

	public void setMaxIterations(int n) throws Exception {
		if (n <= 0) {
			throw new Exception("Maximum number of iterations must be > 0");
		}
		max_iterations = n;
	}

	public int getMaxIterations() {
		return max_iterations;
	}

	@Override
	public String toString() {
		if (centroids == null) {
			return "No clusterer built yet!";
		}

		int maxWidth = 0;
		int maxAttWidth = 0;
		for (int i = 0; i < n_cluster; i++) {
			for (int j = 0; j < centroids.numAttributes(); j++) {
				if (centroids.attribute(j).name().length() > maxAttWidth) {
					maxAttWidth = centroids.attribute(j).name().length();
				}
			}
		}

		for (int i = 0; i < centroids.numAttributes(); i++) {
			if (centroids.attribute(i).isNominal()) {
				Attribute a = centroids.attribute(i);
				for (int j = 0; j < centroids.numInstances(); j++) {
					String val = a.value((int) centroids.instance(j).value(i));
					if (val.length() > maxWidth) {
						maxWidth = val.length();
					}
				}
				for (int j = 0; j < a.numValues(); j++) {
					String val = a.value(j) + " ";
					if (val.length() > maxAttWidth) {
						maxAttWidth = val.length();
					}
				}
			}
		}

		// check for size of cluster sizes
		for (int m_ClusterSize : cluster_size) {
			String size = "(" + m_ClusterSize + ")";
			if (size.length() > maxWidth) {
				maxWidth = size.length();
			}
		}

		maxAttWidth += 2;
		if (maxAttWidth < "Attribute".length() + 2) {
			maxAttWidth = "Attribute".length() + 2;
		}

		if (maxWidth < "Full Data".length()) {
			maxWidth = "Full Data".length() + 1;
		}

		if (maxWidth < "missing".length()) {
			maxWidth = "missing".length() + 1;
		}

		StringBuffer temp = new StringBuffer();
		temp.append("\nMy KMeans\n======\n");
		temp.append("\nNumber of iterations: " + iterations);

		temp.append("\n\nInitial staring points (");

		temp.append("farthest first");
		temp.append("):\n");
		temp.append("\n\nFinal cluster centroids:\n");
		temp.append(part("Cluster#", " ", (maxAttWidth + (maxWidth * 2 + 2))
				- "Cluster#".length(), true));

		temp.append("\n");
		temp
		.append(part("Attribute", " ", maxAttWidth - "Attribute".length(), false));

		temp
		.append(part("Full Data", " ", maxWidth + 1 - "Full Data".length(), true));

		// cluster numbers
		for (int i = 0; i < n_cluster; i++) {
			String clustNum = "" + i;
			temp.append(part(clustNum, " ", maxWidth + 1 - clustNum.length(), true));
		}
		temp.append("\n");

		// cluster sizes
		String cSize = "(" + Utils.sum(cluster_size) + ")";
		temp.append(part(cSize, " ", maxAttWidth + maxWidth + 1 - cSize.length(),
				true));
		for (int i = 0; i < n_cluster; i++) {
			cSize = "(" + cluster_size[i] + ")";
			temp.append(part(cSize, " ", maxWidth + 1 - cSize.length(), true));
		}
		temp.append("\n");

		temp.append(part("", "=",
				maxAttWidth
				+ (maxWidth * (centroids.numInstances() + 1)
						+ centroids.numInstances() + 1), true));
		temp.append("\n");

		for (int i = 0; i < centroids.numAttributes(); i++) {
			String attName = centroids.attribute(i).name();
			temp.append(attName);
			for (int j = 0; j < maxAttWidth - attName.length(); j++) {
				temp.append(" ");
			}

			String strVal;
			String valMeanMode;
			// full data
			valMeanMode = part(
					(strVal = centroids.attribute(i).value(
							(int) modes[i])), " ", maxWidth + 1
							- strVal.length(), true);
			temp.append(valMeanMode);

			for (int j = 0; j < n_cluster; j++) {
				valMeanMode = part(
						(strVal = centroids.attribute(i).value(
								(int) centroids.instance(j).value(i))), " ", maxWidth
								+ 1 - strVal.length(), true);
				temp.append(valMeanMode);
			}
			temp.append("\n");
		}

		temp.append("\n\n");
		return temp.toString();
	}

	private String part(String source, String padChar, int length, boolean leftPad) {
		StringBuffer temp = new StringBuffer();

		if (leftPad) {
			for (int i = 0; i < length; i++) {
				temp.append(padChar);
			}
			temp.append(source);
		} else {
			temp.append(source);
			for (int i = 0; i < length; i++) {
				temp.append(padChar);
			}
		}
		return temp.toString();
	}

	public Instances getClusterCentroids() {
		return centroids;
	}

	public int[][][] getClusterNominalCounts() {
		return n_nominal;
	}

	public double getSquaredError() {
		return Utils.sum(mse);
	}

	public int[] getClusterSizes() {
		return cluster_size;
	}
}

