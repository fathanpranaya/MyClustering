/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    SimpleKMeans.java
 *    Copyright (C) 2000-2012 University of Waikato, Hamilton, New Zealand
 *
 */


import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.classifiers.rules.DecisionTableHashKey;
import weka.clusterers.Canopy;
import weka.clusterers.FarthestFirst;
import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

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

	protected ReplaceMissingValues m_ReplaceMissingFilter;
	protected int m_NumClusters = 2; //parameter
	protected Instances m_initialStartPoints;
	protected Instances m_ClusterCentroids;
	protected int[][][] m_ClusterNominalCounts;
	protected double[] m_FullMeansOrMediansOrModes;
	protected int[][] m_FullNominalCounts;
	protected int[] m_ClusterSizes;
	protected int m_MaxIterations = 500; //parameter
	protected int m_Iterations = 0;
	protected double[] m_squaredErrors;
	protected DistanceFunction m_DistanceFunction = new EuclideanDistance();



	public static final int FARTHEST_FIRST = 3;
	protected int m_initializationMethod = FARTHEST_FIRST;

	protected double m_minClusterDensity = 2;

	public MyKMeans() {
		super();

		m_SeedDefault = 10;
		setSeed(m_SeedDefault);
	}

	@Override
	public void buildClusterer(Instances data) throws Exception {
		m_Iterations = 0;

		m_ReplaceMissingFilter = new ReplaceMissingValues();
		Instances instances = new Instances(data);

		instances.setClassIndex(-1);
		m_FullNominalCounts = new int[instances.numAttributes()][0];

		m_FullMeansOrMediansOrModes = moveCentroid(0, instances, false, false);
		for (int i = 0; i < instances.numAttributes(); i++) {
			m_FullNominalCounts[i] = instances.attributeStats(i).nominalCounts;
		}

		m_ClusterCentroids = new Instances(instances, m_NumClusters);
		int[] clusterAssignments = new int[instances.numInstances()];

		m_DistanceFunction.setInstances(instances);

		Instances initInstances = null;
		initInstances = instances;

		//FarthestFirst init
		farthestFirstInit(initInstances);
		m_initialStartPoints = new Instances(m_ClusterCentroids);

		m_NumClusters = m_ClusterCentroids.numInstances();

		// removing reference
		initInstances = null;

		int i;
		boolean converged = false;
		int emptyClusterCount;
		Instances[] tempI = new Instances[m_NumClusters];
		m_squaredErrors = new double[m_NumClusters];
		m_ClusterNominalCounts = new int[m_NumClusters][instances.numAttributes()][0];
		//startExecutorPool();

		while (!converged) {


			emptyClusterCount = 0;
			m_Iterations++;
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
			m_ClusterCentroids = new Instances(instances, m_NumClusters);
			for (i = 0; i < m_NumClusters; i++) {
				tempI[i] = new Instances(instances, 0);
			}
			for (i = 0; i < instances.numInstances(); i++) {
				tempI[clusterAssignments[i]].add(instances.instance(i));
			}

			for (i = 0; i < m_NumClusters; i++) {
				if (tempI[i].numInstances() == 0) {
					// empty cluster
					emptyClusterCount++;
				} else {
					moveCentroid(i, tempI[i], true, true);
				}
			}

			if (m_Iterations == m_MaxIterations) {
				converged = true;
			}

			if (emptyClusterCount > 0) {
				m_NumClusters -= emptyClusterCount;
				if (converged) {
					Instances[] t = new Instances[m_NumClusters];
					int index = 0;
					for (int k = 0; k < tempI.length; k++) {
						if (tempI[k].numInstances() > 0) {
							t[index] = tempI[k];

							for (i = 0; i < tempI[k].numAttributes(); i++) {
								m_ClusterNominalCounts[index][i] = m_ClusterNominalCounts[k][i];
							}
							index++;
						}
					}
					tempI = t;
				} else {
					tempI = new Instances[m_NumClusters];
				}
			}

			if (!converged) {
				m_ClusterNominalCounts = new int[m_NumClusters][instances
				                                                .numAttributes()][0];
			}
		}

		m_ClusterSizes = new int[m_NumClusters];
		for (i = 0; i < m_NumClusters; i++) {
			m_ClusterSizes[i] = tempI[i].numInstances();
		}

		// save memory!
		m_DistanceFunction.clean();
	}

	protected void farthestFirstInit(Instances data) throws Exception {
		FarthestFirst ff = new FarthestFirst();
		ff.setNumClusters(m_NumClusters);
		ff.buildClusterer(data);

		m_ClusterCentroids = ff.getClusterCentroids();
	}

	protected double[] moveCentroid(int centroidIndex, Instances members,
			boolean updateClusterInfo, boolean addToCentroidInstances) {
		double[] vals = new double[members.numAttributes()];

		// used only for Manhattan Distance
		Instances sortedMembers = null;
		int middle = 0;
		boolean dataIsEven = false;

		if (m_DistanceFunction instanceof ManhattanDistance) {
			middle = (members.numInstances() - 1) / 2;
			dataIsEven = ((members.numInstances() % 2) == 0);
			sortedMembers = new Instances(members);
		}

		for (int j = 0; j < members.numAttributes(); j++) {
			if (m_DistanceFunction instanceof EuclideanDistance
					|| members.attribute(j).isNominal()) {
				vals[j] = members.meanOrMode(j);
			} else if (m_DistanceFunction instanceof ManhattanDistance) {
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
				m_ClusterNominalCounts[centroidIndex][j] = members.attributeStats(j).nominalCounts;
			}
		}
		if (addToCentroidInstances) {
			m_ClusterCentroids.add(new DenseInstance(1.0, vals));
		}
		return vals;
	}

	private int clusterProcessedInstance(Instance instance, boolean updateErrors,
			boolean useFastDistCalc, long[] instanceCanopies) {
		double minDist = Integer.MAX_VALUE;
		int bestCluster = 0;
		for (int i = 0; i < m_NumClusters; i++) {
			double dist;
			if (useFastDistCalc) {


				dist = m_DistanceFunction.distance(instance,
						m_ClusterCentroids.instance(i), minDist);

			} else {
				dist = m_DistanceFunction.distance(instance,
						m_ClusterCentroids.instance(i));
			}
			if (dist < minDist) {
				minDist = dist;
				bestCluster = i;
			}
		}
		if (updateErrors) {
			if (m_DistanceFunction instanceof EuclideanDistance) {
				// Euclidean distance to Squared Euclidean distance
				minDist *= minDist;
			}
			m_squaredErrors[bestCluster] += minDist;
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
		return m_NumClusters;
	}

	public void setNumClusters(int n) throws Exception {
		if (n <= 0) {
			throw new Exception("Number of clusters must be > 0");
		}
		m_NumClusters = n;
	}

	public void setMaxIterations(int n) throws Exception {
		if (n <= 0) {
			throw new Exception("Maximum number of iterations must be > 0");
		}
		m_MaxIterations = n;
	}

	public int getMaxIterations() {
		return m_MaxIterations;
	}

	@Override
	public String toString() {
		if (m_ClusterCentroids == null) {
			return "No clusterer built yet!";
		}

		int maxWidth = 0;
		int maxAttWidth = 0;
		for (int i = 0; i < m_NumClusters; i++) {
			for (int j = 0; j < m_ClusterCentroids.numAttributes(); j++) {
				if (m_ClusterCentroids.attribute(j).name().length() > maxAttWidth) {
					maxAttWidth = m_ClusterCentroids.attribute(j).name().length();
				}
			}
		}

		for (int i = 0; i < m_ClusterCentroids.numAttributes(); i++) {
			if (m_ClusterCentroids.attribute(i).isNominal()) {
				Attribute a = m_ClusterCentroids.attribute(i);
				for (int j = 0; j < m_ClusterCentroids.numInstances(); j++) {
					String val = a.value((int) m_ClusterCentroids.instance(j).value(i));
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
		for (int m_ClusterSize : m_ClusterSizes) {
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
		temp.append("\nNumber of iterations: " + m_Iterations);

		temp.append("\n\nInitial staring points (");

		temp.append("farthest first");
		temp.append("):\n");
		temp.append("\n\nFinal cluster centroids:\n");
		temp.append(pad("Cluster#", " ", (maxAttWidth + (maxWidth * 2 + 2))
				- "Cluster#".length(), true));

		temp.append("\n");
		temp
		.append(pad("Attribute", " ", maxAttWidth - "Attribute".length(), false));

		temp
		.append(pad("Full Data", " ", maxWidth + 1 - "Full Data".length(), true));

		// cluster numbers
		for (int i = 0; i < m_NumClusters; i++) {
			String clustNum = "" + i;
			temp.append(pad(clustNum, " ", maxWidth + 1 - clustNum.length(), true));
		}
		temp.append("\n");

		// cluster sizes
		String cSize = "(" + Utils.sum(m_ClusterSizes) + ")";
		temp.append(pad(cSize, " ", maxAttWidth + maxWidth + 1 - cSize.length(),
				true));
		for (int i = 0; i < m_NumClusters; i++) {
			cSize = "(" + m_ClusterSizes[i] + ")";
			temp.append(pad(cSize, " ", maxWidth + 1 - cSize.length(), true));
		}
		temp.append("\n");

		temp.append(pad("", "=",
				maxAttWidth
				+ (maxWidth * (m_ClusterCentroids.numInstances() + 1)
						+ m_ClusterCentroids.numInstances() + 1), true));
		temp.append("\n");

		for (int i = 0; i < m_ClusterCentroids.numAttributes(); i++) {
			String attName = m_ClusterCentroids.attribute(i).name();
			temp.append(attName);
			for (int j = 0; j < maxAttWidth - attName.length(); j++) {
				temp.append(" ");
			}

			String strVal;
			String valMeanMode;
			// full data
			valMeanMode = pad(
					(strVal = m_ClusterCentroids.attribute(i).value(
							(int) m_FullMeansOrMediansOrModes[i])), " ", maxWidth + 1
							- strVal.length(), true);
			temp.append(valMeanMode);

			for (int j = 0; j < m_NumClusters; j++) {
				valMeanMode = pad(
						(strVal = m_ClusterCentroids.attribute(i).value(
								(int) m_ClusterCentroids.instance(j).value(i))), " ", maxWidth
								+ 1 - strVal.length(), true);
				temp.append(valMeanMode);
			}
			temp.append("\n");
		}

		temp.append("\n\n");
		return temp.toString();
	}

	private String pad(String source, String padChar, int length, boolean leftPad) {
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
		return m_ClusterCentroids;
	}

	public int[][][] getClusterNominalCounts() {
		return m_ClusterNominalCounts;
	}

	public double getSquaredError() {
		return Utils.sum(m_squaredErrors);
	}

	public int[] getClusterSizes() {
		return m_ClusterSizes;
	}

	public static void main(String[] args) {
		runClusterer(new MyKMeans(), args);
	}

}

