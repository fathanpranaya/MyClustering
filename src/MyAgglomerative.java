import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.PriorityQueue;
import java.util.Vector;

import weka.clusterers.AbstractClusterer;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.CapabilitiesHandler;
import weka.core.DistanceFunction;
import weka.core.Drawable;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;

/**
 * <!-- options-end -->
 * 
 * @author Fathan Adi Pranaya 13511027
 * @author Sonny Lazuardi Hermawan 13511029
 * @author Aldi Doanta Kurnia 13511031
 */
public class MyAgglomerative extends AbstractClusterer implements
        OptionHandler, CapabilitiesHandler, Drawable {

    /** training data **/
    Instances my_instances;

    /** jumlah clusters dalam clustering **/
    int number_cluster = 2;

    /** distance function untuk membandingkan anggota dari cluster **/
    protected DistanceFunction m_DistanceFunction = new EuclideanDistance();

    public DistanceFunction getDistanceFunction() {
        return m_DistanceFunction;
    }

    public void setDistanceFunction(DistanceFunction distanceFunction) {
        m_DistanceFunction = distanceFunction;
    }

    class ClusterSet implements Comparable<ClusterSet>{
        public ClusterSet(double d, int i, int j, int nSize1, int nSize2) {
            jarak = d;
            cluster1 = i;
            cluster2 = j;
            cluster_size1 = nSize1;
            cluster_size2 = nSize2;
        }

        double jarak;
        int cluster1;
        int cluster2;
        int cluster_size1;
        int cluster_size2;
        @Override
        public int compareTo(ClusterSet c1) {
        	// TODO Auto-generated method stub
        	if (c1.jarak > this.jarak) {
                return -1;
            } else if (c1.jarak == this.jarak) {
                return 0;
            }
            return 1;
        }
    }
    
    boolean m_bPrintNewick = true;;

    public boolean getPrintNewick() {
        return m_bPrintNewick;
    }

    public void setPrintNewick(boolean bPrintNewick) {
        m_bPrintNewick = bPrintNewick;
    }

    /** class representing node in cluster hierarchy **/
    class Node implements Serializable {

        /** ID added to avoid warning */
        private static final long serialVersionUID = 7639483515789717908L;

        Node m_left;
        Node m_right;
        Node m_parent;
        int m_iLeftInstance;
        int m_iRightInstance;
        double m_fLeftLength = 0;
        double m_fRightLength = 0;
        double m_fHeight = 0;

        void setHeight(double fHeight1, double fHeight2) {
            m_fHeight = fHeight1;
            if (m_left == null) {
                m_fLeftLength = fHeight1;
            } else {
                m_fLeftLength = fHeight1 - m_left.m_fHeight;
            }
            if (m_right == null) {
                m_fRightLength = fHeight2;
            } else {
                m_fRightLength = fHeight2 - m_right.m_fHeight;
            }
        }

        void setLength(double fLength1, double fLength2) {
            m_fLeftLength = fLength1;
            m_fRightLength = fLength2;
            m_fHeight = fLength1;
            if (m_left != null) {
                m_fHeight += m_left.m_fHeight;
            }
        }
    }

    protected Node[] m_clusters;
    ArrayList<Integer> m_nClusterNr;

    @Override
    public void buildClusterer(Instances data) throws Exception {
        my_instances = data;
        int nInstances = my_instances.numInstances();
        if (nInstances == 0) {
            return;
        }
        m_DistanceFunction.setInstances(my_instances);
        // use array of integer vectors to store cluster indices,
        // starting with one cluster per instance
        @SuppressWarnings("unchecked")
        Vector<Integer>[] nClusterID = new Vector[data.numInstances()];
        for (int i = 0; i < data.numInstances(); i++) {
            nClusterID[i] = new Vector<Integer>();
            nClusterID[i].add(i);
        }
        // calculate distance matrix
        int nClusters = data.numInstances();

        // used for keeping track of hierarchy
        Node[] clusterNodes = new Node[nInstances];
        doLinkClustering(nClusters, nClusterID, clusterNodes);

        // move all clusters in m_nClusterID array
        // & collect hierarchy
        int iCurrent = 0;
        m_clusters = new Node[number_cluster];
        m_nClusterNr = new ArrayList<Integer>();
        for (int i = 0; i < nInstances; i++) {
            m_nClusterNr.add(0);
        }
        for (int i = 0; i < nInstances; i++) {
            if (nClusterID[i].size() > 0) {
                for (int j = 0; j < nClusterID[i].size(); j++) {
                    m_nClusterNr.set(nClusterID[i].elementAt(j), iCurrent);
                }
                m_clusters[iCurrent] = clusterNodes[i];
                iCurrent++;
            }
        }

    }


    // melakukan clustering menggunakan link method. Sorting menggunakan merge sort
    void doLinkClustering(int nClusters, Vector<Integer>[] nClusterID,
            Node[] clusterNodes) {
    	int nInstances = my_instances.numInstances();
        
        ArrayList<ClusterSet> clusterList = new ArrayList<ClusterSet>();
        double[][] fDistance0 = new double[nClusters][nClusters];
        double[][] fClusterDistance = null;

        for (int i = 0; i < nClusters; i++) {
            fDistance0[i][i] = 0;
            for (int j = i + 1; j < nClusters; j++) {
                fDistance0[i][j] = getDistance0(nClusterID[i], nClusterID[j]);
                fDistance0[j][i] = fDistance0[i][j];
                clusterList.add(new ClusterSet(fDistance0[i][j], i, j, 1, 1));
            }
        }
        Collections.sort(clusterList);
        while (nClusters > number_cluster) {
            int iMin1 = -1;
            int iMin2 = -1;
            // find closest two clusters
            
            // use priority queue to find next best pair to cluster
            ClusterSet t;
            do {
                t = clusterList.get(0);
                clusterList.remove(0);
            } while (t != null
                    && (nClusterID[t.cluster1].size() != t.cluster_size1 || nClusterID[t.cluster2]
                            .size() != t.cluster_size2));
            iMin1 = t.cluster1;
            iMin2 = t.cluster2;
            merge(iMin1, iMin2, t.jarak, t.jarak, nClusterID, clusterNodes);
            // merge clusters

            // update distances & queue
            for (int i = 0; i < nInstances; i++) {
                if (i != iMin1 && nClusterID[i].size() != 0) {
                    int i1 = Math.min(iMin1, i);
                    int i2 = Math.max(iMin1, i);
                    double fDistance = getDistance(fDistance0, nClusterID[i1],
                            nClusterID[i2]);
                    clusterList.add(new ClusterSet(fDistance, i1, i2, nClusterID[i1]
                            .size(), nClusterID[i2].size()));
                }
            }
            Collections.sort(clusterList);

            nClusters--;
        }
    }

    void merge(int iMin1, int iMin2, double fDist1, double fDist2,
            Vector<Integer>[] nClusterID, Node[] clusterNodes) {
        if (iMin1 > iMin2) {
            int h = iMin1;
            iMin1 = iMin2;
            iMin2 = h;
            double f = fDist1;
            fDist1 = fDist2;
            fDist2 = f;
        }
        nClusterID[iMin1].addAll(nClusterID[iMin2]);
        nClusterID[iMin2].removeAllElements();

        // track hierarchy
        Node node = new Node();
        if (clusterNodes[iMin1] == null) {
            node.m_iLeftInstance = iMin1;
        } else {
            node.m_left = clusterNodes[iMin1];
            clusterNodes[iMin1].m_parent = node;
        }
        if (clusterNodes[iMin2] == null) {
            node.m_iRightInstance = iMin2;
        } else {
            node.m_right = clusterNodes[iMin2];
            clusterNodes[iMin2].m_parent = node;
        }
        node.setHeight(fDist1, fDist2);
        clusterNodes[iMin1] = node;
    } // merge

    /** calculate distance the first time when setting up the distance matrix **/
    double getDistance0(Vector<Integer> cluster1, Vector<Integer> cluster2) {
        double fBestDist = Double.MAX_VALUE;
            // set up two instances for distance function
          Instance instance1 = (Instance) my_instances.instance(
            cluster1.elementAt(0)).copy();
          Instance instance2 = (Instance) my_instances.instance(
            cluster2.elementAt(0)).copy();
          fBestDist = m_DistanceFunction.distance(instance1, instance2);
        return fBestDist;
    } // getDistance0

    /**
     * calculate the distance between two clusters
     * 
     * @param cluster1
     *            list of indices of instances in the first cluster
     * @param cluster2
     *            dito for second cluster
     * @return distance between clusters based on link type
     */
    double getDistance(double[][] fDistance, Vector<Integer> cluster1,
            Vector<Integer> cluster2) {
        double fBestDist = Double.MAX_VALUE;
        // find single link distance aka minimum link, which is the closest
        // distance between
        // any item in cluster1 and any item in cluster2
        fBestDist = Double.MAX_VALUE;
        for (int i = 0; i < cluster1.size(); i++) {
            int i1 = cluster1.elementAt(i);
            for (int j = 0; j < cluster2.size(); j++) {
                int i2 = cluster2.elementAt(j);
                double fDist = fDistance[i1][i2];
                if (fBestDist > fDist) {
                    fBestDist = fDist;
                }
            }
        }

        return fBestDist;
    } // getDistance

    /** calculated error sum-of-squares for instances wrt centroid **/
    double calcESS(Vector<Integer> cluster) {
        double[] fValues1 = new double[my_instances.numAttributes()];
        for (int i = 0; i < cluster.size(); i++) {
            Instance instance = my_instances.instance(cluster.elementAt(i));
            for (int j = 0; j < my_instances.numAttributes(); j++) {
                fValues1[j] += instance.value(j);
            }
        }
        for (int j = 0; j < my_instances.numAttributes(); j++) {
            fValues1[j] /= cluster.size();
        }
        // set up two instances for distance function
        Instance centroid = (Instance) my_instances.instance(
                cluster.elementAt(0)).copy();
        for (int j = 0; j < my_instances.numAttributes(); j++) {
            centroid.setValue(j, fValues1[j]);
        }
        double fESS = 0;
        for (int i = 0; i < cluster.size(); i++) {
            Instance instance = my_instances.instance(cluster.elementAt(i));
            fESS += m_DistanceFunction.distance(centroid, instance);
        }
        return fESS / cluster.size();
    } // calcESS

    @Override
    /** instances are assigned a cluster by finding the instance in the training data 
     * with the closest distance to the instance to be clustered. The cluster index of
     * the training data point is taken as the cluster index.
     */
    public int clusterInstance(Instance instance) throws Exception {
        if (my_instances.numInstances() == 0) {
            return 0;
        }
        double fBestDist = Double.MAX_VALUE;
        int iBestInstance = -1;
        for (int i = 0; i < my_instances.numInstances(); i++) {
            double fDist = m_DistanceFunction.distance(instance,
                    my_instances.instance(i));
            if (fDist < fBestDist) {
                fBestDist = fDist;
                iBestInstance = i;
            }
        }
        return m_nClusterNr.get(iBestInstance);
    }

    @Override
    /** create distribution with all clusters having zero probability, except the
     * cluster the instance is assigned to.
     */
    public double[] distributionForInstance(Instance instance) throws Exception {
        if (numberOfClusters() == 0) {
            double[] p = new double[1];
            p[0] = 1;
            return p;
        }
        double[] p = new double[numberOfClusters()];
        p[clusterInstance(instance)] = 1.0;
        return p;
    }

    @Override
    public int numberOfClusters() throws Exception {
        return Math.min(number_cluster, my_instances.numInstances());
    }

    @Override
    public String toString() {
        StringBuffer buf = new StringBuffer();
        int attIndex = my_instances.classIndex();
        if (attIndex < 0) {
            // try find a string, or last attribute otherwise
            attIndex = 0;
            while (attIndex < my_instances.numAttributes() - 1) {
                if (my_instances.attribute(attIndex).isString()) {
                    break;
                }
                attIndex++;
            }
        }
        return buf.toString();
    }

    @Override
    public String graph() throws Exception {
        if (numberOfClusters() == 0) {
            return "Newick:(no,clusters)";
        }
        int attIndex = my_instances.classIndex();
        if (attIndex < 0) {
            // try find a string, or last attribute otherwise
            attIndex = 0;
            while (attIndex < my_instances.numAttributes() - 1) {
                if (my_instances.attribute(attIndex).isString()) {
                    break;
                }
                attIndex++;
            }
        }
        String sNewick = null;
        return "Newick:" + sNewick;
    }

    @Override
    public int graphType() {
        return Drawable.Newick;
    }

}
