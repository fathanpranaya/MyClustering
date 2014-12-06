import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.EM;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class Main {
	public static void main(String args[]) throws Exception {
		DataSource source = new DataSource("E:/workspace/MYClustering/weather.clustering.arff");
		Instances data = source.getDataSet();
		
		System.out.println("===============K-Means===============");
		//train
		MyKMeans kmeans = new MyKMeans();   // new instance of clusterer
		kmeans.buildClusterer(data);    // build the clusterer
		
		//k-means
		ClusterEvaluation eval = new ClusterEvaluation();
		eval.setClusterer(kmeans);                                   // the cluster to evaluate
		eval.evaluateClusterer(data);                                // data to evaluate the clusterer on
		System.out.println("# of clusters: " + eval.getNumClusters());  // output # of clusters
		System.out.println(kmeans.toString());
		
		System.out.println("****************************");
		System.out.println(eval.clusterResultsToString());
		System.out.println("****************************");
		

		System.out.println("===============Hirarkikal===============");
		//train 
		MyAgglomerative hirarkikal = new MyAgglomerative();   // new instance of clusterer
		hirarkikal.buildClusterer(data);    // build the clusterer
		
		//Hirarkikal
		eval.setClusterer(hirarkikal);                                   // the cluster to evaluate
		eval.evaluateClusterer(data);                                // data to evaluate the clusterer on
		System.out.println("# of clusters: " + eval.getNumClusters());  // output # of clusters
		System.out.println(hirarkikal.graph());
		System.out.println("****************************");
		System.out.println(eval.clusterResultsToString());
		System.out.println("****************************");
//		
	}
}
