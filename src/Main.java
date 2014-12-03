import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.EM;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class Main {
	public static void main(String args[]) throws Exception {
		DataSource source = new DataSource("D:/weather.clustering.arff");
		Instances data = source.getDataSet();
		String[] options = new String[2];
		options[0] = "-I";                 // max. iterations
		options[1] = "100";
		
		System.out.println("===============K-Means===============");
		//train
		SimpleKMeans kmeans = new SimpleKMeans();   // new instance of clusterer
		kmeans.setOptions(options);     // set the options
		kmeans.buildClusterer(data);    // build the clusterer
		
		//k-means
		ClusterEvaluation eval = new ClusterEvaluation();
		eval.setClusterer(kmeans);                                   // the cluster to evaluate
		eval.evaluateClusterer(data);                                // data to evaluate the clusterer on
		System.out.println("# of clusters: " + eval.getNumClusters());  // output # of clusters
		System.out.println(kmeans.toString());
		

		System.out.println("===============Hirarkikal===============");
		//train 
		HierarchicalClusterer hirarkikal = new HierarchicalClusterer();   // new instance of clusterer
		hirarkikal.setOptions(options);     // set the options
		hirarkikal.buildClusterer(data);    // build the clusterer
		
		//Hirarkikal
		eval.setClusterer(hirarkikal);                                   // the cluster to evaluate
		eval.evaluateClusterer(data);                                // data to evaluate the clusterer on
		System.out.println("# of clusters: " + eval.getNumClusters());  // output # of clusters
		System.out.println(hirarkikal.graph());
		
	}
}
