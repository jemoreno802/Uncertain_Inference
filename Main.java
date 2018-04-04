import java.io.IOException;

import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;

public class Main {
	
	 public static void main(String[] argv) throws IOException, ParserConfigurationException, SAXException {
		XMLBIFParser parser = new XMLBIFParser();
		BayesianNetwork network = parser.readNetworkFromFile(argv[0]);
		//network.print(System.out);
		
		MyBNInferencer inf = new MyBNInferencer();
		//inf.ask(network, X, e)\
		//getVariableByName
		
		RandomVariable X = network.getVariableByName(argv[1]);
		Assignment e = new Assignment();
		for(int i = 2; i<=argv.length-2;i+=2) {
			e.set(network.getVariableByName(argv[i]), argv[i+1]);
		}
		
		System.out.println("Exact Inference: ");
		Distribution exact = inf.ask(network, X, e);
		System.out.println(exact.toString());
		System.out.println();
		
		System.out.println("Rejection Sampling: ");
		Distribution post = inf.rejection_sampling(X, e, network, 100000);
		System.out.println(post.toString());
		System.out.println();

		
		System.out.println("Likelihood Weighting: ");
		Distribution like = inf.likelihood_weighting(X, e, network, 100000);
		System.out.println(like.toString());
	}
}
