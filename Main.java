import java.io.FileInputStream;
import java.io.IOException;
import java.util.Scanner;

import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;

public class Main {
	
	 public static void main(String[] argv) throws IOException, ParserConfigurationException, SAXException {
		 
		System.out.println("Which part of the project do you want to run?");
		System.out.println("Enter 1: Exact Enumeration");
		System.out.println("Enter 2: Rejection Sampling & Likelihood-Weighting");
		int part = new Scanner(System.in).nextInt();
		MyBNInferencer inf = new MyBNInferencer();
		XMLBIFParser parser = new XMLBIFParser();
		Assignment e = new Assignment();
		BayesianNetwork network;
		
		
		if (part == 1) { //Testing Exact Inference
			if (argv[0].endsWith("xml"))
				  network = parser.readNetworkFromFile(argv[0]);
				else{
					BIFParser bifParser = new BIFParser(new FileInputStream(argv[0]));
					network = bifParser.parseNetwork();	
			}
			RandomVariable X = network.getVariableByName(argv[1]);
			
			for(int i = 2; i<=argv.length-2;i+=2) {
				e.set(network.getVariableByName(argv[i]), argv[i+1]);
			}
			
			System.out.println("Exact Inference: ");
			long start0 = System.nanoTime();
			Distribution exact = inf.ask(network, X, e);
			long end0 = System.nanoTime();
			System.out.println(exact.toString());
			System.out.println("Time = "+ (end0-start0)/Math.pow(10, 6));
			System.out.println();
			
		} else { //Testing Approximate Inference
			int N = Integer.parseInt(argv[0]); //The sample size
			//int N = 10000;
			if (argv[1].endsWith("xml"))
				network = parser.readNetworkFromFile(argv[1]);
			else{
				BIFParser bifParser = new BIFParser(new FileInputStream(argv[1]));
				network = bifParser.parseNetwork();	
			}
			RandomVariable X = network.getVariableByName(argv[2]);
			
			for(int i = 3; i<=argv.length-2;i+=2) {
				e.set(network.getVariableByName(argv[i]), argv[i+1]);
			}
			System.out.println("Rejection Sampling: ");
			long start = System.nanoTime();
			Distribution post = inf.rejection_sampling(X, e, network, N);
			long end = System.nanoTime();
			System.out.println("Time = "+ (end-start)/Math.pow(10, 6));
			System.out.println(post.toString());
			System.out.println();

			
			System.out.println("Likelihood Weighting: ");
			long start1 = System.nanoTime();
			Distribution like = inf.likelihood_weighting(X, e, network, N);
			long end1 = System.nanoTime();
			System.out.println("Time = "+ (end1-start1)/Math.pow(10, 6));
			System.out.println(like.toString());
		}
		
		
		
	}
}
