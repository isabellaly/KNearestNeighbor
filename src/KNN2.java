import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * KNN Implementation
 * 
 * @author Team 2
 * 
 */
public class KNN2 {

	private static final double[] steps = { 1.0, 0.5, 0.2, 0.1, 0.05, 0.02,
			0.01 };
	private static final int sweepSize = 200;
	private static final int sweepRound = 0;

	public static void main(String[] args) throws IOException {
		BufferedReader in = new BufferedReader(new FileReader(args[0]));
		OutputStreamWriter writer = new OutputStreamWriter(System.out);
		BufferedWriter out = new BufferedWriter(writer);
		String line = null;

		int k = Integer.parseInt(args[1]);
		int fold = Integer.parseInt(args[2]);

		int features = 0;
		ArrayList<String> trainData = new ArrayList<String>();
		boolean start = false;
		while ((line = in.readLine()) != null) {
			if (start) {
				trainData.add(line);
			} else if (line.contains("@attribute")) {
				features++;
			} else if (line.contains("@data")) {
				start = true;
			}
		}
		features--;
		int length = trainData.size();

		double[] weights = new double[features];
		for (int i = 0; i < features; i++) {
			weights[i] = 100;
		}
		// get max and min
		double[] mins = new double[features];
		for (int i = 0; i < features; i++) {
			mins[i] = Double.MAX_VALUE;
		}
		double[] maxs = new double[features];
		for (String t : trainData) {
			String[] splitT = t.split(",");
			for (int i = 2; i < features; i++) {
				double doubleT = Double.parseDouble(splitT[i]);
				if (doubleT < mins[i]) {
					mins[i] = doubleT;
				}
				if (doubleT > maxs[i]) {
					maxs[i] = doubleT;
				}
			}
		}

		ArrayList<Node> nodes = new ArrayList<Node>();
		int validationSetSize = length / fold;
		int correct = 0;
		int w = features - 1;
		ArrayList<AccuracyWeight> accuracies = new ArrayList<AccuracyWeight>();
		for (int r = 0; r < sweepRound; r++) {
			w = features - 1;
			while (w >= 0) {
				// if (r == 0) {
				// weights[w] = 1;
				// }
				accuracies.clear();
				for (int m = 0; m < sweepSize; m++) {
					weights[w] += steps[r];
					correct = 0;
					for (int f = 0; f <= fold; f++) {
						// start a new validation set
						int validationSetStart = f * validationSetSize;
						int validationSetEnd = Math.min((f + 1)
								* validationSetSize, length);
						// train the data set
						for (int i = validationSetStart; i < validationSetEnd; i++) {
							String[] splitTest = trainData.get(i).split(",");
							nodes.clear();
							for (int j = 0; j < length; j++) {
								if (j < validationSetStart
										|| j >= validationSetEnd) {
									String[] splitTrain = trainData.get(j)
											.split(",");
									Node n = new Node(splitTrain,
											calculateSimilarity(splitTrain,
													splitTest, weights,
													features, mins, maxs));
									nodes.add(n);
								}
							}
							int predictLabel = predict(nodes, k, features);
							int testLabel = getLabelValue(splitTest[features]);
							if (predictLabel == testLabel) {
								correct++;
							}
						}
					}
					double accuracy = (double) correct / (double) length;
					accuracies.add(new AccuracyWeight(accuracy, weights[w]));
				}
				Collections.sort(accuracies);
				weights[w] = accuracies.get(0).weight;
				out.write("Round " + r + ": " + accuracies.get(0).accuracy
						+ " " + weights[w] + "\n");
				w--;
			}
		}
		out.flush();
		in.close();
		out.close();
	}

	public static int predict(ArrayList<Node> nodes, int k, int features) {
		Collections.sort(nodes);
		double[] classValues = new double[features];
		for (int n = 0; n < k; n++) {
			int label = getLabelValue(nodes.get(n).data[features]);
			classValues[label] += nodes.get(n).similarity;
		}
		// find argmax
		int predictLabel = 0;
		double maxClassValue = 0.0;
		for (int m = 0; m < classValues.length; m++) {
			if (classValues[m] > maxClassValue) {
				maxClassValue = classValues[m];
				predictLabel = m;
			}
		}
		return predictLabel;
	}

	public static double calculateSimilarity(String[] train, String[] test,
			double[] weights, int features, double[] mins, double[] maxs) {
		double sum = 0.0;
		if (!train[0].equals(test[0])) {
			sum += (weights[0]);
		}
		if (!train[1].equals(test[1])) {
			sum += (weights[1]);
		}
		for (int i = 2; i < features; i++) {
			double normalizedTrain = (Double.parseDouble(train[i]) - mins[i])
					/ (maxs[i] - mins[i]);
			double normalizedTest = (Double.parseDouble(test[i]) - mins[i])
					/ (maxs[i] - mins[i]);
			sum += Math.pow((normalizedTrain - normalizedTest), 2) * weights[i];
		}
		return 1 / (Math.sqrt(sum));
	}

	public static int getLabelValue(String label) {
		if (label.equals("C1")) {
			return 0;
		} else if (label.equals("C2")) {
			return 1;
		} else if (label.equals("C3")) {
			return 2;
		} else if (label.equals("C4")) {
			return 3;
		} else {
			return 4;
		}
	}

	public static void shuffle(ArrayList<String> arr) {
		Random rand = new Random();
		for (int i = arr.size() - 1; i > 0; i--) {
			int index = rand.nextInt(i + 1);
			// Simple swap
			String a = arr.get(index);
			arr.set(index, arr.get(i));
			arr.set(i, a);
		}
	}

	public static class Node implements Comparable<Node> {
		String[] data;
		double similarity;

		public Node(String[] data, double similarity) {
			this.data = data;
			this.similarity = similarity;
		}

		public int compareTo(Node other) {
			if (this.similarity < other.similarity) {
				return 1;
			} else if (this.similarity > other.similarity) {
				return -1;
			} else {
				return 0;
			}
		}
	}

	public static class AccuracyWeight implements Comparable<AccuracyWeight> {
		double accuracy;
		double weight;

		public AccuracyWeight(double accuracy, double weight) {
			this.accuracy = accuracy;
			this.weight = weight;
		}

		public int compareTo(AccuracyWeight other) {
			if (this.accuracy < other.accuracy) {
				return 1;
			} else if (this.accuracy > other.accuracy) {
				return -1;
			} else {
				return 0;
			}
		}
	}
}
