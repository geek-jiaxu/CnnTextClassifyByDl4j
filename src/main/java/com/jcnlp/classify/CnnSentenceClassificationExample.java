package com.jcnlp.classify;

import com.google.common.collect.Maps;
import com.google.common.io.Files;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.Charset;
import java.util.*;

/**
 * @author jc
 */
public class CnnSentenceClassificationExample {

    public static final String WORD_VECTORS_PATH = "data/word2vec.bin";
    public static final String TRAIN_FILE_PATH = "data/train.txt";
    public int labelSize = 0;

    public static void main(String[] args) throws Exception {
        CnnSentenceClassificationExample ccn = new CnnSentenceClassificationExample();

        int batchSize = 64;
        int vectorSize = 200;               //Size of the word vectors. 300 in the Google News model
        int nEpochs = 100;                    //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 50;  //Truncate reviews with length (# words) greater than this
        int cnnLayerFeatureMaps = 100;      //Number of feature maps / channels / depth for each CNN layer
        PoolingType globalPoolingType = PoolingType.MAX;
        Random rng = new Random(12345); //For shuffling repeatability
        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        System.out.println("Loading word vectors and creating DataSetIterators");
        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(new File(WORD_VECTORS_PATH));

        //DataSetIterator evalIter = getDataSetIterator(ccn, wordVectors, batchSize, truncateReviewsToLength, rng, "G:/eval.txt");
        DataSetIterator trainIter = getDataSetIterator(ccn, wordVectors, batchSize, truncateReviewsToLength, rng, TRAIN_FILE_PATH);

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
            .trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
            .weightInit(WeightInit.RELU)
            .activation(Activation.LEAKYRELU)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(0.01)
            .graphBuilder()
            .addInputs("input")
            .addLayer("cnn3", new ConvolutionLayer.Builder()
                .kernelSize(3,vectorSize)
                .stride(1,vectorSize)
                .nIn(1)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4", new ConvolutionLayer.Builder()
                .kernelSize(4,vectorSize)
                .stride(1,vectorSize)
                .nIn(1)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5", new ConvolutionLayer.Builder()
                .kernelSize(5,vectorSize)
                .stride(1,vectorSize)
                .nIn(1)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")      //Perform depth concatenation
            .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                .poolingType(globalPoolingType)
                .dropOut(0.5)
                .build(), "merge")
            .addLayer("out", new OutputLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(3*cnnLayerFeatureMaps)
                .nOut(ccn.labelSize)    //2 classes: positive or negative
                .build(), "globalPool")
            .setOutputs("out")
            .build();

        ComputationGraph net = new ComputationGraph(config);
        net.init();

        System.out.println("Number of parameters by layer:");
        for(Layer l : net.getLayers()) {
            System.out.println("\t" + l.conf().getLayer().getLayerName() + "\t" + l.numParams());
        }

        System.out.println("Starting training");
        net.setListeners(new ScoreIterationListener(100));
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainIter);
            System.out.println("Epoch " + i + " complete. Starting evaluation:");
           // Evaluation evaluation = net.evaluate(evalIter);
           // System.out.println(evaluation.stats());
        }

        ModelSerializer.writeModel(net, "data/cnn.model", true);
        List<String> labels = trainIter.getLabels();

        List<String> tests = Files.readLines(new File("data/test.txt"), Charset.forName("gbk"));
        StringBuilder results = new StringBuilder();

        // 输出测试结果
        for (String str : tests) {
            String productName = str.substring(str.indexOf(" ") + 1, str.length());
            String type = str.substring(0, str.indexOf(" "));

            INDArray featuresFirstNegative = ((CnnSentenceDataSetIterator) trainIter).loadSingleSentence(productName);
            INDArray predictionsFirstNegative = net.outputSingle(featuresFirstNegative);
            Map<String, Double> values = Maps.newHashMap();
            for (int i = 0; i < labels.size(); i++) {
                values.put(labels.get(i), predictionsFirstNegative.getDouble(i));
            }
            Map<String, Double> sortMap = sortByComparator(values);
            results.append("\nType:" + type + ", ProductName : " + productName + "\n   CNN Classify Result : [");
            for (String key : sortMap.keySet()) {
                try {
                    BigDecimal b = new BigDecimal(sortMap.get(key));//BigDecimal 类使用户能完全控制舍入行为
                    double f1 = b.setScale(6, BigDecimal.ROUND_HALF_UP).doubleValue();
                    results.append(key + "(" + f1 + "), ");
                } catch (Exception w) {
                }
            }
            results.append("] \n");
        }
        System.out.println(results.toString());
        System.out.println("finish.........");
    }

    public static Map sortByComparator(Map unsortMap) {
        List list = new LinkedList(unsortMap.entrySet());
        Collections.sort(list, new Comparator() {
            public int compare(Object o1, Object o2)
            {
                return ((Comparable) ((Map.Entry) (o2)).getValue())
                    .compareTo(((Map.Entry) (o1)).getValue());
            }
        });
        Map sortedMap = new LinkedHashMap();

        for (Iterator it = list.iterator(); it.hasNext();) {
            Map.Entry entry = (Map.Entry)it.next();
            sortedMap.put(entry.getKey(), entry.getValue());
        }
        return sortedMap;
    }

    public static DataSetIterator getDataSetIterator(CnnSentenceClassificationExample ccn, WordVectors wordVectors, int minibatchSize,
                                                      int maxSentenceLength, Random rng, String file) throws IOException {
        List<String> examples = Files.readLines(new File(file), Charset.forName("utf-8"));
        CnnSentenceProvider sp = new CnnSentenceProvider(examples);
        ccn.labelSize = sp.allLabels().size();
        return new CnnSentenceDataSetIterator.Builder()
            .sentenceProvider(sp)
            .wordVectors(wordVectors)
            .minibatchSize(minibatchSize)
            .maxSentenceLength(maxSentenceLength)
            .useNormalizedWordVectors(false)
            .build();
    }
}
