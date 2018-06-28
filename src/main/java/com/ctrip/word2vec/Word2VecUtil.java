package com.ctrip.word2vec;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * word2vec模型训练
 * @author jc
 */
public class Word2VecUtil {
    private static Logger log = LoggerFactory.getLogger(Word2VecUtil.class.getName());

    public static void main(String[] args) throws Exception {
        trainWord2veModel("G:\\train.txt", "G:\\word2vec.bin");
    }

    /**
     * 训练word2vec模型文件
     * @param trainFile，经过分词之后的训练文件
     * @param modelFile, 模型文件
     */
    public static void trainWord2veModel(String trainFile, String modelFile) {

        // 文件遍历，根据自己训练文件的格式定义
        SentenceIterator iter = new LineSentenceIterator(new File(trainFile));
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                // 因为这里直接拿分类训练文件当做word2vec的训练数据，需要去掉类型字段
                return sentence.substring(sentence.indexOf(" ") + 1, sentence.length()).toLowerCase();
            }
        });

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Star Train Word2vec Model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(10)
                .layerSize(200)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(tokenizerFactory)
                .build();
        vec.fit();

        log.info("Writing word vectors to text file....");
        WordVectorSerializer.writeWord2VecModel(vec, new File(modelFile));
    }


}
