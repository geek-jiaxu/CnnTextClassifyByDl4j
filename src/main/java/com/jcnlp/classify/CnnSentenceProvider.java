package com.jcnlp.classify;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.datavec.api.util.RandomUtils;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.nd4j.linalg.collection.CompactHeapStringList;
import org.nd4j.linalg.primitives.Pair;

/**
 * 训练数据遍历处理
 * @author jc
 */
import java.util.*;

public class CnnSentenceProvider implements LabeledSentenceProvider {
    private final int totalCount;
    private final List<String> allLabels;
    private List<String> examples = Lists.newArrayList();
    private int cursor;


    public CnnSentenceProvider(List<String> examples) {
        this.totalCount = examples.size();
        allLabels = Lists.newArrayList();

        Set<String> labelSet = Sets.newHashSet();
        for (String example : examples) {
            int index = example.indexOf(" ");
            if (index == -1) continue;
            this.examples.add(example);
            String label = example.substring(0, index);
            if (labelSet.contains(label)) {
                continue;
            }
            labelSet.add(label);
            allLabels.add(label);
        }
        Collections.sort(allLabels);
    }

    public boolean hasNext() {
        return this.cursor < this.totalCount - 1;
    }

    public Pair<String, String> nextSentence() {
        String example = examples.get(this.cursor);
        int index = example.indexOf(" ");
        String label = example.substring(0, index);
        String sentence = example.substring(index + 1, example.length());
        this.cursor ++;
      //  System.out.println(this.cursor ++);
        return new Pair(sentence, label);
    }

    public void reset() {
        this.cursor = 0;

    }

    public int totalNumSentences() {
        return this.totalCount;
    }

    public List<String> allLabels() {
        return this.allLabels;
    }

    public int numLabelClasses() {
        return this.allLabels.size();
    }
}
