# Evaluation

## Metrics
For evaluation, MSCOCO Evaluation open source code has been used which calculates the following languge metrics: 

### BLEU

Bilingual Evaluation Understudy (BLEU) is a metric initially designed for machine translation problem, but can be used for other language generation tasks. It counts the number of n-grams occuring in both reference and output sentences. n-gram refers to ‘n‘ contiguous words and it can be 1 (unigram), 2 (bigram), 3 or 4. Here, word order is irrelevant. Its range is from 0 to 1.0.

It is also worthwhile to note, that to not get an unusually high precision value, each reference gram instance is only considered once.
This value is calculated for each reference-candidate gram pair occurring in a sentence, summed up over the whole sentence and subsequently all sentences. There is also a penalty for shorter sentences, as they tend to automatically get a higher score due to mathematical bias. Geometric mean is taken over all sentences to get the corpus level score. 

This metric has several advantages such as its good agreement with human evaluation, simplicity, and language independence.

### ROUGE
ROUGE is Recall-Oriented Understudy for Gisting Evaluation (ROUGE). It is a set of scores made
for text summarization models. In our system, we use the ROUGE-L metric which is based on Longest Common Subsequence (LCS). LCS is the longest common contiguous set of words that occur in reference and candidate sentences, i.e. order is relevant. It calculates the F-score.

### METEOR
Metric for Evaluation of Translation with Explicit ORdering (METEOR) is another Machine Translation metric, that aims to measure the alignment between reference and candidate sentence. This is done by minimizing common chunks between the two. The harmonic mean of precision and recall of n-grams is taken.

### CIDEr
Consensus-based Image Description Evaluation (CIDEr) is a metric which was created for captioning evaluation, and measures a quantity called <b>consensus</b>. Mathematically, the Term Frequency Inverse Document Frequency (TF-IDF) is calculated for each n-gram. <b>Term Frequency</b> for each n-gram scores is based on frequency of <i>co-occurence</i> in reference and candidate sentences. The <b>Inverse Document Frequence</b> measures the <i>rarity</i> of an n-gram and penalizes it based on overall frequency in corpus.

So, the score tries to give higher scores to similarity and creativity.

## Performance

As mentioned before, two types of models organizations - inject and merge, were tried. On top of that, I also experiment on training them with full vocabulary and then a reduced vocabulary. After finalizing reduced vocabulary due to improvement, I finally tried a variation of the system where the encoder had been fine-tuned on the chosen dataset. Below is the summary of results.

### 1. Inject model results

| Score | Inject, Full |Inject,Reduced | Deeper Inject |After Fine -Tuning|
|--|--|--|--|--|
|B1 |0.5725 | 0.582 | 0.566 | 0.559 |
| B2 | 0.273 | 0.285 | 0.291 | 0.319 |
|B3 | 0.189 | 0.203 | 0.180 | 0.228 |
|B4 | 0.085 | 0.095 | 0.843 | 0.113 |
|METEOR | 0.212 | 0.215 | 0.202 | 0.220 |
|ROUGE | 0.442 | 0.450 | 0.448 | 0.454|
|CIDEr | 0.232 | 0.252 | 0.208 | 0.263|


### 2. Merge model results

|Score |Merge,Full| Merge, Reduced| Deeper Merge| After Fine -Tuning|
|--|--|--|--|--|
|B1 | 0.575 | 0.585 | 0.569 | 0.602|
|B2 | 0.316 | 0.313 | 0.309 | 0.365|
|B3 | 0.212 | 0.219 | 0.217 | 0.265|
|B4 | 0.098 | 0.105 | 0.101 | 0.139|
|METEOR | 0.212 | 0.219 | 0.220 | 0.230|
|ROUGE | 0.455 | 0.459 | 0.461 | 0.491|
|CIDEr | 0.229 | 0.267 | 0.278 | 0.377|

## Inferences

1. Reduced vocabulary improves scores and also improves learning rate. This can be seen from the following graphs.

![Learning curve](/assets/learningrate.png)

![CIDEr comparision](/assets/CIDEr.png)


2. Increasing number of layers is not helpful in improving model, especially for inject models. This can be seen in 4th column of both the tables. This could be due to overfitting of the model in the data.


3. In both model types, fine tuning has greatly improved the scores. 


4. Merge models consistently outperforms inject models.




