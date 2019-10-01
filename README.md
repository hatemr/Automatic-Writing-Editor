# Text Classification for Essay Scoring
To help us write better.

`conda env create -f environment.yml`  
`conda activate essy_scoring`

# Business Problem
Clear and effective communication is hard. Especially in a business with complex goals and shifting demands. That's why I am building a writing editor, to automatically grade one's writing. This tool aims to help people write reports in clear, concise prose.

## Plan
* Start with TFIDF approaches. Get a simple working model, then deploy using Flask+AWS.
* After the model is deployed online, refine the model and try fancier NLP techniques, like LSTMs, then add attention.

## Progress
* __9/25/19__: learned about tf-idf approaches. Notes here: [TFIDF.md](notes/TFIDF.md).
  * tf-idf is a method for embedding documents. I could feed in the document embeddings as features to predict the essay scores. This seems straightforward. Where is a tf-idf implementation? How do I pre-process the text to feed in in the right form?
* __9/26/19__: Pre-processed and vectorized essays `where essay_set=1` using tf-idf in `sklearn`. 
  * Need to lower the dimensionality because the number of samples is too low. 
  * Looks like SVD might help for dimensionality reduction. Look up latent semantic indexing. Obviously, PCA is one way to reduce dimensions.
* __9/27/19__: I found the area of NLP that I want to target: [Text and discourse coherence](https://web.stanford.edu/~jurafsky/slp3/21.pdf). 
  * We can use a _coherence_ measure as a predictor variable for essay scoring (e.g. [this paper](https://www.aclweb.org/anthology/D13-1180)).
  * Indeed, _text coherence_ is [very useful for predicting _text readability_ and, yes, _essay scoring_](https://www.aclweb.org/anthology/D18-1464). This paper proposes a next coherence measure and uses it to achieve SoA on readability and essay scoring.
  * A list of features used by EASE (winner of ASAP competition) on page 434 [here](https://www.cs.cmu.edu/~ark/EMNLP-2015/proceedings/EMNLP/pdf/EMNLP049.pdf).
  * I created the length-based features from the paper above. Next, I need to apply SVD to the term-document matrix.
* __9/28/19__: I made a pipeline to pre-process the data, using SVD dim. reduction, and a grid search with SGDClassifier using logistic regression.
  * Next, I need to measure the performance of the model.
  * Also, need to extract the prompts from words docs and save in a .txt file.
  * I didn't use the length-based features, only the LSA features. I will add them in later as a pipeline step in sklearn.
* __9/29/19__: Started learning Flask.
* __9/29/19__: Continued learning Flask. Many roadblocks when following tutorials.

# Datasets
1. CoNLL-2003, named entity recognition (NER): https://www.aclweb.org/anthology/W03-0419
  * __Example__: _U.N. official Ekeus heads for Baghdad_. :arrow_right: [ORG _U.N._] _official_ [PER _Ekeus_] heads for [LOC _Baghdad_].
  * There are four named entity's: person (PER), organization (ORG), location (LOC), and miscellaneous (MIS).
2. Automated Essay Scoring, text classification: https://www.kaggle.com/c/asap-aes/data


# References
* `pytorch_transformers` (Huggingface): https://github.com/huggingface/pytorch-transformers
* BERT: [here](https://arxiv.org/pdf/1810.04805v2.pdf)
* __XLNet: Generalized Autoregressive Pretraining for Language Understanding__: [here](XLNet: Generalized Autoregressive Pretraining for Language Understanding)
* __Automatic Essay Scoring dataset__: [here](https://github.com/nusnlp/nea)
* __Character-level Convolutional Networks for Text Classification__: [here](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf). Describes some common test datasets.
* [BERT tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)
* Project ideas: https://github.com/NirantK/awesome-project-ideas#text
