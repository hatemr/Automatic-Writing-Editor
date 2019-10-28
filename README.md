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
* __9/30/19__: Continued learning Flask. Many roadblocks when following tutorials.
* __10/1/19__: Continued with Flask tutorials. Overcoming many hurdles and debugging a lot but it's coming along.
  * Added comments to aes.py.
* __10/10/19__: Worked on Flask app. Changed input text to paragraph instead of one line. Started adding ML model to the app.
* __10/11/19__: Added essay grade prediction. Still need to get the database working.
* __10/13/19__: I tried to get the databases to work for a couple hours, with not success. After talking with Amit, we got it to work. Problem was I needed to define the port since my datbase uses a non-default port (5433).
* __10/14/19__: Read about Flask Boostrap. I don't yet see how I can use it to make my app prettier.
* __10/15/19__: Read about HTML and CSS. Glad to learn, but I need a _quick_ way to make my app look nicer. Just moving text around and adding color is not what I need, I need it to look slick.
* __10/16/19__: Continued reading about HTML and CSS. Still mystified.
* __10/18/19__: Breakthrough. I added a Bootstrap Theme to Flask, using the tutorial [here](https://www.youtube.com/watch?v=3NsEGaCIT38). This is exactly what I wanted.
* __10/19/19__: Incorporated Jinja extends with the bootstrap/base.html (I think this is where Flask-Bootstrap comes into play).
* __10/23/19__: Debugged database connection to avoid circular imports. The app's frontend is ready. So I should work on the conent.
* __10/27/19__: Started working on applying the writing rule "keep subject near their verbs". 3 hours.
* __10/28/19__: Wrote code to highlight the words which violated the rules, and improved the function's code. 2 hours.

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
