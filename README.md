# Text Classification
For essay scoring.

# Business Problem
Clear and effective communication is hard. Especially in a business with complex goals and shifting demands. That's why I am building a writing editor, to automatically grade one's writing. This tool aims to help people write reports in clear, concise prose.

## Plan
* Start with TFIDF approaches. Then try LSTMs, and add attention. If we get really far, then maybe BERT. But only we get far.

## Progress
* __9/25/19__: learned about tf-idf approaches. Notes here: [TFIDF.md](notes/TFIDF.md).
  * tf-idf is a method for embedding documents. I could feed in the document embeddings as features to predict the essay scores. This seems straightforward. Where is a tf-idf implementation? How do I pre-process the text to feed in in the right form?

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
