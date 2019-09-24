# Text Classification
Text classification for essay grading


# Business Problem

1. Start with BERT. Get it to run, see which tasks it performs on.
2. Use HuggingFace's pytorch_transformers to try BERT.
  * I tried BERT. The fine-tuning step takes too long on a CPU. However, I think that
  fine-tuning just applies softmax regression with the final hidden layer as input(?).
  If so, then it's straightforward to extend BERT to our task. However, I should try
  simpler approaches first.
  * Indeed, I found exactly such a blog. I guess I could try it along with other
  approaches to grade the essays.
3. For my automatic essay scoring model, the paper proposes a NN model and compares
to two baselines, readily available. I guess I should take a stab at it from a blank-
slate, then try the methods found in literature.


* https://github.com/NirantK/awesome-project-ideas#text

# Datasets
1. CoNLL-2003, named entity recognition (NER): https://www.aclweb.org/anthology/W03-0419
  * __Example__: _U.N. official Ekeus heads for Baghdad_. :arrow_right: [ORG _U.N._] _official_ [PER _Ekeus_] heads for [LOC _Baghdad_].
  * There are four named entity's: person (PER), organization (ORG), location (LOC), and miscellaneous (MIS).
2. Automated Essay Scoring, text classification: https://www.kaggle.com/c/asap-aes/data

# BERT Overview



# References
* `pytorch_transformers` (Huggingface): https://github.com/huggingface/pytorch-transformers
* BERT: [here](https://arxiv.org/pdf/1810.04805v2.pdf)
* __XLNet: Generalized Autoregressive Pretraining for Language Understanding__: [here](XLNet: Generalized Autoregressive Pretraining for Language Understanding)
* __Automatic Essay Scoring dataset__: [here](https://github.com/nusnlp/nea)
* __Character-level Convolutional Networks for Text Classification__: [here](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf). Describes some common test datasets.
* [BERT tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)
