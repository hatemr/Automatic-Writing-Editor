#https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-download-auto-examples-model-selection-grid-search-text-feature-extraction-py
import numpy as np
import pandas as pd
from time import time
from joblib import dump, load
import pickle

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


def load_data(prompts=[1], 
    filepath='../data/asap-aes/training_set_rel3.xlsx',
    bucketize=True,
    num_buckets=4):    
    """
    Load essays
    """
    t0 = time()
    print('Loading prompts: {}'.format(prompts))
    
    df = pd.read_excel(filepath)
    df = df[df.essay_set.isin(prompts)]

    data = list(df.essay)
    domain1_score = df.domain1_score

    # to combine sparse classes
    if bucketize:
        target = pd.qcut(domain1_score, num_buckets, labels=False, duplicates='drop')
        
        bum_buckets_temp = num_buckets
        while target.unique().shape[0]<num_buckets:
            bum_buckets_temp += 1
            target = pd.qcut(df[df.essay_set==1].domain1_score, bum_buckets_temp, labels=False, duplicates='drop')
        target = target.values
    else:
        target = domain1_score.values

    # to make sure
    assert type(data)==list
    assert len(data)==target.shape[0]

    print('Data uploaded. Done in {:.0f} seconds'.format(time()-t0))
    return data, target



def build_model(data, target):
    """
    Transform text corpus into a document vector.

    Parameters
    ----------
    corpus: list of strings
        Each element of the corpus is a document.
    ...

    returns
    -------
    X: nmatrix, [n_sample, n_features]
        Data matrix where rows are documents and columns are features
    
    """

    # Pipeline first apply tf-idf, then reduced the dimension using SVD (PCA),
    # then Normalizer normalizes the rows (observations) to be length one by dividing
    # by the l2 norm. This allows the cosine similarity of two document vectors to be
    # the inner product of the two doc vecs. This is convention in the Information
    # Retrieval community. Could also get the same result by using norm: 'l2' in
    # TfidfVectorizer.

    # The classifier used is logistic regression
    pipeline = Pipeline([
        ('vec', TfidfVectorizer(stop_words='english', use_idf=True)), 
        ('svd', TruncatedSVD()), 
        ('norm', Normalizer(copy=False)),
        ('clf', SGDClassifier(loss='log', tol=1e-3))])

    # max_df: If int, ignores terms that appear in more than max_df documents.
    # Can be used to discard unuseful words.
    # max_features: take only the top max_features words ordered by term
    # frequency.
    # ngram_range: takes individual terms (1grams) or also 2grams in its vocab.
    # The last two are number of final PCs and regularization strength.
    parameters = {
        #'vec__max_df': (None), #0.5, 0.75, 1.0),
        #'vec__max_features': (None), # 100, 500, 1000, 10000),
        #'vec__ngram_range': ((1,1)), # (1,2)),
        'svd__n_components': (100, 200),
        'clf__alpha': (0.00001, 0.000001)
    }

    grid_search = GridSearchCV(pipeline,
        param_grid = parameters,
        scoring = 'accuracy',
        cv = 5)

    t0 = time()
    print("Fitting grid search")
    grid_search.fit(data, target)
    print('Done fitting in {:.0f} seconds'.format(time()-t0))

    # save model. Use pickle + dictionaries instead
    #dump(grid_search, 'grid_search.joblib')
    result = {'grid_search': grid_search}
    file = open('grid_search', 'wb')
    pickle.dump(result, file)
    file.close()
    #print(grid_search.best_score_, grid_search.scorer_, grid_search.cv_results_.keys())
    return

# Custom transformer to extract more features.
# To be added to Pipeline later.

# class Ease_feature_extractor:
#     def __init__(self, ):
#         self.__feature_names = ['num_chars', 'num_words', 'num_commas', 'num_apostrophies', 'sent_end_punc', 'avg_sent_len', 'avg_word_len']
#         self.__features = defaultdict()
#         for name in self.__feature_names:
#             self.__features[name] = []
        
#     def fit_transform(self, raw_documents):
#         assert type(raw_documents)==list
#         for doc in raw_documents:
            
#             # Length features
#             self.__features['num_chars'].append(len(re.sub(r"\s+", "", doc)))
#             self.__features['num_words'].append(len(nltk.RegexpTokenizer(r'\w+').tokenize(doc)))
#             self.__features['num_commas'].append(len(nltk.RegexpTokenizer(r',').tokenize(doc)))
#             self.__features['num_apostrophies'].append(len(nltk.RegexpTokenizer(r"'").tokenize(doc)))
#             self.__features['sent_end_punc'].append(len(nltk.RegexpTokenizer(r'[.?!]').tokenize(doc)))
            
#             sentence_lengths = [len(nltk.RegexpTokenizer(r'\w+').tokenize(sentence)) for sentence in nltk.sent_tokenize(doc)]
#             self.__features['avg_sent_len'].append(sum(sentence_lengths)/len(sentence_lengths))
            
#             words = [len(word) for word in nltk.RegexpTokenizer(r'\w+').tokenize(doc)]
#             self.__features['avg_word_len'].append(sum(words)/len(words))
            
#             # tf-idf
#             vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
            
#             vectorizer = TfidfVectorizer(max_df=0.5, 
#                                          max_features=opts.n_features,
#                                          stop_words='english',
#                                          use_idf=opts.use_idf)
#         return self.__features
#     def get_feature_names(self):
#         return self.__feature_names


def main():
    data, target = load_data()
    build_model(data, target)


if __name__ == "__main__":  # if run as .py script
    main()