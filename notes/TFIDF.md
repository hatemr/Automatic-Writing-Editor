# Scoring, term weighting, and the vector space model
Notes from chapter 6 of [_Introduction to Information Retrieval_](https://nlp.stanford.edu/IR-book/information-retrieval-book.html) by Chris Manning.

## Tf-Idf
Tf-idf (term frequency-inverse document frequency) is a weighting scheme to tell how important a word is to a document. In tf-idf, a word's importance increases proportionally to the number of times the word appears in the document, but is offset by the frequency of the word in the corpus. Tf-idf weighting are often used as a central tool in search engines to rank relevant documents given a user query.

We have two terms, tf and idf:
1. $tf_{t,d}$: number of occurences of term t in document d. This is a weight assigned to term t.
2. $idf_{t} = \log \frac{N}{df_{t}}$: number of documents in the collection that contain term t.

Tf-idf treats the document as a bag-of-words, where the word ordering does not matter. The tf term by itself has a critical problem: all terms are equally important for assessing relevancy on a query. For example, in a collection of documents on the auto industry, the term _auto_ appears in almost every document and doesn't help narrow down relevant documents.

To reduce auto's weight, we can lower auto's weight according to how often it appears in any doc in the collection. If _auto_ appears in all documents, its weight is reduced heavily. Conversely, if _auto_ appears in only few documents, then _auto_'s weight should remain high because it provides information about which document is relevant.

We denote $df_{t}$ as the number of documents in the collection containing term t. We then define the inverse document frequency term as $idf_{t} = \log \frac{N}{df_{t}}$. If the term is common, then $df_{t}$ is high and $idf_{t}$ is low, lowering the weight a lot. The idf of a rare term is high, and idf of a common term is low.

## Tf-idf Weighting
With these two terms, we can construct the term's weight. The tf-idf weight of term t in document d is:
$$ tf-idf_{t,d} = tf_{t,d} \cdot idf_{t} $$

$tf-idf_{t,d}$ is:
1. highest when t occurs many times in a small number of documents. Thus it has high discriminating power to choose those documents.
2. lower when the term occurs few times in a document, or occurs in many documents. It offers a smaller relevance signal.
3. lowest when the term occurs in almost all documents.

We can then compute the relevance score of a document d to a query q by summing the tf-idf of all terms in the query:

$$ Score(q, d) = \sum_{t \in q} tf-idf_{t,d}

## Vector space model
We construct the vector $V(d) = \[w_1, w_2,...,0,...,w_N\]^T$ as the vector of tf-idf weights for the N possible terms for the document. Thus the document is represented by this N-dim vector. We represent documents in a common vector space.

We can measure the similarity of two documents in this vector space using _cosine similarity_:

$$ sim(d_1, d_2) = \frac{V(d_1) \cdot V(d_2)}{ |V(d_1)| |V(d_2)|}  $$

Recall that the cosine similarity is the cosine of the angle between the two vectors. If the angle is 0, there is perfect overlap and similarity is 1. Also note that the denominator serves to normalize each vector to unit vectors (length of one). Thus we can rewrite using the unit vectors:

$$ sim(d_1, d_2) = v(d_1) \cdot v(d_2) $$

With these vector representations of the documents, we can measure which documents are similar to each other.

### Collection of documents as matrix
We can construct a _term-document matrix_; an M x N matrix where rows are terms, M in total, and columns represent documents, N in total. This matrix represents a collection of documents.

### Scoring
We can now represent both documents and queries as vectors (of dimension M), treating the query as a small document. So, we can assign the score of each document d as the dot product between the document and the query:

$$ v(q) \cdot v(d) $$

MOreover, we can calculate the scores of all N documents using simple matric multiplication, where the collection of N documents is an N x M matrix and the query is an M x 1 vector. We just left multiply, and select the element with the highest score. In practice, this naive calculation is infeasible because the number of terms M is very large.

## Variants of tf-idf
### Sublinear tf scaling
The tf term grows too quickly with its frequency: 20 occurences doesn't signal 20 times the significance of 1 occurence. Therefore, we try to suppress the growth of tf by applying log:

$$ wf_{t,d} = 1+\log tf_{t,d} if tf_{t,d}>0 else 0. $$

![sublinear tf](sublinear_tf.png)

And we replace tf with wf:

$$ wf-idf_{t,d} = wf_{t,d} \cdot idf_{t}
 $$

### Maximum tf normalization
The tf can vary based largely on the document length, making a term found in two differently sized documents uncomparable. As an extreme example, consider taking a document and pasting a copy of itself onto the end. The tf would double, even though the term isn't any more important to the document.

To normalize the tf across documents, we divide tf by the maximum tf in that document. Then, we smooth this normalized term frequency using a smoothing term a:

$$ ntf_{t,d} = a+(1-a) \frac{tf_{t,d}}{tf_{max}(d)}

where a varies from 0 to 1. 

Maximum tf normalization suffers from the following issues:
1. A document might contain one outlier term that appears unusually highly frequently. This maximum would suppress the tf's of all the other terms.
2. Some documents have terms that appear roughly equally often; other documents have more skewed term frequencies. Even distributions and skewed distributions should be treated differently.

### Pivoted normalized document length
This method addresses the situation when some docuemnts are of vastly different lengths. Given that our dataset contains timed essays, I will assume the lengths are roughly similar. Therefore, I will skip this method.


## Summary
We laid out a document embedding where the elements of the vector are weights assigned to each term in the vocabulary. If the term does not appear in the document, then its weight is zero. Otherwise, the weight for each term in a given document is given by the tf-idf weighting. There are many other weighting schemes available, but tf-idf offers us a baseline. Finally, note that the query is treated as a small document. Thus, we have not only a document embedding but also a sentence embedding.


## References
1. http://www.tfidf.com
2. https://nlp.stanford.edu/IR-book/pdf/06vect.pdf