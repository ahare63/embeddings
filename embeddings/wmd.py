"""
Gensim WMD function adapted to work with embedder library rather than gensim model.
Original gensim function: https://tedboy.github.io/nlps/_modules/gensim/models/word2vec.html#Word2Vec.wmdistance
"""

from gensim.corpora.dictionary import Dictionary
from pyemd import emd
import numpy as np
from math import sqrt

def wmdistance(document1, document2, embedder):
    """
    Compute the Word Mover's Distance between two documents. When using this
    code, please consider citing the following papers:

    .. Ofir Pele and Michael Werman, "A linear time histogram metric for improved SIFT matching".
    .. Ofir Pele and Michael Werman, "Fast and robust earth mover's distances".
    .. Matt Kusner et al. "From Word Embeddings To Document Distances".
    
    `document1` and `document2` should be lists of words. `embedder` should be an embeddings object from this library, e.g. `FastTextEmbedding()`.
    You should verify that each word in `document1` and `document2` have valid embeddings in the `embedder` object. Otherwise this may throw a `NoneTypeError`
    from line 39.
"""

    dictionary = Dictionary(documents=[document1, document2])
    vocab_len = len(dictionary)

    # Sets for faster look-up.
    docset1 = set(document1)
    docset2 = set(document2)

    # Compute distance matrix.
    distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
    for i, t1 in dictionary.items():
        for j, t2 in dictionary.items():
            if not t1 in docset1 or not t2 in docset2:
                continue
            # Compute Euclidean distance between word vectors.
            distance_matrix[i, j] = sqrt(np.sum((np.asarray(embedder.emb(t1)) - np.asarray(embedder.emb(t2)))**2))

    if np.sum(distance_matrix) == 0.0:
        # `emd` gets stuck if the distance matrix contains only zeros.
        logger.info('The distance matrix is all zeros. Aborting (returning inf).')
        return float('inf')

    def nbow(document):
        d = np.zeros(vocab_len, dtype=np.double)
        nbow = dictionary.doc2bow(document)  # Word frequencies.
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)  # Normalized word frequencies.
        return d

    # Compute nBOW representation of documents.
    d1 = nbow(document1)
    d2 = nbow(document2)

    # Compute WMD.
    return emd(d1, d2, distance_matrix)
