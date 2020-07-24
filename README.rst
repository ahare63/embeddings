Embeddings
==========

.. image:: https://readthedocs.org/projects/embeddings/badge/?version=latest
    :target: http://embeddings.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://travis-ci.org/vzhong/embeddings.svg?branch=master
    :target: https://travis-ci.org/vzhong/embeddings

Embeddings is a python package that provides pretrained word embeddings for natural language processing and machine learning.

Instead of loading a large file to query for embeddings, ``embeddings`` is backed by a database and fast to load and query:

.. code-block:: python

    >>> %timeit GloveEmbedding('common_crawl_840', d_emb=300)
    100 loops, best of 3: 12.7 ms per loop
    
    >>> %timeit GloveEmbedding('common_crawl_840', d_emb=300).emb('canada')
    100 loops, best of 3: 12.9 ms per loop
    
    >>> g = GloveEmbedding('common_crawl_840', d_emb=300)
    
    >>> %timeit -n1 g.emb('canada')
    1 loop, best of 3: 38.2 µs per loop


Installation
------------

.. code-block:: sh

    pip install git+https://github.com/ahare63/embeddings.git  # from github
    # If this doesn't work, install manually after downloading and unzipping
    cd embeddings
    pip install -r requirements.txt
    python setup.py install --user


Usage
-----

Upon first use, the embeddings are first downloaded to disk in the form of a SQLite database.
This may take a long time for large embeddings such as GloVe.
Further usage of the embeddings are directly queried against the database.
Embedding databases are stored in the ``$EMBEDDINGS_ROOT`` directory (defaults to ``~/.embeddings``). Note that this location is probably **undesirable** if your home directory is on NFS, as it would slow down database queries significantly.


.. code-block:: python

    from embeddings import GloveEmbedding, FastTextEmbedding, ConcatEmbedding, ElmoEmbedding
    import numpy as np
    
    g = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True)
    f = FastTextEmbedding()
    k = ElmoEmbedding(). # static/global ELMo embeddings
    c = ConcatEmbedding([g, f, k])
    for w in ['canada', 'vancouver', 'toronto']:
        print('embedding {}'.format(w))
        print(g.emb(w))
        print(f.emb(w))
        print(k.emb(w))
        print(c.emb(w))
        
    # Embed an entire sentence with ELMo Contextual Embeddings
    sentence = ['We', 'went', 'to', 'canada', ',' 'vancouver', ',', 'and', 'toronto', '.']
    sen_embedding = k.sentence_embedder.embed_sentence()
    # Print the embedding for each word in the sentence as a single vector
    # See AllenNLP documentation for what these mean
    for i in range(len(sentence)):
        print(np.concatenate((sen_embedding[0][i], sen_embedding[1][i], sen_embedding[2][i])))


Docker
------

If you use Docker, an image prepopulated with the Common Crawl 840 GloVe embeddings and Kazuma Hashimoto's character ngram embeddings is available at `vzhong/embeddings <https://hub.docker.com/r/vzhong/embeddings>`_.
To mount volumes from this container, set ``$EMBEDDINGS_ROOT`` in your container to ``/opt/embeddings``.

For example:

.. code-block:: bash

    docker run --volumes-from vzhong/embeddings -e EMBEDDINGS_ROOT='/opt/embeddings' myimage python train.py


Contribution
------------

Pull requests welcome!
