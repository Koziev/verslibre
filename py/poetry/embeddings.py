# -*- coding: utf-8 -*-

import json
import os
import itertools
import six
import codecs
import operator
import logging

from random import choice
import gensim
from gensim.models import word2vec
import corpus_analyser

w2v_path = r'f:\Word2Vec\w2v.CBOW=1_WIN=5_DIM=32.bin'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

logging.info(u'Loading word embeddings {}'.format(w2v_path))
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=not w2v_path.endswith('.txt'))
w2v_dim = len(w2v.syn0[0])

corpus = corpus_analyser.CorpusWords()
corpus.load()
dataset_words = list(corpus.all_words)

model_data = []
for word in dataset_words:
    if word in w2v:
        model_data.append((word, w2v[word]))

nb_words = len(model_data)
new_path = '../tmp/w2v.txt'
logging.info('Writing {} vectors to {}'.format(nb_words, new_path))
with codecs.open(new_path, 'w', 'utf-8') as wrt:
    wrt.write('{} {}\n'.format(nb_words, w2v_dim))
    for word, word_vect in model_data:
        wrt.write(u'{} {}\n'.format(word, u' '.join([str(x) for x in word_vect])))

w2v = gensim.models.KeyedVectors.load_word2vec_format(new_path, binary=False)

#new_path = '../tmp/w2v.bin'
#w2v.wv.save_word2vec_format(new_path, binary=True)

