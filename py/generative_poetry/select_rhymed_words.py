import io
import pickle
import os
import collections
import math

import numpy as np
import pyconll
import rutokenizer
import ruword2tags
from ufal.udpipe import Model, Pipeline, ProcessingError
import fasttext

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from poetry.phonetic import Accents, rhymed


def ngrams(s, n):
    return set(u''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    #if len(shingles1) == 0 or len(shingles2) == 0:
    #    print('ERROR@135 s1="{}" s2="{}"'.format(s1, s2))
    #    exit(0)

    return float(len(shingles1 & shingles2))/float(len(shingles1 | shingles2) + 1e-6)


def get_attr(token, tag_name):
    if tag_name in token.feats:
        v = list(token.feats[tag_name])[0]
        return v

    return ''


def v_cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom > 0:
        return np.dot(a, b) / denom
    else:
        return 0



class RhymeSelector:
    def __init__(self, accentuator, w2v_model, gren, tokenizer):
        self.tokenizer = tokenizer
        self.w2v_model = w2v_model
        self.gren = gren
        self.accentuator = accentuator
        self.word2rhymes = collections.defaultdict(set)
        self.word_p = dict()

    def load_udpipe(self, udp_model_file):
        self.model = Model.load(udp_model_file)
        self.pipeline = Pipeline(self.model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        self.error = ProcessingError()

    def get_last_word_and_tags(self, line):
        processed = self.pipeline.process(line, self.error)
        if self.error.occurred():
            print("An error occurred when running run_udpipe: ")
            raise RuntimeError()

        parsings = pyconll.load_from_string(processed)
        last_parsing = parsings[-1]
        ilast = -1
        last_token = last_parsing[ilast]
        while last_token.upos == 'PUNCT':
            ilast -= 1
            last_token = last_parsing[ilast]

        tags = [last_token.upos]
        if last_token.upos in ('NOUN', 'ADJ'):
            tags.append('Case={}'.format(get_attr(last_token, 'Case')))

        return last_token.form, tags

    def initialize_from_corpus(self, poetry_corpus_path):
        word2count = collections.Counter()
        with io.open(poetry_corpus_path, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                for word in self.tokenizer.tokenize(line.strip()):
                    if word in self.accentuator.word_accents_dict:
                        word2count[word] += 1

                if False:
                    last_words = []
                    for pline in [x.strip() for x in line.strip().split('|')]:
                        last_words.append(self.get_last_word_and_tags(pline))

                    for word1, tags1 in last_words[:-1]:
                        if word1 in self.accentuator.word_accents_dict:
                            for word2, tags2 in last_words[1:]:
                                if word2 in self.accentuator.word_accents_dict:
                                    if rhymed(self.accentuator, word1, tags1, word2, tags2):
                                        self.word2rhymes[word1].add(word2)

        total_count = float(sum(word2count.values()))
        for word, count in word2count.items():
            self.word_p[word] = count / total_count

    def get_rhymes(self, word):
        res = []
        if self.w2v_model is not None:
            v1 = self.w2v_model[word]
        else:
            v1 = None

        tagsets1 = list(self.gren[word])
        pox1 = set(tagset.split(' ')[0] for tagset in tagsets1)

        for word2 in self.accentuator.word_accents_dict.keys():
            if word != word2 and not word.endswith(word2) and not word2.endswith(word):
                if rhymed(self.accentuator, word, [], word2, []):
                    score = 1.0
                    tagsets2 = list(self.gren[word2])
                    pox2 = set(tagset.split(' ')[0] for tagset in tagsets2)
                    if 'ПРЕДЛОГ' in pox2:
                        # Предлоги (всякие НАД и ПОД) не могут быть последними словами в строке
                        continue

                    pox2 = pox2.intersection(pox1)
                    if pox2:
                        score *= 0.5

                    score *= self.word_p.get(word2, 0.0)

                    if self.w2v_model is not None:
                        v2 = self.w2v_model[word2]
                        c = v_cosine(v1, v2)
                        score *= c

                    res.append((word2, score))

        return sorted(res, key=lambda z: -z[1])

    def save_pickle(self, dir):
        with open(os.path.join(dir, 'rselector.pkl'), 'wb') as f:
            pickle.dump(self.word2rhymes, f)
            pickle.dump(self.word_p, f)

    def load_pickle(self, dir):
        with open(os.path.join(dir, 'rselector.pkl'), 'rb') as f:
            self.word2rhymes = pickle.load(f)
            self.word_p = pickle.load(f)


if __name__ == '__main__':
    tmp_dir = '../../tmp'
    models_dir = '../../models'

    accents = Accents()
    accents.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
    accents.after_loading(stress_model_dir=os.path.join(tmp_dir, 'stress_model'))

    tokenizer = rutokenizer.Tokenizer()
    tokenizer.load()

    gren = ruword2tags.RuWord2Tags()
    gren.load()

    fasttext_model = None  #fasttext.load_model("/home/inkoziev/polygon/w2v/fasttext.CBOW=1_WIN=5_DIM=64")

    rselector = RhymeSelector(accents, fasttext_model, gren, tokenizer)
    rselector.load_udpipe(os.path.join(models_dir, 'udpipe_syntagrus.model'))
    rselector.initialize_from_corpus(os.path.join(tmp_dir, 'poetry_corpus.txt'))
    rselector.save_pickle(tmp_dir)

    for word1 in ['мною', 'интеллект', 'упасут', 'матильда']:
        sx = []
        for word2, sim in rselector.get_rhymes(word1)[:5]:
            sx.append('{}({:<6.2g})'.format(word2, sim))
        print('{} => {}'.format(word1, ' '.join(sx)))


