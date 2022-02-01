import io
import os
import logging
import pickle

from poetry.utils import decode_pos


class Word2Lemmas(object):
    def __init__(self):
        self.lemmas = None
        self.forms = None
        self.noun_lemmas = None
        self.adj_lemmas = None
        self.verb_lemmas = None
        self.adv_lemmas = None

    def load(self, path, all_words=None):
        logging.info('Loading lexicon from "%s"', path)
        self.lemmas = dict()
        self.forms = dict()
        self.noun_lemmas = set()
        self.adj_lemmas = set()
        self.verb_lemmas = set()
        self.adv_lemmas = set()

        with io.open(path, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split('\t')
                if len(tx) == 4:
                    form = tx[0].replace(u' - ', u'-').lower()
                    if all_words is None or form in all_words:
                        lemma = tx[1].replace(u' - ', u'-').lower()
                        pos = decode_pos(tx[2])

                        if form not in self.forms:
                            self.forms[form] = [lemma]
                        else:
                            self.forms[form].append(lemma)

                        k = lemma+'|'+pos
                        if k not in self.lemmas:
                            self.lemmas[k] = {form}
                        else:
                            self.lemmas[k].add(form)

                        if pos == 'СУЩЕСТВИТЕЛЬНОЕ':
                            self.noun_lemmas.add(lemma)
                        elif pos == 'ПРИЛАГАТЕЛЬНОЕ':
                            self.adj_lemmas.add(lemma)
                        elif pos == 'ГЛАГОЛ':
                            self.verb_lemmas.add(lemma)
                        elif pos == 'НАРЕЧИЕ':
                            self.adv_lemmas.add(lemma)

        logging.info('Lexicon loaded: %d lemmas, %d wordforms', len(self.lemmas), len(self.forms))

    def get_lemma(self, word):
        if word in self.forms:
            return self.forms[word][0]
        else:
            return word

    def get_forms(self, lemma, part_of_speech):
        k = lemma + '|' + part_of_speech
        if k in self.lemmas:
            return self.lemmas[k]
        else:
            return [lemma]

    def save_pickle(self, fpath):
        with open(fpath, 'wb') as f:
            pickle.dump(self.lemmas, f)
            pickle.dump(self.forms, f)
            pickle.dump(self.noun_lemmas, f)
            pickle.dump(self.adj_lemmas, f)
            pickle.dump(self.verb_lemmas, f)
            pickle.dump(self.adv_lemmas, f)

    def load_pickle(self, fpath):
        with open(fpath, 'rb') as f:
            self.lemmas = pickle.load(f)
            self.forms = pickle.load(f)
            self.noun_lemmas = pickle.load(f)
            self.adj_lemmas = pickle.load(f)
            self.verb_lemmas = pickle.load(f)
            self.adv_lemmas = pickle.load(f)


if __name__ == '__main__':
    data_dir = '../../data'
    output_dir = '../../tmp'

    lexicon = Word2Lemmas()
    lexicon.load(os.path.join(data_dir, 'poetry', 'dict', 'word2lemma.dat'))
    lexicon.save_pickle(os.path.join(output_dir, 'word2lemmas.pkl'))
