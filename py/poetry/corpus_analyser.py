# -*- coding: utf-8 -*-
"""
Подготовка списка валидных слов - то есть тех слов, которые
встречаются в текстовом корпусе или в шаблонах. Это необходимо,
чтобы 1) исключить подстановку неестественно выглядящих словоформ ("любовями")
и 2) сократить объем словарей.
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import pickle
import io
import collections
import glob
import itertools
import os
import logging

from rutokenizer import Tokenizer
from poetry.poetry_template import PoetryTemplate
from poetry.utils import decode_pos
import poetry.template_tokenizer


def iter_files(root_folder, filename_mask):
    """
    Рекурсивно обходим все подкаталоги, начиная с указанной папки, перечисляем
    файлы в этих подкаталогах, удовлетворяющих заданной маске имени.
    :param root_folder: начальная папка
    :param filename_mask: маска имени, например u'*.txt'
    :return: итератор по именам файлов
    """
    cx = itertools.chain.from_iterable(glob.iglob(os.path.join(x[0], filename_mask)) for x in os.walk(root_folder))
    return cx


class CorpusWords:
    """Анализ большого текстового корпуса и сбор самых частотных слов"""
    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.storage_path = '../../tmp/all_words.dat'
        self.all_words = set()

    def analyze(self, corpora, thesaurus_path, grdict_path):
        tokenizer = Tokenizer()
        tokenizer.load()

        self.all_words = set()  # здесь накопим слова, которые будут участвовать в перефразировках.

        # Тезаурус содержит связи между леммами, соберем список этих лемм.
        thesaurus_entries = set()
        with io.open(thesaurus_path, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split('\t')
                if len(tx) == 5:
                    word1 = tx[0].replace(u' - ', u'-')
                    pos1 = decode_pos(tx[1])
                    word2 = tx[2].replace(u' - ', u'-')
                    pos2 = decode_pos(tx[3])

                    thesaurus_entries.add((word1, pos1))
                    thesaurus_entries.add((word2, pos2))

        # Теперь для всех лемм, упомянутых в тезаурусе, получим все грамматические формы.
        thesaurus_forms = set()
        with io.open(grdict_path, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split('\t')
                if len(tx) == 5:
                    word = tx[0].replace(u' - ', u'-')
                    pos = decode_pos(tx[1])
                    lemma = tx[2]

                    if (lemma, pos) in thesaurus_entries:
                        thesaurus_forms.add(word)

        self.all_words.update(thesaurus_forms)

        if True:
            # добавим слова из ассоциаций
            with io.open(r'../../data/poetry/dict/word2freq_wiki.dat', 'r', encoding='utf-8') as rdr:
                for line in rdr:
                    word = line.strip()
                    self.all_words.add(word)


        if False:
            # Теперь читаем текстовый корпус.
            self.word2freq = collections.Counter()
            for fname in corpora:
                logging.info(u'Reading corpus from "{}"'.format(fname))
                with io.open(fname, 'r', encoding='utf-8') as rdr:
                    for iline, line in enumerate(rdr):
                        phrase = line.strip()
                        words = tokenizer.tokenize(phrase)
                        self.word2freq.update(words)
                        #phrase = u' '.join(words)
                        # tfidf_corpus.add(phrase)
                        if iline > 2000000:
                            break

            self.all_words.update(w for (w, freq) in self.word2freq.items() if freq > self.min_freq)
            #logging.info('{} words with more than {} occurencies in corpus vocabulary'.format(len(self.all_words), self.min_freq))

        logging.info('Total number of words in corpus = %d', len(self.all_words))

    def __contains__(self, item):
        return item in self.all_words

    def save(self):
        logging.info(u'Storing frequent words in "%s"', self.storage_path)
        with open(self.storage_path, 'wb') as f:
            pickle.dump(self.all_words, f)

        if True:
            with io.open('../../tmp/all_words.txt', 'w', encoding='utf-8') as wrt:
                for word in sorted(self.all_words):
                    wrt.write(u'{}\n'.format(word))

    def load(self):
        logging.info(u'Loading frequent words from {}'.format(self.storage_path))
        with open(self.storage_path, 'rb') as f:
            self.all_words = pickle.load(f)

    def add_words_from_templates(self, templates_root_folder):
        tokenizer = poetry.template_tokenizer.TemplateTokenizer()
        for path in iter_files(templates_root_folder, '*.txt'):
            with io.open(path, 'r', encoding='utf-8') as rdr:
                for line in rdr:
                    tokens = tokenizer.tokenize(line.strip())
                    for token in tokens:
                        px = token.split('|')
                        word = px[0].lower()
                        self.all_words.add(word)

        return


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

    corpus = CorpusWords()

    data_folder = '../../data/poetry'

    # Большой текстовый корпус, в котором текст уже токенизирован и нормализован.
    #corpus1 = r'f:\Corpus\word2vector\ru\SENTx.corpus.w2v.txt'
    corpus1 = '/mnt/7383b08e-ace3-49d3-8991-5b9aa07d2596/Corpus/word2vector/ru/SENTx.corpus.w2v.txt'
    corpus.analyze([corpus1], os.path.join(data_folder, 'dict/links.csv'), os.path.join(data_folder, 'dict/word2tags.dat'))

    # Добавим также все слова из шаблонов, чтобы делать замены для
    # малочастотных слов, которые не встретятся в большой корпусе corpus1.
    corpus.add_words_from_templates(os.path.join(data_folder, 'templates'))

    corpus.save()
