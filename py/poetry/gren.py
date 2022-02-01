# -*- coding: utf-8 -*-
"""
Словари (кроме фонетического) для генератора-шаблонизатора стихов.
При запуске этот скрипт генерирует pickle-файл со всеми словарями.
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import os
import pickle
import io
import collections
import logging

import networkx

from poetry.utils import decode_pos
from poetry.corpus_analyser import CorpusWords
from poetry.word2lemmas import Word2Lemmas


class Thesaurus:
    def __init__(self):
        self.word2links = collections.defaultdict(list)
        self.word2oppos = None
        self.word2cognates = None

    def load(self, thesaurus_path, all_words):
        logging.info('Loading thesaurus from "%s"', thesaurus_path)
        with io.open(thesaurus_path, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split('\t')
                if len(tx) == 5:
                    word1 = tx[0].replace(' - ', '-').lower().replace('ё', 'е')
                    pos1 = decode_pos(tx[1])
                    word2 = tx[2].replace(' - ', '-').lower().replace('ё', 'е')
                    pos2 = decode_pos(tx[3])
                    relat = tx[4]

                    if relat in (u'в_класс', u'член_класса'):
                        continue

                    k1 = (word1, pos1)
                    k2 = (word2, pos2)

                    if k1 != k2:
                        self.word2links[k1].append((word2, pos2, relat))

        logging.info('%d items in thesaurus loaded', len(self.word2links))

        # Подготовка поисковых графов после загрузки из pickle
        cognate_rels = 'в_ся без_ся в_сущ в_прил в_деепричастие в_наречие в_несов в_сов уменьш_ласк нейтрал груб sex_synonym'.split()
        oppos_rels = 'antonym opposite'.split()

        graf_cognates = networkx.Graph()
        all_nodes = set()
        for (word1, pos1), links in self.word2links.items():
            for word2, pos2, rel in links:
                k1 = (word1, pos1)
                k2 = (word2, pos2)
                if rel in cognate_rels:
                    graf_cognates.add_edge(k1, k2)
                all_nodes.add(k1)
                all_nodes.add(k2)

        # Раскроем списки однокоренных слов
        print('Building cognate groups...')
        self.word2cognates = dict()
        for node1 in all_nodes:
            if node1 in graf_cognates:
                self.word2cognates[node1] = list(networkx.ego_graph(graf_cognates, n=node1, radius=3))
        print('{} cognate groups with {} items in total'.format(len(self.word2cognates), sum(map(len, self.word2cognates.values()))))

        # Теперь раскроем список контрастных слов (с учетом однокоренных!)
        print('Building opposition groups...')
        self.word2oppos = collections.defaultdict(set)
        for (word1, pos1), links in self.word2links.items():
            for word2, pos2, rel in links:
                if rel in oppos_rels:
                    node1 = (word1, pos1)
                    nodes1 = set([node1])
                    if node1 in graf_cognates:
                        nodes1.update(networkx.ego_graph(graf_cognates, n=node1, radius=3))

                    node2 = (word2, pos2)
                    nodes2 = set([node2])
                    if node2 in graf_cognates:
                        nodes2.update(networkx.ego_graph(graf_cognates, n=node2, radius=3))

                    # Теперь перелинковываем слова из первого списка со словами из второго
                    for node11 in nodes1:
                        for node22 in nodes2:
                            if node22 not in nodes1:
                                self.word2oppos[node11].add(node22)
                                self.word2oppos[node22].add(node11)

        print('{} opposition groups with {} items in total'.format(len(self.word2oppos), sum(map(len, self.word2oppos.values()))))

    def get_linked(self, word1, p_o_s1):
        return self.word2links.get((word1, p_o_s1), [])

    def after_loading(self):
        pass

    def get_cognate_lemmas(self, lemma, p_o_s):
        # Возвращает все однокоренные слова
        return self.word2cognates.get((lemma, p_o_s), [])

    def are_cognates(self, lemma1, pos1, lemma2, pos2):
        k1 = (lemma1, pos1)
        k2 = (lemma2, pos2)
        if k1 in self.word2cognates and k2 in self.word2cognates[k1]:
            return True

        if k2 in self.word2cognates and k1 in self.word2cognates[k2]:
            return True

        return False

    def are_contrast(self, lemma1, pos1, lemma2, pos2):
        k1 = (lemma1, pos1)
        k2 = (lemma2, pos2)
        if k1 in self.word2oppos and k2 in self.word2oppos[k1]:
            return True

        if k2 in self.word2oppos and k1 in self.word2oppos[k2]:
            return True

        return False


class GrammarDict:
    def __init__(self):
        pass

    def split_tag(self, tag):
        return tuple(tag.split(':'))

    def split_tags(self, tags_str):
        return [self.split_tag(tag) for tag in tags_str.split(' ')]

    def is_good(self, tags_str):
        # Исключаем краткие формы прилагательных в среднем роде, так как
        # они обычно омонимичны с более употребимыми наречиями.
        return u'КРАТКИЙ:1 ПАДЕЖ:ИМ РОД:СР' not in tags_str

    def load(self, path, all_words):
        #self.nouns = set()
        #self.adjs = set()
        #self.adverbs = set()
        #self.verbs = set()
        self.word2pos = dict()
        self.word2tags = dict()
        self.tagstr2id = dict()
        self.tagsid2list = dict()
        logging.info(u'Loading morphology information from {}'.format(path))

        # первый проход нужен, чтобы собрать список слов, которые будут неоднозначно распознаваться.
        ambiguous_words = set()
        if False:
            with io.open(path, 'r', encoding='utf-8') as rdr:
                for line in rdr:
                    tx = line.strip().split('\t')
                    if len(tx) == 4:
                        word = tx[0].replace(u' - ', u'-')
                        lemma = tx[2]
                        tags_str = tx[3]
                        if self.is_good(tags_str):
                            if word in all_words:
                                pos0 = tx[1]
                                pos = decode_pos(pos0)
                                key = pos+u'|'+lemma
                                if word in self.word2pos and self.word2pos[word] != key:
                                    ambiguous_words.add(word)
                                elif word not in ambiguous_words:
                                    self.word2pos[word] = key

        self.word2pos = dict()

        # Второй проход - сохраняем информацию для всех слов, кроме вошедших
        # в список неоднозначных.
        with io.open(path, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split('\t')
                if len(tx) == 5:
                    word = tx[0].replace(u' - ', u'-')
                    tags_str = tx[3]
                    if self.is_good(tags_str):
                        if word in ambiguous_words:
                            continue

                        tags_str = tags_str.replace(u'ПЕРЕЧИСЛИМОСТЬ:ДА', u'')\
                            .replace(u'ПЕРЕЧИСЛИМОСТЬ:НЕТ', u'')\
                            .replace(u'ПЕРЕХОДНОСТЬ:ПЕРЕХОДНЫЙ', u'')\
                            .replace(u'ПЕРЕХОДНОСТЬ:НЕПЕРЕХОДНЫЙ', u'')

                        if word in all_words:
                            pos0 = tx[1]
                            if pos0 == 'ИНФИНИТИВ':
                                tags_str += ' НАКЛОНЕНИЕ:ИНФИНИТИВ'
                            elif pos0 == 'ДЕЕПРИЧАСТИЕ':
                                tags_str += ' НАКЛОНЕНИЕ:ДЕЕПРИЧАСТИЕ'

                            pos = decode_pos(pos0)
                            self.add_word(word, pos, tags_str)

            # отдельно добавляем фиктивную информацию для местоимений в 3м лице, чтобы
            # они могли меняться на существительные
            s_noun = u'СУЩЕСТВИТЕЛЬНОЕ'
            self.add_word(u'никто', s_noun, u'ПАДЕЖ:ИМ ЧИСЛО:ЕД РОД:МУЖ')
            self.add_word(u'ничто', s_noun, u'ПАДЕЖ:ИМ ЧИСЛО:ЕД РОД:МУЖ')
            self.add_word(u'он', s_noun, u'ПАДЕЖ:ИМ ЧИСЛО:ЕД РОД:МУЖ')
            self.add_word(u'она', s_noun, u'ПАДЕЖ:ИМ ЧИСЛО:ЕД РОД:ЖЕН')
            self.add_word(u'оно', s_noun, u'ПАДЕЖ:ИМ ЧИСЛО:ЕД РОД:СР')
            self.add_word(u'они', s_noun, u'ПАДЕЖ:ИМ ЧИСЛО:МН')
            self.add_word(u'тебе', s_noun, u'ПАДЕЖ:ДАТ ЧИСЛО:ЕД')
            self.add_word(u'тебя', s_noun, u'ПАДЕЖ:ВИН ЧИСЛО:ЕД')
            self.add_word(u'тобой', s_noun, u'ПАДЕЖ:ТВОР ЧИСЛО:ЕД')
            self.add_word(u'тобою', s_noun, u'ПАДЕЖ:ТВОР ЧИСЛО:ЕД')
            self.add_word(u'мне', s_noun, u'ПАДЕЖ:ДАТ ЧИСЛО:ЕД')
            self.add_word(u'меня', s_noun, u'ПАДЕЖ:ВИН ЧИСЛО:ЕД')
            self.add_word(u'мной', s_noun, u'ПАДЕЖ:ТВОР ЧИСЛО:ЕД')
            self.add_word(u'мною', s_noun, u'ПАДЕЖ:ТВОР ЧИСЛО:ЕД')
            self.add_word(u'нам', s_noun, u'ПАДЕЖ:ДАТ ЧИСЛО:ЕД')
            self.add_word(u'нами', s_noun, u'ПАДЕЖ:ТВОР ЧИСЛО:ЕД')
            self.add_word(u'вам', s_noun, u'ПАДЕЖ:ДАТ ЧИСЛО:ЕД')
            self.add_word(u'вами', s_noun, u'ПАДЕЖ:ТВОР ЧИСЛО:ЕД')
            self.add_word(u'вас', s_noun, u'ПАДЕЖ:ВИН ЧИСЛО:ЕД')
            logging.info('Total number of wordforms={}/{}'.format(len(self.word2pos), len(self.word2tags)))

    def add_word(self, word, pos, tags_str0):
        self.word2pos[word] = pos
        #if pos == u'СУЩЕСТВИТЕЛЬНОЕ':
        #    self.nouns.add(word)
        #elif pos == u'ПРИЛАГАТЕЛЬНОЕ':
        #    self.adjs.add(word)
        #elif pos == u'ГЛАГОЛ':
        #    self.verbs.add(word)
        #elif pos == u'НАРЕЧИЕ':
        #    self.adverbs.add(word)

        tags_str = u'ЧАСТЬ_РЕЧИ:'+pos + u' ' + tags_str0
        if '  ' in tags_str:
            tags_str = tags_str.replace('  ', ' ')

        # формы прилагательных в винительном падеже дополняем тегами ОДУШ:ОДУШ и ОДУШ:НЕОДУШ, если не указан тег ОДУШ:НЕОДУШ
        if pos == u'ПРИЛАГАТЕЛЬНОЕ' and u'ПАДЕЖ:ВИН' in tags_str and u'ОДУШ:' not in tags_str:
            tags_str += u' ОДУШ:ОДУШ ОДУШ:НЕОДУШ'

        if tags_str not in self.tagstr2id:
            tags_id = len(self.tagstr2id)
            self.tagstr2id[tags_str] = tags_id
            self.tagsid2list[tags_id] = self.split_tags(tags_str)
        else:
            tags_id = self.tagstr2id[tags_str]

        if word not in self.word2tags:
            self.word2tags[word] = [tags_id]
        else:
            self.word2tags[word].append(tags_id)

    def __contains__(self, word):
        return word in self.word2pos

    def get_pos(self, word):
        return self.word2pos.get(word)

    def get_word_tagsets(self, word):
        tagsets = []
        for tagset_id in self.word2tags.get(word, []):
            tagsets.append(self.tagsid2list[tagset_id])
        return tagsets


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

    data_folder = '../../data/poetry'
    tmp_folder = '../../data/poetry'
    word2lemmas_path = 'dict/word2lemma.dat'
    dict_pickle_path = '../../tmp/poetry_dict.pickle'

    corpus = CorpusWords()
    corpus.load()

    thesaurus = Thesaurus()
    thesaurus.load(os.path.join(data_folder, 'dict/links.csv'), corpus)

    assert(thesaurus.are_cognates('боль', 'СУЩЕСТВИТЕЛЬНОЕ', 'болеть', 'ГЛАГОЛ'))
    assert(thesaurus.are_cognates('боль', 'СУЩЕСТВИТЕЛЬНОЕ', 'болевший', 'ПРИЛАГАТЕЛЬНОЕ'))
    assert(thesaurus.are_cognates('боль', 'СУЩЕСТВИТЕЛЬНОЕ', 'проболеть', 'ГЛАГОЛ'))

    assert(thesaurus.are_contrast('радостный', 'ПРИЛАГАТЕЛЬНОЕ', 'грустный', 'ПРИЛАГАТЕЛЬНОЕ'))
    assert(thesaurus.are_contrast('радость', 'СУЩЕСТВИТЕЛЬНОЕ', 'грусть', 'СУЩЕСТВИТЕЛЬНОЕ'))
    assert(thesaurus.are_contrast('радостно', 'НАРЕЧИЕ', 'грустный', 'ПРИЛАГАТЕЛЬНОЕ'))
    assert(thesaurus.are_contrast('заболеть', 'ГЛАГОЛ', 'выздороветь', 'ГЛАГОЛ'))

    lexicon = Word2Lemmas()
    lexicon.load(os.path.join(data_folder, word2lemmas_path), corpus)

    grdict = GrammarDict()
    grdict.load(os.path.join(data_folder, 'dict/word2tags.dat'), corpus)

    # НАЧАЛО ОТЛАДКИ
    if False:
        with open(dict_pickle_path, 'wb') as f:
            pickle.dump(thesaurus, f)

        with open(dict_pickle_path, 'rb') as f:
            dbg = pickle.load(f)

        exit(0)
    # КОНЕЦ ОТЛАДКИ


    logging.info('Storing dictionary parts to "%s"', dict_pickle_path)
    with open(dict_pickle_path, 'wb') as f:
        pickle.dump(thesaurus, f)
        pickle.dump(lexicon, f)
        pickle.dump(grdict, f)

    # НАЧАЛО ОТЛАДКИ
    if True:
        with open(dict_pickle_path, 'rb') as f:
            thesaurus2 = pickle.load(f)
            lexicon2 = pickle.load(f)
            grdict2 = pickle.load(f)

        exit(0)
    # КОНЕЦ ОТЛАДКИ

    print('All done.')

