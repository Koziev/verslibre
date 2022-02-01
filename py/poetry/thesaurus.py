import os
import pickle
import io
import collections
import logging

import networkx

from poetry.utils import decode_pos


class Thesaurus:
    def __init__(self):
        self.word2links = collections.defaultdict(list)
        self.word2oppos = None
        self.word2cognates = None

    def load(self, thesaurus_path):
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


if __name__ == "__main__":
    data_dir = '../../data'
    output_dir = '../../tmp'

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

    thesaurus = Thesaurus()
    thesaurus.load(os.path.join(data_dir, 'poetry', 'dict', 'links.csv'))

    with open(os.path.join(output_dir, 'thesaurus.pkl'), 'wb') as f:
        pickle.dump(thesaurus, f)
