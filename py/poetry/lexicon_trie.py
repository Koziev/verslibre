# -*- coding: utf-8 -*-
'''
TRIE для лексикона, для быстрого поиска похожих слов.
'''

from __future__ import print_function

import codecs
import logging
import os
import pickle


# Keep some interesting statistics
NodeCount = 0


# The Trie data structure keeps a set of words, organized with one node for
# each letter. Each node has a branch for each letter that may follow it in the
# set of words.
class TrieNode:
    def __init__(self):
        self.word = None
        self.children = {}

        global NodeCount
        NodeCount += 1

    def insert(self, word):
        node = self
        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode()

            node = node.children[letter]

        node.word = word


# --------------------------------------------------------------

# The search function returns a list of all words that are less than the given
# maximum distance from the target word
def search(word, trie, maxCost):
    # build first row
    currentRow = range(len(word) + 1)

    results = []

    # recursively search each branch of the trie
    for letter in trie.children:
        searchRecursive(trie.children[letter], letter, word, currentRow,
                        results, maxCost)

    return results


# This recursive helper is used by the search function above. It assumes that
# the previousRow has been filled in already.
def searchRecursive(node, letter, word, previousRow, results, maxCost):
    columns = len(word) + 1
    currentRow = [previousRow[0] + 1]

    # Build one row for the letter, with a column for each letter in the target
    # word, plus one for the empty string at column 0
    for column in xrange(1, columns):

        insertCost = currentRow[column - 1] + 1
        deleteCost = previousRow[column] + 1

        if word[column - 1] != letter:
            replaceCost = previousRow[column - 1] + 1
        else:
            replaceCost = previousRow[column - 1]

        currentRow.append(min(insertCost, deleteCost, replaceCost))

    # if the last entry in the row indicates the optimal cost is less than the
    # maximum cost, and there is a word in this trie node, then add it.
    if currentRow[-1] <= maxCost and node.word != None:
        results.append((node.word, currentRow[-1]))

    # if any entries in the row are less than the maximum cost, then
    # recursively search each branch of the trie
    if min(currentRow) <= maxCost:
        for letter in node.children:
            searchRecursive(node.children[letter], letter, word, currentRow, results, maxCost)


# --------------------------------------------------------------------------------

def find_nbest(word, trie, ntop):
    word_len = float(len(word))
    results = search(word, trie, len(word))
    return [(w, 1.0 - float(p) / word_len) for (w, p) in sorted(results, key=lambda z: z[1])[:ntop]]



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

    trie = TrieNode()

    data_folder = '../data'
    grpath = os.path.join(data_folder, 'dict/word2tags.dat')
    trie_pickle_path = '../tmp/trie.dat'

    logging.info(u'Building TRIE using wordforms from {}'.format(grpath))
    with codecs.open(grpath, 'r', 'utf-8') as rdr:
        for line in rdr:
            tx = line.strip().split('\t')
            if len(tx) == 4:
                word = tx[0].replace(u' - ', u'-')
                if len(word) < 15:
                    trie.insert(word)

    logging.info('Done, {} nodes in trie'.format(NodeCount))

    logging.info(u'Storing TRIE in {}'.format(trie_pickle_path))
    with open(trie_pickle_path, 'wb') as f:
        pickle.dump(trie, f)
