import collections


def ngrams(s, n):
    return set(''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1 & shingles2))/float(len(shingles1 | shingles2) + 1e-6)


class Antiplagiat:
    def __init__(self):
        self.doc2index = dict()
        self.index2doc = dict()
        self.shingle2docs = collections.defaultdict(set)

    def add_document(self, doc_text):
        i = self.doc2index.get(doc_text)
        if i is None:
            i = len(self.doc2index)
            self.doc2index[doc_text] = i
            self.index2doc[i] = doc_text

            for c1, c2, c3 in zip(doc_text, doc_text[1:], doc_text[2:]):
                c123 = c1+c2+c3
                self.shingle2docs[c123].add(i)

    def find_nearest(self, doc_text, n):
        idoc2hits = collections.defaultdict(set)
        for c1, c2, c3 in zip(doc_text, doc_text[1:], doc_text[2:]):
            c123 = c1 + c2 + c3
            for idoc in self.shingle2docs.get(c123, []):
                idoc2hits[idoc].add(c123)

        doc_hits = [(idoc, len(hits)) for idoc, hits in idoc2hits.items()]
        doc_hits = sorted(doc_hits, key=lambda z: -z[1])[:n]
        return [self.index2doc[idoc] for idoc, _ in doc_hits]

    def score(self, doc_text):
        similar_docs = self.find_nearest(doc_text, 100)
        max_sim = 0.0
        for doc in similar_docs:
            j = jaccard(doc_text, doc, 3)
            max_sim = max(max_sim, j)
        return max_sim
