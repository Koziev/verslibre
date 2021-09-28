"""
Модель для генерации хайку
"""

import os
import logging.handlers

#import numpy as np
#import fasttext

import rutokenizer

from rugpt_generator import RugptGenerator


def v_cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom > 0:
        return np.dot(a, b) / denom
    else:
        return 0


class TopicRanker:
    def __init__(self, tokenizer):
        self.fasttext_model = None
        self.tokenizer = tokenizer

    def load(self, fasttext_dir):
        self.fasttext_model = fasttext.load_model(fasttext_dir)

    def vectorize_phrase(self, phrase):
        vx = []
        for t in self.tokenizer.tokenize(phrase):
            vx.append(self.fasttext_model[t])

        return np.mean(vx, axis=0)

    def rerank(self, topic_phrase, phrases):
        topic_v = self.vectorize_phrase(topic_phrase)
        cosx = []
        for phrase in phrases:
            v = self.vectorize_phrase(phrase.replace('|', ' '))
            c = v_cosine(topic_v, v)
            cosx.append((c, phrase))

        phrases = sorted(cosx, key=lambda f: -f[0])
        phrases = [z[1] for z in phrases]
        return phrases


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.ERROR)

    tokenizer = rutokenizer.Tokenizer()
    tokenizer.load()

    #ranker = TopicRanker(tokenizer)
    #ranker.load("/home/inkoziev/polygon/w2v/fasttext.CBOW=1_WIN=5_DIM=64")

    models_dir = '/text_generator/models'

    haiku_generator = RugptGenerator()
    haiku_generator.load(os.path.join(models_dir, 'rugpt_haiku_generator'))

    caption_generator = RugptGenerator()
    caption_generator.load(os.path.join(models_dir, 'rugpt_caption_generator'))

    while True:
        q = input(':> ').strip()
        if q:
            context = q

            px = haiku_generator.generate_output(q)
            #px = ranker.rerank(q, px)
            for ipoem, p in enumerate(px, start=1):
                if '|' in p:
                    captions = caption_generator.generate_output(p)
                    caption = captions[0]
                    print('HAIKU #{} for seed={}:'.format(ipoem, q))
                    print('--- {} ---\n'.format(caption))
                    print(p.replace(' | ', '\n'))
                    print('\n\n')

            print('')
