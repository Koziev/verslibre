import io
import pickle
import os
import collections
import math
import argparse

import numpy as np
import pyconll
import rutokenizer
import ruword2tags
from ufal.udpipe import Model, Pipeline, ProcessingError
import fasttext

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

from poetry.phonetic import Accents, rhymed
from select_rhymed_words import RhymeSelector
from rugpt_generator import RugptGenerator


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


def get_last_word(tokens):
    if tokens:
        last_token = tokens[-1]
        if last_token in [',', '-', '.', '!', '?', ';']:
            return get_last_word(tokens[:-1])
        else:
            return last_token.lower()
    else:
        return ''


def is_good_poem(poem_text):
    last_words = collections.Counter()
    for poem_line in [l.strip() for l in poem_text.split('|')]:
        tokens = tokenizer.tokenize(poem_line)
        last_words[get_last_word(tokens)] += 1

    # если одно из слов встречается минимум дважды в конце строки (не считая пунктуации),
    # то вернем False.
    return last_words.most_common(1)[0][1] == 1



sentinel0 = '<extra_id_0>'
sentinel1 = '<extra_id_1>'
eos_token = '</s>'


def extract_spans(t5_output):
    span0 = None
    span1 = None

    if sentinel0 in t5_output and sentinel1 in t5_output:
        pos0 = t5_output.index(sentinel0)
        pos1 = t5_output.index(sentinel1)
        if eos_token in t5_output:
            eos_pos = t5_output.index(eos_token)
        else:
            eos_pos = len(t5_output)

        span0 = t5_output[pos0 + len(sentinel0): pos1].strip()
        span1 = t5_output[pos1 + len(sentinel1): eos_pos].strip()

    return span0, span1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verslibre generator v.2')
    parser.add_argument('--topic', type=str)
    parser.add_argument('--tmp_dir', default='../../tmp', type=str)
    parser.add_argument('--models_dir', default='../../models', type=str)

    args = parser.parse_args()
    topic = args.topic
    tmp_dir = args.tmp_dir
    models_dir = args.models_dir

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Гибридный ударятор - словарь ударений + нейросетевая модель для oov слов
    accents = Accents()
    accents.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
    accents.after_loading(stress_model_dir=os.path.join(tmp_dir, 'stress_model'))

    tokenizer = rutokenizer.Tokenizer()
    tokenizer.load()

    gren = ruword2tags.RuWord2Tags()
    gren.load()

    print('Loading the word2vec model...')
    fasttext_model = None  #fasttext.load_model("/home/inkoziev/polygon/w2v/fasttext.CBOW=1_WIN=5_DIM=64")

    # Модель подбора рифмующихся слов
    rselector = RhymeSelector(accents, fasttext_model, gren, tokenizer)
    rselector.load_udpipe(os.path.join(models_dir, 'udpipe_syntagrus.model'))
    #rselector.initialize_from_corpus(os.path.join(tmp_dir, 'poetry_corpus.txt'))
    rselector.load_pickle(tmp_dir)

    # Генератор тела стиха (первых 2х строк)
    print('Loading GPT poetry generator...')
    poem_generator = RugptGenerator()
    poem_generator.load(os.path.join(models_dir, 'rugpt_poem_generator'))

    # Генератор заголовка стиха по тексту стиха
    print('Loading GPT caption generator...')
    caption_generator = RugptGenerator()
    caption_generator.load(os.path.join(models_dir, 'rugpt_caption_generator'))

    # Генератор двух строк, заканчивающихся заданными рифмами, на базе T5
    print('Loading the T5 generator...')
    model_name = 'sberbank-ai/ruT5-large'
    t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
    t5_config = T5Config.from_pretrained(model_name)
    t5_model = T5ForConditionalGeneration(t5_config)
    t5_model.load_state_dict(torch.load(os.path.join(tmp_dir, 'rut5_for_poem_completion.pt'), map_location=device))

    while True:
        if topic:
            q = topic
        else:
            q = input(':> ').strip()

        if q:
            context = q

        print('Start generating for topic="{}"'.format(context))
        px = poem_generator.generate_output(context)
        for ipoem, p in enumerate(px, start=1):
            if '|' in p:
                if is_good_poem(p):
                    # Напечатам получившийся "верлибр"
                    poem_lines = [z.strip() for z in p.split('|')]
                    captions = caption_generator.generate_output(p)
                    caption = captions[0]
                    print('\nVERS-LIBRE #{} for seed={}:'.format(ipoem, context))
                    print('--- {} ---\n'.format(caption))
                    for l in poem_lines:
                        print(l)

                    # Берем первые 2 строки
                    head_lines = poem_lines[:2]

                    # Подбираем рифмы к первым двум строкам
                    rhymes1 = rselector.get_rhymes(get_last_word(tokenizer.tokenize(head_lines[0])))[:2]
                    rhymes2 = rselector.get_rhymes(get_last_word(tokenizer.tokenize(head_lines[1])))[:2]

                    for rhyme1, score1 in rhymes1:
                        for rhyme2, score2 in rhymes2:
                            t5_input = '{} ; {} ; <extra_id_0> {} ; <extra_id_1> {}'.format(poem_lines[0], poem_lines[1], rhyme1, rhyme2)
                            input_ids = t5_tokenizer(t5_input, return_tensors='pt').input_ids
                            #print('Running T5 generator...')
                            out_ids = t5_model.generate(input_ids=input_ids,
                                                        max_length=40,
                                                        eos_token_id=t5_tokenizer.eos_token_id,
                                                        early_stopping=True)

                            t5_output = t5_tokenizer.decode(out_ids[0][1:])
                            span0, span1 = extract_spans(t5_output)
                            if span0 and span1:
                                #print('span0={}'.format(span0))
                                #print('span1={}'.format(span1))

                                final_text = t5_input.replace(sentinel0, span0).replace(sentinel1, span1)

                                p = final_text.replace(';', '|')
                                captions = caption_generator.generate_output(p)
                                caption = captions[0]
                                print('\nPOEM #{} for seed={}:'.format(ipoem, context))
                                print('--- {} ---\n'.format(caption))
                                print(final_text.replace(' ; ', '\n'))
                                print('\n\n')

        if topic:
            break
