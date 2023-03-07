import os
import json
import logging
import traceback
import re
import warnings
import collections
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn
import transformers
import transformers.generation_utils

from nltk.stem.snowball import RussianStemmer

from generative_poetry.experiments.rugpt_with_stress.break_to_syllables import break_to_syllables
from generative_poetry.experiments.rugpt_with_stress.arabize import arabize
from generative_poetry.experiments.rugpt_with_stress.stressed_gpt_tokenizer import StressedGptTokenizer
from generative_poetry.whitespace_normalization import normalize_whitespaces
from generative_poetry.metre_classifier import get_syllables
from poetry.phonetic import Accents
from generative_poetry.udpipe_parser import UdpipeParser
from generative_poetry.poetry_alignment import PoetryStressAligner


upper_cyr = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'


logits_booster = None


def ngrams(s, n):
    return set(''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1 & shingles2))/float(len(shingles1 | shingles2) + 1e-6)


def decode_line2(line0, remove_stress_marks=True):
    out_words = []

    tokens = [z.strip() for z in line0.split('|')]
    for token in tokens:
        if remove_stress_marks:
            syllabs = token.replace('\u0301', '').split(' ')
        else:
            syllabs = token.split(' ')

        out_word = ''.join(syllabs[::-1])
        out_words.append(out_word)

    s = ' '.join(out_words[::-1])
    s = normalize_whitespaces(s)
    return s


class RugptGenerator:
    def __init__(self, device):
        self.device = device
        self.tokenizer = None
        self.model = None

    def load(self, model_dir):
        with open(os.path.join(model_dir, 'tokenizer_config.json'), 'r') as f:
            config = json.load(f)
            tokenizer_class = config['tokenizer_class']
            if tokenizer_class == 'BertTokenizer':
                self.tokenizer = transformers.BertTokenizer.from_pretrained(model_dir)
            elif tokenizer_class == 'StressedGptTokenizer':
                self.tokenizer = StressedGptTokenizer.from_pretrained(model_dir)
            else:
                raise NotImplementedError()

        self.model = transformers.GPT2LMHeadModel.from_pretrained(model_dir)

        self.model.tokenizer = self.tokenizer  # он нам понадобится внутри нашей версии sample()
        self.model.to(self.device)
        self.model.eval()

    def generate_output(self, context, num_return_sequences=10, temperature=1.0, top_k=30, top_p=0.40,
                        penalty_alpha=0.0, typical_p=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0,
                        positive_words=None, negative_words=None, max_len=384):
        global logits_booster

        # 27.12.2022 Если затравка пустая, то не будем добавлять токен-сепаратор $
        if context:
            prompt_text = "<s> " + context + ' $'
        else:
            prompt_text = "<s>"

        stop_token = "</s>"

        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        # 15-05-2022 НАЧАЛО ЭКСПЕРИМЕНТА с управлением генерацией через логиты
        # if positive_words is not None or negative_words is not None:
        #     positive_syllabic_ngrams = collections.defaultdict(list)
        #     if positive_words is not None:
        #         for word, score in positive_words.items():
        #             sx = ['|'] + [x.text for x in get_syllables(word)][::-1]
        #             if len(sx) > 1:
        #                 positive_syllabic_ngrams[sx[0]].append((sx[1], score))
        #                 if len(sx) > 2:
        #                     positive_syllabic_ngrams[(sx[0], sx[1])].append((sx[2], score))
        #
        #     negative_syllabic_ngrams = collections.defaultdict(list)
        #     for word, score in negative_words.items():
        #         sx = ['|'] + [x.text for x in get_syllables(word)][::-1]
        #         if len(sx) > 1:
        #             negative_syllabic_ngrams[sx[0]].append((sx[1], score))
        #             if len(sx) > 2:
        #                 negative_syllabic_ngrams[(sx[0], sx[1])].append((sx[2], score))
        #
        #     # может получится, что некоторые n-граммы входят и в позитивные, и в негативные.
        #     # такие нграммы мы просто исключим из списков, и не будем на них влиять.
        #     nx1 = set(positive_syllabic_ngrams.keys())
        #     nx2 = set(negative_syllabic_ngrams.keys())
        #     for k in nx1 & nx2:
        #         del positive_syllabic_ngrams[k]
        #         del negative_syllabic_ngrams[k]
        #
        #     logits_booster = ForceTopicWordsLogitsProcessor(positive_syllabic_ngrams,
        #                                                     negative_syllabic_ngrams,
        #                                                     self.tokenizer)
        # else:
        #     logits_booster = None
        # 15-05-2022 КОНЕЦ ЭКСПЕРИМЕНТА

        do_sample = True  #penalty_alpha == 0.0

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=max_len,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            typical_p=typical_p,
            #penalty_alpha=penalty_alpha,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            pad_token_id=0,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = set()
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()[encoded_prompt.shape[1]:]

            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            if stop_token in text:
                text = text[: text.find(stop_token)]

            generated_sequences.add(text.strip().replace('<pad>', '').strip())

        return list(generated_sequences)


def tokenize(s):
    return re.split(r'[.,!?\- ;:]', s)


class LongPoemGeneratorCore2(object):
    def __init__(self, device):
        self.device = device
        self.poem_generator = None
        self.parser = None
        self.accents = None
        self.aligner = None
        self.stemmer = None

    def load(self, gpt_model_path, models_dir, data_dir, tmp_dir):
        # Генеративная модель стихов
        self.poem_generator = RugptGenerator(self.device)
        self.poem_generator.load(gpt_model_path)

        # Транслятор для получения хороших затравок
        self.prompt_generator = RugptGenerator(self.device)
        self.prompt_generator.load(os.path.join(tmp_dir, 'prompt_generator_medium'))

        self.parser = UdpipeParser()
        self.parser.load(models_dir)

        self.accents = Accents()
        self.accents.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
        self.accents.after_loading(stress_model_dir=os.path.join(tmp_dir, 'stress_model'))

        self.aligner = PoetryStressAligner(self.parser, self.accents, os.path.join(data_dir, 'poetry', 'dict'))

        self.stemmer = RussianStemmer()

    def translate_prompt(self, user_prompt):
        prompts = []
        if user_prompt[0] in '!*@':
            # Если пользователь ввел затравку, начинающуюся символом !, то ее не транслируем, а прямо подаем в модель.
            prompts = [user_prompt[1:].strip()]
        else:
            try:
                a = self.aligner.align1([user_prompt])
                if a is not None:
                    signature = a.poetry_lines[0].stress_signature_str
                    if signature in ('0101010001', '0101010101'):
                        prompts = [user_prompt]
            except Exception as ex:
                logging.error(ex)

            if len(prompts) == 0:
                # Запускаем генеративную модель для получения затравок
                seed2 = arabize(break_to_syllables(self.parser, self.accents, user_prompt))
                #encoded_prompt = self.prompt_generator.tokenizer.encode('<s> ' + seed2 + ' $', add_special_tokens=False, return_tensors="pt")
                #encoded_prompt = encoded_prompt.to(self.device)

                awesome_prompts = self.prompt_generator.generate_output(context=seed2,
                                                                        num_return_sequences=20,
                                                                        temperature=1.0,
                                                                        top_p=0.8,
                                                                        top_k=50,
                                                                        #penalty_alpha=penalty_alpha,
                                                                        typical_p=None,
                                                                        )

                # Ранжируем полученные затравки, чтобы вверху оказались строки, содержащие максимальное
                # число слов исходной затравки.
                ranked_prompts = []
                for prompt0 in awesome_prompts:
                    try:
                        prompt = decode_line2(prompt0)
                        while prompt[-1] in '.,!/':
                            prompt = prompt[:-1].strip()

                        a = self.aligner.align1([prompt])
                        if a is not None and a.meter == 'ямб':
                            score = a.score * jaccard(prompt, user_prompt, 3)
                            ranked_prompts.append((prompt, score))
                    except Exception as ex:
                        print(ex)

                ranked_prompts = sorted(ranked_prompts, key=lambda z: z[1], reverse=True)
                prompts = [s[0] for s in ranked_prompts[:5]]

        if len(prompts) == 0:
            prompts = [user_prompt]

        return prompts

    def generate_poems(self, topic, num_return_sequences=10, temperature=1.0, top_p=0.5, top_k=30,
                       score_threshold=0.20, penalty_alpha=0.0, typical_p=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0):

        # Чтобы определить, когда останавливать генерацию новых вариантов, соберем
        # список ключевых слов из пользовательского запроса. Когда текст генерации
        # будет содержать эти ключевые слова - генерацию можно остановить.
        user_keywords = []
        try:
            for parsing in self.parser.parse_text(topic):
                for t in parsing:
                    if t.upos in ('NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV'):
                        user_keywords.append(self.stemmer.stem(t.lemma)[:4])  # берем первые 4 символа стема, чтобы сымитировать учет однокоренных
        except Exception as ex:
            logging.error(ex)

        if len(user_keywords) == 0:
            user_keywords = [word for word in tokenize(topic) if len(word) > 2]

        poems = []
        try:
            for prompt in self.translate_prompt(topic):
                seed = arabize(break_to_syllables(self.parser, self.accents, prompt))

                poems1 = self.poem_generator.generate_output(seed,
                                                            num_return_sequences=num_return_sequences,
                                                            temperature=temperature,
                                                            top_p=top_p,
                                                            top_k=top_k,
                                                            penalty_alpha=penalty_alpha,
                                                            typical_p=typical_p,
                                                            repetition_penalty=repetition_penalty,
                                                            no_repeat_ngram_size=no_repeat_ngram_size,
                                                            )

                # 05.03.2023 Если среди сгенерированных стихов попался текст, содержащий слова затравки - прекращаем генерацию.
                keyword_hit = False
                for poem in poems1:
                    lines = [decode_line2(line) for line in poem.split('<nl>') if len(line) > 0]
                    poems.append(lines)

                    for line in lines:
                        for keyword in user_keywords:
                            if re.search(r'\b'+keyword, line, flags=re.I) is not None:
                                keyword_hit = True
                                logging.debug('Keyword stem "%s" found in generation line "%s"', keyword, line)
                                break
                        if keyword_hit:
                            break
                if keyword_hit:
                    break
        except Exception as ex:
            #logging.error(ex)
            logging.error(traceback.format_exc())
            return []

        threshold_score = 0.1
        ranked_poems = []
        for ipoem, lines in enumerate(poems):
            try:
                a = self.aligner.align(lines, check_rhymes=True)
                if a is not None and a.score >= threshold_score:
                    score = a.score

                    if a.rhyme_scheme == 'AAAA':
                        # штрафуем за ситуацию, когда все строки между собой зарифмованы
                        score *= 0.5

                    if self.aligner.detect_repeating(a):
                        # Генерации с повторами штрафуем.
                        logging.warning('Repetition detected: %s', a.get_stressed_lines().replace('\n', '|'))
                        score = 0.5 * score

                    if self.aligner.detect_poor_poetry(a):
                        # штрафуем за бедную рифмовку
                        score *= 0.5

                    if score > score_threshold:
                        ranked_poems.append((lines, score))
            except Exception as ex:
                logging.error(ex)
                continue

        ranked_poems = sorted(ranked_poems, key=lambda z: -z[1])
        return ranked_poems
