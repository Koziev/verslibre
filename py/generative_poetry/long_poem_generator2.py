import os
import json
import logging
import traceback
import warnings
import collections
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn
import transformers
import transformers.generation_utils


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
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                        positive_words=None, negative_words=None, max_len=256):
        global logits_booster

        prompt_text = "<s> " + context + ' $'
        #prompt_text = context + ' $'
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


class LongPoemGeneratorCore2(object):
    def __init__(self, gpt_name):
        self.gpt_name = gpt_name
        self.poem_generator = None
        self.parser = None
        self.accents = None
        self.aligner = None

    def load(self, models_dir, data_dir, tmp_dir):
        self.poem_generator = RugptGenerator()
        self.poem_generator.load(os.path.join(models_dir, self.gpt_name))

        self.parser = UdpipeParser()
        self.parser.load(models_dir)

        self.accents = Accents()
        self.accents.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
        self.accents.after_loading(stress_model_dir=os.path.join(tmp_dir, 'stress_model'))

        self.aligner = PoetryStressAligner(self.parser, self.accents, os.path.join(data_dir, 'poetry', 'dict'))

    def generate_poems(self, topic, genre=None, emotion_token=None, num_return_sequences=10, temperature=1.0, top_p=0.5, top_k=30,
                       score_threshold=0.20, penalty_alpha=0.0, typical_p=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0):
        try:
            if genre:
                seed = arabize(break_to_syllables(self.parser, self.accents, genre + ' , ' + topic))
            elif emotion_token is not None:
                seed = arabize(break_to_syllables(self.parser, self.accents, topic))
                if emotion_token:
                    seed += ' ' + emotion_token
            elif genre is None and topic is not None:
                seed = arabize(break_to_syllables(self.parser, self.accents, topic))
            else:
                raise NotImplementedError()

            poems = self.poem_generator.generate_output(seed,
                                                        num_return_sequences=num_return_sequences,
                                                        temperature=temperature,
                                                        top_p=top_p,
                                                        top_k=top_k,
                                                        penalty_alpha=penalty_alpha,
                                                        typical_p=typical_p,
                                                        repetition_penalty=repetition_penalty,
                                                        no_repeat_ngram_size=no_repeat_ngram_size,
                                                        )
        except Exception as ex:
            logging.error(ex)
            return []

        threshold_score = 0.1
        ranked_poems = []
        for ipoem, poem in enumerate(poems):
            lines = [decode_line2(line) for line in poem.split('<nl>') if len(line) > 0]
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
