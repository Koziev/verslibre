"""
15-12-2021 Введен сильный штраф за 2 подряд ударных слога
17-12-2021 Регулировка ударности некоторых наречий и частиц, удаление лишних пробелов вокруг дефиса при выводе строки с ударениями
18-12-2021 Коррекция пробелов вынесена в отдельный модуль whitespace_normalization
22-12-2021 Добавлена рифмовка AABB
28-12-2021 Добавлены еще штрафы за всякие нехорошие с точки зрения вокализма ситуации в строке, типа 6 согласных подряд в смежных словах "пёстр страх"
23-01-2022 Добавлен код для словосочетаний с вариативным ударением типа "пО лесу | по лЕсу"
26-01-2022 # Если слово допускает альтернативные ударения по списку и теги не позволяют сделать выбор, то берем первое ударение, а не бросаем исключение.
"""

import collections
import itertools
from functools import reduce
import os
import io
import math
import jellyfish
import hyperopt
import re
from hyperopt import hp, tpe, STATUS_OK, Trials

from poetry.phonetic import Accents, rhymed2
from generative_poetry.udpipe_parser import UdpipeParser
from generative_poetry.stanza_parser import StanzaParser
from generative_poetry.metre_classifier import get_syllables
from generative_poetry.experiments.rugpt_with_stress.arabize import arabize
from generative_poetry.whitespace_normalization import normalize_whitespaces


# алгоритм сэмплирования гиперпараметров
HYPEROPT_ALGO = tpe.suggest  # tpe.suggest OR hyperopt.rand.suggest


COEFF = dict()
COEFF['@68'] = 0.5
COEFF['@68_2'] = 0.95
COEFF['@71'] = 1.0
COEFF['@75'] = 0.9  # 0.8
COEFF['@77'] = 1.0
COEFF['@77_2'] = 1.0
COEFF['@79'] = 1.0
COEFF['@126'] = 0.98
COEFF['@225'] = 0.95  #0.9
COEFF['@143'] = 0.9


class WordStressVariant(object):
    def __init__(self, poetry_word, new_stress_pos, score):
        self.poetry_word = poetry_word
        self.new_stress_pos = new_stress_pos
        self.score = score

        self.stress_signature = []
        output = []
        n_vowels = 0
        for c in self.poetry_word.form:
            output.append(c)
            if c.lower() in 'уеыаоэёяию':
                n_vowels += 1
                if n_vowels == self.new_stress_pos:
                    output.append('\u0301')
                    self.stress_signature.append(1)
                else:
                    self.stress_signature.append(0)

        self.is_cyrillic = self.poetry_word.form[0].lower() in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        self.stressed_form = ''.join(output)

    def build_stressed(self, new_stress_pos):
        return WordStressVariant(self.poetry_word, new_stress_pos, self.score)

    def build_unstressed(self):
        return WordStressVariant(self.poetry_word, -1, self.score)

    def get_stressed_form(self):
        return self.stressed_form

    def is_short_word(self):
        if self.is_cyrillic:
            return len(self.poetry_word.form) <= 2
        else:
            return False

    def __repr__(self):
        s = self.stressed_form
        if self.score != 1.0:
            s += '({:5.3f})'.format(self.score)
        return s

    def split_to_syllables(self):
        output_syllables = []

        sx = get_syllables(self.poetry_word.form)
        if sx:
            vcount = 0
            for syllable in sx:
                syllable2 = []
                for c in syllable.text:
                    syllable2.append(c)
                    if c.lower() in 'уеыаоэёяию':
                        vcount += 1
                        if vcount == self.new_stress_pos:
                            syllable2.append('\u0301')
                syllable2 = ''.join(syllable2)
                output_syllables.append(syllable2)
        else:
            vcount = 0
            syllable2 = []
            for c in self.poetry_word.form:
                syllable2.append(c)
                if c.lower() in 'уеыаоэёяию':
                    vcount += 1
                    if vcount == self.new_stress_pos:
                        syllable2.append('\u0301')
            syllable2 = ''.join(syllable2)
            output_syllables.append(syllable2)

        return output_syllables


class PoetryWord(object):
    def __init__(self, lemma, form, upos, tags, stress_pos):
        self.lemma = lemma
        self.form = form
        self.upos = upos
        self.tags = tags
        self.stress_pos = stress_pos
        self.is_rhyming_word = False  # отмечаем последнее слово в каждой строке

        self.leading_consonants = 0  # кол-во согласных ДО первой гласной
        self.trailing_consonants = 0  # кол-во согласных ПОСЛЕ последней гласной
        n_vowels = 0
        for c in self.form:
            clower = c.lower()
            if clower in 'уеыаоэёяию':
                self.trailing_consonants = 0
                n_vowels += 1
            elif c in 'бвгджзклмнпрстфхцчшщ':
                if n_vowels == 0:
                    self.leading_consonants += 1
                else:
                    self.trailing_consonants += 1

    def __repr__(self):
        output = []
        n_vowels = 0
        for c in self.form:
            output.append(c)
            if c.lower() in 'уеыаоэёяию':
                n_vowels += 1
                if n_vowels == self.stress_pos:
                    output.append('\u0301')
        return ''.join(output)

    def get_stress_variants(self, aligner):
        variants = []

        nvowels = sum((c in 'уеыаоэёяию') for c in self.form.lower())
        uform = self.form.lower()

        if uform == 'начала' and self.upos == 'NOUN':
            # У этого слова аж 3 варианта ударения, два для глагола и 1 для существительного.
            # Если тэггер считает, что у нас именно существительное - не используем вариативность глагола.
            # TODO вынести такую логику как-то в словарь ambiguous_accents_2.yaml, чтобы не приходилось тут
            # хардкодить.
            variants.append(WordStressVariant(self, 2, 1.0))
        elif uform in aligner.accentuator.ambiguous_accents2:
            # Вариативное ударение в рамках одной грамматической формы пОнял-понЯл
            for accent in aligner.accentuator.ambiguous_accents2[uform]:
                stress_pos = -1
                n_vowels = 0
                for c in accent:
                    if c.lower() in 'уеыаоэёяию':
                        n_vowels += 1

                    if c in 'АЕЁИОУЫЭЮЯ':
                        stress_pos = n_vowels
                        break

                if stress_pos == -1:
                    raise ValueError('Could not find stressed position in word "{}" in ambiguous_accents2'.format(accent))

                variants.append(WordStressVariant(self, stress_pos, 1.0))
        elif uform == 'нибудь':
            # Частицу "нибудь" не будем ударять:
            # Мо́жет что́ - нибу́дь напла́чу
            #             ~~~~~~
            variants.append(WordStressVariant(self, -1, 1.0))
        elif self.upos in ('ADP', 'CCONJ', 'SCONJ', 'PART', 'INTJ'):
            if uform in ('о', 'у', 'из', 'от', 'под', 'подо', 'за', 'при', 'до', 'про', 'для', 'ко', 'со', 'во') and self.upos == 'ADP':
                # эти предлоги никогда не делаем ударными
                variants.append(WordStressVariant(self, -1, 1.0))
            elif uform in ('не', 'бы', 'ли', 'же'):
                # Частицы "не" и др. никогда не делаем ударной
                variants.append(WordStressVariant(self, -1, 1.0))
            elif uform == 'а':
                # союз "а" всегда безударный:
                # А была бы ты здорова
                # ^
                variants.append(WordStressVariant(self, -1, 1.0))
            elif uform == 'и' and self.upos == 'PART':
                # Частицу "и" не делаем ударной:
                # Вот и она... Оставив магазин
                #     ^
                variants.append(WordStressVariant(self, -1, 1.0))
            elif nvowels == 0:
                # Предлоги без единой гласной типа "в"
                variants.append(WordStressVariant(self, -1, 1.0))
            else:
                # Предлоги, союзы, частицы предпочитаем без ударения,
                # поэтому базовый вариант добавляем с дисконтом:
                if uform in ['лишь', 'вроде', 'если', 'чтобы', 'когда', 'просто', 'мимо', 'даже', 'всё', 'хотя', 'едва', 'нет']:
                    variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68_2']))
                else:
                    variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68']))

                # а вариант без ударения - с нормальным скором:
                if not self.is_rhyming_word:
                    variants.append(WordStressVariant(self, -1, COEFF['@71']))
        elif self.upos in ('PRON', 'ADV', 'DET'):
            if self.is_rhyming_word:
                # исходный вариант с ударением.
                variants.append(WordStressVariant(self, self.stress_pos, 1.0))
            else:
                # Для односложных местоимений (Я), наречий (ТУТ, ГДЕ) и слов типа МОЙ, ВСЯ, если они не последние в строке,
                # добавляем вариант без ударения с дисконтом.
                if nvowels == 1:
                    variants.append(WordStressVariant(self, self.stress_pos, COEFF['@75']))
                    # вариант без ударения
                    variants.append(WordStressVariant(self, -1, COEFF['@77']))
                else:
                    if uform in ['эти', 'эту', 'это', 'мои', 'твои', 'моих', 'твоих', 'моим', 'твоим', 'моей', 'твоей',
                                 'мою', 'твою', 'его', 'ее', 'её', 'себе', 'меня', 'тебя', 'свою', 'свои', 'своим', 'они', 'она',
                                 'уже', 'этом', 'тебе']:
                        # Безударный вариант для таких двусложных прилагательных
                        variants.append(WordStressVariant(self, -1, COEFF['@77_2']))

                    variants.append(WordStressVariant(self, self.stress_pos, COEFF['@79']))
        else:
            if uform in ['есть', 'раз', 'быть', 'будь'] and not self.is_rhyming_word:
                # безударный вариант
                variants.append(WordStressVariant(self, -1, COEFF['@143']))

            # Добавляем исходный вариант с ударением
            variants.append(WordStressVariant(self, self.stress_pos, 1.0))

        # TODO: + сделать вариант смещения позиции ударения в существительном или глаголе
        # ...

        return variants


def sum1(arr):
    return sum((x == 1) for x in arr)


class LineStressVariant(object):
    def __init__(self, poetry_line, stressed_words, aligner):
        self.poetry_line = poetry_line
        self.stressed_words = stressed_words
        self.stress_signature = list(itertools.chain(*(w.stress_signature for w in stressed_words)))
        self.stress_signature_str = ''.join(map(str, self.stress_signature))
        self.total_score = reduce(lambda x, y: x*y, [w.score for w in stressed_words])
        self.penalties = []

        # добавка от 15-12-2021: два подряд ударных слога наказываем сильно!
        if '11' in self.stress_signature_str:
            self.total_score *= 0.1
            self.penalties.append('@236')

        # добавка от 16-12-2021: безударное последнее слово наказываем сильно!
        if self.stressed_words[-1].new_stress_pos == -1:
            self.total_score *= 0.1
            self.penalties.append('@241')

        # 01-01-2022 ударную частицу "и" в начале строки наказываем сильно
        # 〚И́(0.500) споко́йно детворе́〛(0.500)
        if self.stressed_words[0].new_stress_pos == 1 and self.stressed_words[0].poetry_word.form.lower() == 'и':
            self.total_score *= 0.1
            self.penalties.append('@247')

        for word1, word2 in zip(stressed_words, stressed_words[1:]):
            # 28-12-2021 проверяем цепочки согласных в смежных словах
            n_adjacent_consonants = word1.poetry_word.trailing_consonants + word2.poetry_word.leading_consonants
            if n_adjacent_consonants > 5:
                self.total_score *= 0.5
                self.penalties.append('@254')

            # 01-01-2022 Штрафуем за ударный предлог перед существительным:
            # Все по́ дома́м - она и ра́да
            #     ^^^^^^^^
            if word1.poetry_word.upos == 'ADP' and word1.new_stress_pos > 0 and word2.poetry_word.upos in ('NOUN', 'PROPN') and word2.new_stress_pos > 0:
                self.total_score *= 0.5
                self.penalties.append('@261')

        for word1, word2, word3 in zip(stressed_words, stressed_words[1:], stressed_words[2:]):
            # 29-12-2021 Более двух подряд безударных слов - штрафуем
            if word1.new_stress_pos == -1 and word2.new_stress_pos == -1 and word3.new_stress_pos == -1:
                self.total_score *= 0.1
                self.penalties.append('@267')

        # 28-12-2021 штрафуем за подряд идущие короткие слова (1-2 буквы)
        #for word1, word2, word3 in zip(stressed_words, stressed_words[1:], stressed_words[2:]):
        #    if word1.is_short_word() and word2.is_short_word() and word3.is_short_word():
        #        self.total_score *= 0.2

        if sum(self.stress_signature) == 1:
            # Всего один ударный слог в строке... Очень странно.
            # 〚Что за недоразуме́нье〛
            # 00000010
            self.total_score *= 0.1
            self.penalties.append('@279')
        else:
            # 01-01-2022 Детектируем разные сбои ритма
            if self.stress_signature_str in aligner.bad_signature1:
                self.total_score *= 0.1
                self.penalties.append('@284')

    def __repr__(self):
        s = '〚' + ' '.join(w.__repr__() for w in self.stressed_words) + '〛'
        if self.total_score != 1.0:
            s += '({:5.3f})'.format(self.total_score)
        return s

    def get_stressed_line(self):
        s = ' '.join(w.get_stressed_form() for w in self.stressed_words)
        s = normalize_whitespaces(s)
        return s

    def get_unstressed_line(self):
        s = self.get_stressed_line()
        s = s.replace('\u0301', '')
        return s

    def get_score(self):
        return self.total_score

    def score_sign1(self, etalog_sign, line_sign):
        if etalog_sign == 0 and line_sign == 1:
            # Ударный слог в безударном месте - сильный штраф
            return 0.1
        elif etalog_sign == 1 and line_sign == 0:
            # Безударный слог в ударном месте - небольшой штраф
            return 0.9
        else:
            # Ударности полностью соответствуют.
            return 1.0

    def map_meter(self, signature):
        l = len(signature)
        n = len(self.stress_signature) // l
        if (len(self.stress_signature) % l) > 0:
            n += 1

        expanded_sign = (signature * n)[:len(self.stress_signature)]

        sign_scores = [self.score_sign1(x, y) for x, y in zip(expanded_sign, self.stress_signature)]
        sign_score = reduce(lambda x, y: x * y, sign_scores)

        if sign_score < 1.0 and signature[0] == 1 and self.stress_signature[0] == 0:
            # Попробуем сместить вправо, добавив в начало один неударный слог.
            expanded_sign2 = (0,) + expanded_sign[:-1]
            sign_scores2 = [self.score_sign1(x, y) for x, y in zip(expanded_sign2, self.stress_signature)]
            sign_score2 = reduce(lambda x, y: x * y, sign_scores2)
            if sign_score2 > sign_score:
                return sign_score2

        return sign_score

    def split_to_syllables(self):
        output_tokens = []
        for word in self.stressed_words:
            if len(output_tokens) > 0:
                output_tokens.append('|')
            output_tokens.extend(word.split_to_syllables())
        return output_tokens

    def get_last_rhyming_word(self):
        # Вернем последнее слово в строке, которое надо проверять на рифмовку.
        # Финальную пунктуацию игнорируем.
        for pword in self.stressed_words[::-1]:
            if pword.poetry_word.upos != 'PUNCT':
                return pword
        return None


def count_vowels(s):
    return sum((c.lower() in 'уеыаоэёяию') for c in s)


class PoetryLine(object):
    def __init__(self):
        self.text = None
        self.pwords = None
        #self.stress_signature = None

    @staticmethod
    def build(text, udpipe_parser, accentuator):
        pline = PoetryLine()
        pline.text = text
        pline.pwords = []

        text2 = text
        for c in '.,?!:;…-–—«»”“„‘’`"':
            text2 = text2.replace(c, ' ' + c + ' ').replace('  ', ' ')

        parsings = udpipe_parser.parse_text(text2)
        for parsing in parsings:
            for ud_token in parsing:
                word = ud_token.form.lower()
                stress_pos = accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])
                if count_vowels(word) > 0 and stress_pos == -1:
                    # Если слово допускает альтернативные ударения по списку, то берем первое из них (обычно это
                    # основное, самое частотное ударение), и не бросаем исключение, так как дальше матчер все равно
                    # будет перебирать все варианты по списку.
                    if word in accentuator.ambiguous_accents2:
                        ax = accentuator.ambiguous_accents2[word]
                        a1 = ax[0]
                        n_vowels = 0
                        for c in a1:
                            if c.lower() in 'уеыаоэёяию':
                                n_vowels += 1

                            if c in 'АЕЁИОУЫЭЮЯ':
                                stress_pos = n_vowels
                                break

                    if stress_pos == -1:
                        msg = 'Could not locate stress position in word "{}"'.format(word)
                        raise ValueError(msg)

                pword = PoetryWord(ud_token.lemma, ud_token.form, ud_token.upos, ud_token.tags, stress_pos)
                pline.pwords.append(pword)

        # Отмечаем последнее слово в строке, так как должно ударяться, за исключением
        # очень редких случаев:
        # ... я же
        # ... ляжет
        pline.pwords[-1].is_rhyming_word = True

        return pline

    def __repr__(self):
        return ' '.join([pword.__repr__() for pword in self.pwords])

    def get_stress_variants(self, aligner):
        wordx = [pword.get_stress_variants(aligner) for pword in self.pwords]
        variants = [LineStressVariant(self, swords, aligner) for swords in itertools.product(*wordx)]

        # 23-01-2022 добавляем варианты, возникающие из-за особых ударений в словосочетаниях типа "пО полю"
        lwords = [w.form.lower() for w in self.pwords]
        if any((w in aligner.collocation2_first) for w in lwords) and any((w in aligner.collocation2_second) for w in lwords):
            # В строке возможно присутствует одно из особых словосочетаний длиной 2
            add_variants = []
            for colloc in aligner.collocations:
                for i1, (w1, w2) in enumerate(zip(lwords, lwords[1:])):
                    if colloc.hit(w1, w2):
                        # из всех вариантов в variants делаем еще по 1 варианту
                        for variant in variants:
                            v = colloc.produce_stressed_line(variant, aligner)
                            add_variants.append(v)

            if add_variants:
                variants.extend(add_variants)

        return variants

    #def get_last_rhyming_word(self):
    #    for pword in self.pwords[::-1]:
    #        if pword.upos != 'PUNCT':
    #            return pword
    #    return None


class PoetryAlignment(object):
    def __init__(self, poetry_lines, score, meter, rhyme_scheme):
        self.poetry_lines = poetry_lines
        self.score = score
        self.meter = meter
        self.rhyme_scheme = rhyme_scheme
        self.error_text = None

    def __repr__(self):
        s = '{} {}({:5.3f}):\n'.format(self.meter, self.rhyme_scheme, self.score)
        s += '\n'.join(map(str, self.poetry_lines))
        return s

    @staticmethod
    def build_no_rhyming_result(poetry_lines):
        a = PoetryAlignment(poetry_lines, 0.0, None, None)
        a.error_text = 'Отсутствует рифмовка последних слов'
        return a

    def get_stressed_lines(self):
        return '\n'.join(x.get_stressed_line() for x in self.poetry_lines)

    def get_unstressed_lines(self):
        return '\n'.join(x.get_unstressed_line() for x in self.poetry_lines)

    def split_to_syllables(self, do_arabize):
        lx = []
        for line in self.poetry_lines:
            sx = line.split_to_syllables()
            if do_arabize:
                sx = sx[::-1]
            lx.append(' '.join(sx))
        return lx


meters = [('хорей', (1, 0)),
          ('ямб', (0, 1)),
          ('дактиль', (1, 0, 0)),
          ('амфибрахий', (0, 1, 0)),
          ('анапест', (0, 0, 1))]


class CollocationStress(object):
    def __init__(self):
        self.words = []
        self.stressed_word_index = -1
        self.stress_pos = -1

    def __repr__(self):
        return ' '.join(self.words) if self.words else ''

    @staticmethod
    def load_collocation(colloc_str):
        res = CollocationStress()
        words = colloc_str.split()
        res.words = [w.lower() for w in words]
        uvx = 'АЕЁИОУЫЭЮЯ'
        vx = 'аеёиоуыэюя'
        for iword, word in enumerate(words):
            if any((c in word) for c in uvx):
                res.stressed_word_index = iword
                vowel_count = 0
                for c in word:
                    if c.lower() in vx:
                        vowel_count += 1
                        if c in uvx:
                            res.stress_pos = vowel_count
                            break

        return res

    def hit(self, word1, word2):
        return self.words[0] == word1 and self.words[1] == word2

    def produce_stressed_line(self, src_line, aligner):
        nw1 = len(src_line.stressed_words) - 1
        for i1, word1 in enumerate(src_line.stressed_words):
            if word1.poetry_word.form.lower() == self.words[0]:
                if i1 < nw1:
                    word2 = src_line.stressed_words[i1+1]
                    if word2.poetry_word.form.lower() == self.words[1]:
                        new_stressed_words = list(src_line.stressed_words[:i1])

                        if self.stressed_word_index == 0:
                            # первое слово становится ударным, второе - безударное
                            new_stressed_words.append(word1.build_stressed(self.stress_pos))
                            new_stressed_words.append(word2.build_unstressed())
                        else:
                            # первое слово становится безударным, второе - ударное
                            new_stressed_words.append(word1.build_unstressed())
                            new_stressed_words.append(word2.build_stressed(self.stress_pos))

                        # остаток слов справа от второго слова
                        new_stressed_words.extend(src_line.stressed_words[i1+2:])
                        new_variant = LineStressVariant(src_line.poetry_line, new_stressed_words, aligner)
                        return new_variant

        raise ValueError('Inconsistent call of CollocationStress::produce_stressed_line')

class PoetryStressAligner(object):
    def __init__(self, udpipe, accentuator, data_dir):
        self.udpipe = udpipe
        self.accentuator = accentuator

        self.collocations = []
        self.collocation2_first = set()
        self.collocation2_second = set()
        with io.open(os.path.join(data_dir, 'collocation_accents.dat'), 'r', encoding='utf-8') as rdr:
            for line in rdr:
                line = line.strip()
                if line.startswith('#') or len(line) == 0:
                    continue

                c = CollocationStress.load_collocation(line)
                self.collocations.append(c)
                if len(c.words) == 2:
                    # обновляем хэш для быстрой проверки наличия словосочетания длиной 2
                    self.collocation2_first.add(c.words[0])
                    self.collocation2_second.add(c.words[1])

        self.bad_signature1 = set()
        with io.open(os.path.join(data_dir, 'bad_signature1.dat'), 'r', encoding='utf-8') as rdr:
            for line in rdr:
                s = line.strip()
                if s:
                    if s[0] == '#':
                        continue
                    else:
                        pattern = s
                        if pattern[0] == "'" and pattern[-1] == "'":
                            pattern = pattern[1:-1]

                        self.bad_signature1.add(pattern)

        self.bad_alignments2 = set()
        with io.open(os.path.join(data_dir, 'bad_alignment2.dat'), 'r', encoding='utf-8') as rdr:
            pattern_lines = []
            for line in rdr:
                s = line.strip()
                if s:
                    if s[0] == '#':
                        continue
                    else:
                        pattern_lines.append(s)
                else:
                    if len(pattern_lines) == 2:
                        for pattern1 in pattern_lines[0].split(','):
                            pattern1 = pattern1.strip()
                            if pattern1[0] == "'" and pattern1[-1] == "'":
                                pattern1 = pattern1[1:-1]
                            for pattern2 in pattern_lines[1].split(','):
                                pattern2 = pattern2.strip()
                                if pattern2[0] == "'" and pattern2[-1] == "'":
                                    pattern2 = pattern2[1:-1]

                                self.bad_alignments2.add((pattern1, pattern2))
                    elif len(pattern_lines) != 0:
                        raise RuntimeError()

                    pattern_lines = []

    def map_meter(self, signature, lines):
        scores = [line.map_meter(signature) for line in lines]
        return reduce(lambda x, y: x*y, scores)

    def map_meters(self, lines):
        best_score = -1.0
        best_meter = None
        for name, signature in meters:
            score = self.map_meter(signature, lines)
            if score > best_score:
                best_score = score
                best_meter = name
        return best_meter, best_score

    def get_spectrum(sline1):
        spectrum = set()
        unstressed_seq_len = 0
        for sign in pline.stress_signature:
            if sign == 1:
                spectrum.add(unstressed_seq_len)
                unstressed_seq_len = 0
            else:
                unstressed_seq_len += 1
        return spectrum


    def map_2signatures(self, sline1, sline2):
        # ВАРИАНТ 1
        #if len(sline1.stress_signature_str) == len(sline2.stress_signature_str):
        #    d = jellyfish.hamming_distance(sline1.stress_signature_str, sline2.stress_signature_str)
        #else:
        #    d = jellyfish.damerau_levenshtein_distance(sline1.stress_signature_str, sline2.stress_signature_str)
        #score = math.exp(-d*1.0)

        # ВАРИАНТ 2
        #d = jellyfish.damerau_levenshtein_distance(sline1.stress_signature_str, sline2.stress_signature_str)
        #score = 1.0 - float(d) / max(len(sline1.stress_signature), len(sline2.stress_signature), 1e-6)

        # ВАРИАНТ 3
        d = jellyfish.damerau_levenshtein_distance(sline1.stress_signature_str, sline2.stress_signature_str)
        score = math.exp(-d*1.0)

        # Добавка от 31-12-2021: если во второй строке больше ударных слогов, чем в первой - штрафуем сильно.
        #nstressed_1 = sum(sline1.stress_signature)
        #nstressed_2 = sum(sline2.stress_signature)
        #if nstressed_2 > nstressed_1:
        #    score *= pow(0.5, (nstressed_2-nstressed_1))

        #spectrum1 = self.get_spectrum(sline1)
        #spectrum2 = self.get_spectrum(sline2)

        # 01-01-2022 Разные корявые рифмовки детектируем по словарю сигнатур.
        if (sline1.stress_signature_str, sline2.stress_signature_str) in self.bad_alignments2:
            score *= 0.1

        return score

    def align(self, lines0, check_rhymes=True):
        # Иногда для наглядности можем выводить сгенерированные стихи вместе со значками ударения.
        # Эти значки мешают работе алгоритма транскриптора, поэтому уберем их сейчас.
        lines = [line.replace('\u0301', '') for line in lines0]

        if len(lines) == 2:
            return self.align2(lines, check_rhymes)
        elif len(lines) == 4:
            return self.align4(lines, check_rhymes)
        elif len(lines) == 1:
            return self.align1(lines)
        else:
            raise NotImplementedError()

    def check_rhyming(self, poetry_word1, poetry_word2):
        #return rhymed(self.accentuator,
        #              poetry_word1.form, [poetry_word1.upos] + poetry_word1.tags,
        #              poetry_word2.form, [poetry_word2.upos] + poetry_word2.tags)
        # 01.02.2022 проверяем слова с ударениями по справочнику рифмовки
        f1 = poetry_word1.stressed_form
        f2 = poetry_word2.stressed_form
        if (f1, f2) in self.accentuator.rhymed_words or (f2, f1) in self.accentuator.rhymed_words:
            return True

        return rhymed2(self.accentuator,
                       poetry_word1.poetry_word.form, poetry_word1.new_stress_pos, [poetry_word1.poetry_word.upos] + poetry_word1.poetry_word.tags,
                       poetry_word2.poetry_word.form, poetry_word2.new_stress_pos, [poetry_word2.poetry_word.upos] + poetry_word2.poetry_word.tags)

    def _align_line_group(self, metre_signature, lines):
        line1 = lines[0]
        for line2 in lines[1:]:
            if (line1.stress_signature_str, line2.stress_signature_str) in self.bad_alignments2:
                # Отсеиваем заведомо несопоставимые пары
                return 0.0
        score = self.map_meter(metre_signature, lines)
        return score

    def _align_line_groups(self, line_groups):
        # Для каждого эталонного метра проверяем, насколько хорошо ВСЕ группы строк вписываются в этот метр
        best_score = -1.0
        best_meter = None
        for metre_name, metre_signature in meters:
            group_scores = [self._align_line_group(metre_signature, group) for group in line_groups]
            score = reduce(lambda x, y: x*y, group_scores)
            if score > best_score:
                best_score = score
                best_meter = metre_name

        return best_score, best_meter

    def align4(self, lines, check_rhymes):
        plines = [PoetryLine.build(line, self.udpipe, self.accentuator) for line in lines]

        # Список вариантов простановки ударения с учётом опциональности ударений для союзов, предлогов и т.д.
        plinevx = [pline.get_stress_variants(self) for pline in plines]

        # Идем по списку вариантов, отображаем на эталонные метры и ищем лучший вариант.
        best_variant = None
        best_score = 0.0
        best_meter = None
        best_ivar = None
        best_rhyme_scheme = None

        total_permutations = reduce(lambda x, y: x * y, [len(z) for z in plinevx])
        if total_permutations > 10000:
            raise ValueError('Too many optional stresses: {}'.format(total_permutations))

        vvx = list(itertools.product(*plinevx))

        for ivar, plinev in enumerate(vvx):
            # plinev это набор из четырех экземпляров LineStressVariant.

            # Проверим, что эти 4 строки имеют рифмовку
            if check_rhymes:
                rhyme_scheme = None
                last_pwords = [pline.get_last_rhyming_word() for pline in plinev]
                # TODO - можно закэшировать проверки пар, так как обычно последние слова не вариативны.
                if self.check_rhyming(last_pwords[0], last_pwords[2]) and self.check_rhyming(last_pwords[1], last_pwords[3]):
                    # Считаем, что слово не рифмуется само с собой, чтобы не делать для отсечения
                    # таких унылых рифм отдельную проверку в другой части кода.
                    if last_pwords[0].poetry_word.form.lower() != last_pwords[2].poetry_word.form.lower() and last_pwords[1].poetry_word.form.lower() != last_pwords[3].poetry_word.form.lower():
                        rhyme_scheme = 'ABAB'
                elif self.check_rhyming(last_pwords[0], last_pwords[3]) and self.check_rhyming(last_pwords[1], last_pwords[2]):
                    if last_pwords[0].poetry_word.form.lower() != last_pwords[3].poetry_word.form.lower() and last_pwords[1].poetry_word.form.lower() != last_pwords[2].poetry_word.form.lower():
                        rhyme_scheme = 'ABBA'
                # 22-12-2021 добавлена рифмовка AABB
                elif self.check_rhyming(last_pwords[0], last_pwords[1]) and self.check_rhyming(last_pwords[2], last_pwords[3]):
                    if last_pwords[0].poetry_word.form.lower() != last_pwords[1].poetry_word.form.lower() and last_pwords[2].poetry_word.form.lower() != last_pwords[3].poetry_word.form.lower():
                        rhyme_scheme = 'AABB'
                # 28-12-2021 добавлена рифмовка "рубаи" AABA
                elif self.check_rhyming(last_pwords[0], last_pwords[1]) and\
                     self.check_rhyming(last_pwords[0], last_pwords[3]) and\
                     not self.check_rhyming(last_pwords[0], last_pwords[2]):
                    if last_pwords[0].poetry_word.form.lower() != last_pwords[1].poetry_word.form.lower() and last_pwords[0].poetry_word.form.lower() != last_pwords[3].poetry_word.form.lower():
                        rhyme_scheme = 'AABA'
            else:
                rhyme_scheme = 'ABAB'

            if rhyme_scheme is not None:
                rhyming_score = 0.0
                mapped_meter = None
                if rhyme_scheme == 'ABAB':
                    # Оцениваем, насколько хорошо соответствуют сигнатуры строк для схемы рифмовки ABAB
                    score1234, mapped_meter = self._align_line_groups([(plinev[0], plinev[2]), (plinev[1], plinev[3])])
                    rhyming_score = 1.0 - COEFF['@225']*(1.0 - score1234)
                elif rhyme_scheme == 'ABBA':
                    score1234, mapped_meter = self._align_line_groups([(plinev[0], plinev[3]), (plinev[1], plinev[2])])
                    rhyming_score = 1.0 - COEFF['@225']*(1.0 - score1234)
                elif rhyme_scheme == 'AABB':
                    score1234, mapped_meter = self._align_line_groups([(plinev[0], plinev[1]), (plinev[2], plinev[3])])
                    rhyming_score = 1.0 - COEFF['@225']*(1.0 - score1234)
                elif rhyme_scheme == 'AABA':
                    score1234, mapped_meter = self._align_line_groups([(plinev[0], plinev[1], plinev[2], plinev[3])])
                    rhyming_score = 1.0 - COEFF['@225']*(1.0 - pow(score1234, 0.5))
                else:
                    raise NotImplementedError()

                score = rhyming_score * reduce(lambda x, y: x*y, [l.get_score() for l in plinev])
                if score > best_score:
                    best_variant = plinev
                    best_score = score
                    best_meter = mapped_meter
                    best_ivar = ivar
                    best_rhyme_scheme = rhyme_scheme

        if best_rhyme_scheme is None:
            # Не получилось подобрать рифмовку окончаний строк.
            # В этом случае вернем результат с нулевым скором и особым текстом, чтобы
            # можно было вывести в лог строки с каким-то дефолтными
            return PoetryAlignment.build_no_rhyming_result([pline.get_stress_variants(self)[0] for pline in plines])

        # Возвращаем найденный вариант разметки и его оценку
        return PoetryAlignment(best_variant, best_score, best_meter, best_rhyme_scheme)

    def align2(self, lines, check_rhymes):
        plines = [PoetryLine.build(line, self.udpipe, self.accentuator) for line in lines]
        plinevx = [pline.get_stress_variants(self) for pline in plines]

        # Идем по списку вариантов, отображаем на эталонные метры и ищем лучший вариант.
        best_variant = None
        best_score = 0.0
        best_meter = None
        best_ivar = None
        best_rhyme_scheme = None

        vvx = list(itertools.product(*plinevx))
        if len(vvx) > 10000:
            raise ValueError('Too many optional stresses: {}'.format(len(vvx)))

        for ivar, plinev in enumerate(vvx):
            # plinev это набор из двух экземпляров LineStressVariant.

            # Определяем рифмуемость
            rhyme_scheme = None
            if check_rhymes:
                last_pwords = [pline.get_last_rhyming_word() for pline in plinev]
                if self.check_rhyming(last_pwords[0], last_pwords[1]):
                    # Считаем, что слово не рифмуется само с собой, чтобы не делать для отсечения
                    # таких унылых рифм отдельную проверку в другой части кода.
                    if last_pwords[0].poetry_word.form.lower() != last_pwords[1].poetry_word.form.lower():
                        rhyme_scheme = 'AA'
            else:
                rhyme_scheme = '--'

            score_12 = self.map_2signatures(plinev[0], plinev[1])

            # Ищем лучшее отображение метра.
            mapped_meter, mapping_score = self.map_meters(plinev)
            score = score_12 * mapping_score * reduce(lambda x, y: x*y, [l.get_score() for l in plinev])
            if score > best_score:
                best_variant = plinev
                best_score = score
                best_meter = mapped_meter
                best_ivar = ivar
                best_rhyme_scheme = rhyme_scheme

        if best_rhyme_scheme is None:
            # Не получилось подобрать рифмовку окончаний строк.
            # В этом случае вернем результат с нулевым скором и особым текстом, чтобы
            # можно было вывести в лог строки с каким-то дефолтными
            return PoetryAlignment.build_no_rhyming_result([pline.get_stress_variants(self)[0] for pline in plines])

        # Возвращаем найденный вариант разметки и его оценку
        return PoetryAlignment(best_variant, best_score, best_meter, rhyme_scheme=best_rhyme_scheme)

    def align1(self, lines):
        plines = [PoetryLine.build(line, self.udpipe, self.accentuator) for line in lines]
        plinevx = [pline.get_stress_variants(self) for pline in plines]

        # Идем по списку вариантов, отображаем на эталонные метры и ищем лучший вариант.
        best_variant = None
        best_score = 0.0
        best_meter = None
        best_ivar = None

        vvx = list(itertools.product(*plinevx))
        if len(vvx) > 10000:
            raise ValueError('Too many optional stresses: {}'.format(len(vvx)))

        for ivar, plinev in enumerate(vvx):
            # plinev это набор из двух экземпляров LineStressVariant.

            # Ищем лучшее отображение метра.
            mapped_meter, mapping_score = self.map_meters(plinev)
            score = mapping_score * reduce(lambda x, y: x*y, [l.get_score() for l in plinev])
            if score > best_score:
                best_variant = plinev
                best_score = score
                best_meter = mapped_meter
                best_ivar = ivar

        # Возвращаем найденный вариант разметки и его оценку
        return PoetryAlignment(best_variant, best_score, best_meter, rhyme_scheme='')

    def detect_repeating(self, alignment):
        # Иногда генеративная модель выдает повторы существтельных типа "любовь и любовь" в одной строке.
        # Такие генерации выглядят криво.
        # Данный метод детектирует повтор леммы существительного в строке.
        for pline in alignment.poetry_lines:
            n_lemmas = collections.Counter()
            for pword in pline.poetry_line.pwords:
                if pword.upos in ('NOUN', 'PROPN'):
                    n_lemmas[pword.lemma] += 1
            if n_lemmas and n_lemmas.most_common(1)[0][1] > 1:
                return True

            # любой повтор XXX XXX
            for w1, w2 in zip(pline.poetry_line.pwords, pline.poetry_line.pwords[1:]):
                if w1.form.lower() == w2.form.lower():
                    return True

            # также штрафуем за паттерн "XXX и XXX"
            for w1, w2, w3 in zip(pline.poetry_line.pwords, pline.poetry_line.pwords[1:], pline.poetry_line.pwords[2:]):
                if w2.form in ('и', ',', 'или', 'иль', 'аль', 'да'):
                    if w1.form.lower() == w3.form.lower():
                        return True

        return False

    def detect_poor_poetry(self, alignment):
        """Несколько эвристик для обнаружения скучных рифм, которые мы не хотим получать"""

        last_words = [pline.poetry_line.pwords[-1].form.lower() for pline in alignment.poetry_lines]

        # Если два глагольных оконания, причем одно является хвостом другого - это бедная рифма:
        # ждать - подождать
        # смотреть - посмотреть
        # etc.
        rhyme_pairs = []
        if alignment.rhyme_scheme == 'ABAB':
            rhyme_pairs.append((alignment.poetry_lines[0].stressed_words[-1], alignment.poetry_lines[2].stressed_words[-1]))
            rhyme_pairs.append((alignment.poetry_lines[1].stressed_words[-1], alignment.poetry_lines[2].stressed_words[-1]))
        elif alignment.rhyme_scheme == 'ABBA':
            rhyme_pairs.append((alignment.poetry_lines[0].stressed_words[-1], alignment.poetry_lines[3].stressed_words[-1]))
            rhyme_pairs.append((alignment.poetry_lines[1].stressed_words[-1], alignment.poetry_lines[2].stressed_words[-1]))
        elif alignment.rhyme_scheme == 'AABA':
            rhyme_pairs.append((alignment.poetry_lines[0].stressed_words[-1], alignment.poetry_lines[1].stressed_words[-1]))
            rhyme_pairs.append((alignment.poetry_lines[0].stressed_words[-1], alignment.poetry_lines[3].stressed_words[-1]))
        elif alignment.rhyme_scheme == 'AABB':
            rhyme_pairs.append((alignment.poetry_lines[0].stressed_words[-1], alignment.poetry_lines[1].stressed_words[-1]))
            rhyme_pairs.append((alignment.poetry_lines[2].stressed_words[-1], alignment.poetry_lines[3].stressed_words[-1]))

        for word1, word2 in rhyme_pairs:
            if word1.poetry_word.upos == 'VERB' and word2.poetry_word.upos == 'VERB':
                # 11-01-2022 если пара слов внесена в специальный список рифмующихся слов, то считаем,
                # что тут все нормально:  ВИТАЮ-ТАЮ
                if (word1.poetry_word.form.lower(), word2.poetry_word.form.lower()) in self.accentuator.rhymed_words:
                    continue

                if word1.poetry_word.form.endswith(word2.poetry_word.form):
                    return True
                elif word2.poetry_word.form.endswith(word1.poetry_word.form):
                    return True

        # Для других частей речи проверяем заданные группы слов.
        for bad_group in ['твой мой свой'.split(), 'тебе мне себе'.split(), 'него его'.split(), 'твои свои'.split(),
                          'наши ваши'.split(), 'меня тебя себя'.split(), 'мной тобой собой'.split(),
                          'мною тобой собою'.split(),
                          'нее ее неё её'.split(), '~шел ~шёл'.split(),
                          'твоем твоём своем своём моем моём'.split(),
                          'когда никогда навсегда кое-когда'.split(),
                          'кто никто кое-кто'.split(),
                          'где нигде везде'.split(),
                          'каких никаких таких сяких'.split(),
                          'какого никакого такого сякого'.split(),
                          'какую никакую такую сякую'.split(),
                          'сможем поможем можем'.split(),
                          'смогу помогу могу'.split(),
                          'поехал уехал наехал приехал въехал заехал доехал подъехал'.split(),
                          'того чего ничего никого кого'.split(),
                          'ждали ожидали'.split(),
                          'подумать придумать думать продумать надумать удумать'.split(),
                          ]:
            if bad_group[0][0] == '~':
                n_hits = 0
                for last_word in last_words:
                    n_hits += sum((re.match('^.+' + ending[1:] + '$', last_word, flags=re.I) is not None) for ending in bad_group)
            else:
                n_hits = sum((word in last_words) for word in bad_group)

            if n_hits > 1:
                return True

        return False

hp_trial_count = 0
hp_cur_best = 0.0

def objective(space):
    # Целевая функция для hyperopt

    global COEFF, hp_trial_count, hp_cur_best
    COEFF = space

    hp_trial_count += 1

    total_score = 0.0
    for poem in good_poems:
        alignment = aligner.align(poem)
        total_score += alignment.score

    print('Hyperopt trial#{} total_score={} hp_cur_best={}'.format(hp_trial_count, total_score, hp_cur_best))
    if total_score > hp_cur_best:
        print('\n!!! NEW BEST SCORE={} for params={}\n'.format(total_score, space))
        hp_cur_best = total_score

    return {'loss': -total_score, 'status': STATUS_OK}


if __name__ == '__main__':
    data_dir = '../../data'
    tmp_dir = '../../tmp'
    models_dir = '../../models'

    #udpipe = UdpipeParser()
    #udpipe.load(models_dir)
    udpipe = StanzaParser()

    accents = Accents()
    accents.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
    accents.after_loading(stress_model_dir=os.path.join(tmp_dir, 'stress_model'))

    aligner = PoetryStressAligner(udpipe, accents, os.path.join(data_dir, 'poetry', 'dict'))

    x = accents.get_accent('самоцветы')
    print(x)

    # ================================================

    poem = """Махаончик наш пушист
Приставуч, как банный лист
Он мяукает, как кошка
И изящен, словно глист"""
    poem = [z.strip() for z in poem.split('\n') if z.strip()]

    alignment = aligner.align(poem)
    print(alignment)
    print('='*80)
    #print(alignment.get_unstressed_lines())
    for pline in alignment.poetry_lines:
        print(pline.stress_signature_str)

    print('is_poor={}'.format(aligner.detect_poor_poetry(alignment)))

    exit(0)

    # =============================================

    pline = PoetryLine.build('Ветер, ветер, ты могуч', udpipe, accents)
    print(pline)

    print('='*60)
    vx = pline.get_stress_variants()
    for line_variant in vx:
        print(str(line_variant))

    print('='*60)

    exit(0)
