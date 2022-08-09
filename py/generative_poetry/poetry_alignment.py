"""
15-12-2021 Введен сильный штраф за 2 подряд ударных слога
17-12-2021 Регулировка ударности некоторых наречий и частиц, удаление лишних пробелов вокруг дефиса при выводе строки с ударениями
18-12-2021 Коррекция пробелов вынесена в отдельный модуль whitespace_normalization
22-12-2021 Добавлена рифмовка AABB
28-12-2021 Добавлены еще штрафы за всякие нехорошие с точки зрения вокализма ситуации в строке, типа 6 согласных подряд в смежных словах "пёстр страх"
23-01-2022 Добавлен код для словосочетаний с вариативным ударением типа "пО лесу | по лЕсу"
26-01-2022 Если слово допускает альтернативные ударения по списку и теги не позволяют сделать выбор, то берем первое ударение, а не бросаем исключение.
07-04-2022 Если из-за ошибки частеречной разметки не удалось определить вариант ударения омографа, то будем перебирать все варианты.
22-04-2022 отдельно детектируем рифмовку AAAA, так как она зачастую выглядит очень неудачно и ее желательно устранять из обучающего датасета.
07.06.2022 не штрафуем строку с одним ударным слогом, если строка состоит из единственного слова или сочетания предлог+сущ
10.06.2022 Если в строке есть только одно слово (без учета пунктуации), то для него берем все известные варианты ударения. Это нужно
           для корректной разметки депрессяшек/артишоков, так как частеречная разметка на одном слове не работает и не позволяет
           выбрать корректный вариант ударения.
22.06.2022 в артишоках для последней строки с одним словом для OOV делаем перебор всех вариантов ударности.
04.08.2022 добавлен учет 3-словных словосочетаний типа "бок О бок"
"""

import collections
import itertools
from functools import reduce
import os
import io
import math
import jellyfish
import re

from poetry.phonetic import Accents, rhymed2, rhymed_fuzzy2
from generative_poetry.udpipe_parser import UdpipeParser
from generative_poetry.stanza_parser import StanzaParser
from generative_poetry.metre_classifier import get_syllables
from generative_poetry.whitespace_normalization import normalize_whitespaces


# Коэффициенты для штрафов за разные отступления от идеальной метрики.
COEFF = dict()
COEFF['@68'] = 0.5
COEFF['@68_2'] = 0.95
COEFF['@71'] = 1.0
COEFF['@75'] = 0.9
COEFF['@77'] = 1.0
COEFF['@77_2'] = 1.0
COEFF['@79'] = 1.0
COEFF['@126'] = 0.98
COEFF['@225'] = 0.95
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

    @property
    def form(self):
        return self.poetry_word.form

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
    def __init__(self, lemma, form, upos, tags, stress_pos, alternative_stress_positions=None):
        self.lemma = lemma
        self.form = form
        self.upos = upos
        self.tags = tags
        self.stress_pos = stress_pos
        if alternative_stress_positions:
            self.alternative_stress_positions = alternative_stress_positions  # все варианты положения ударения, первый вариант - условной основной
        else:
            self.alternative_stress_positions = [stress_pos]

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
        elif len(self.alternative_stress_positions) > 1:
            # 07.04.2022 из-за ошибки pos-tagger'а не удалось обработать омограф.
            # будем перебирать все варианты ударения в нем.
            for stress_pos in self.alternative_stress_positions:
                variants.append(WordStressVariant(self, stress_pos, 1.0))
        elif uform == 'нибудь':
            # Частицу "нибудь" не будем ударять:
            # Мо́жет что́ - нибу́дь напла́чу
            #             ~~~~~~
            variants.append(WordStressVariant(self, -1, 1.0))
        elif uform == 'нет':  # and not self.is_rhyming_word:
            # частицу (или глагол для Stanza) "нет" с ударением штрафуем
            variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68_2']))

            # а вариант без ударения - с нормальным скором:
            variants.append(WordStressVariant(self, -1, COEFF['@71']))
        elif self.upos in ('ADP', 'CCONJ', 'SCONJ', 'PART', 'INTJ'): # and not self.is_rhyming_word:
            if uform in ('о', 'у', 'из', 'от', 'под', 'подо', 'за', 'при', 'до', 'про', 'для', 'ко', 'со', 'во') and self.upos == 'ADP':
                # эти предлоги никогда не делаем ударными
                variants.append(WordStressVariant(self, -1, 1.0))

                # Но если это последнее слово в строке, то допускается вариант:
                # необосно́ванная ве́ра
                # в своё́ владе́ние дзюдо́
                # не так вредна́ в проце́ссе дра́ки
                # как до
                if self.is_rhyming_word:
                    variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68']))
            elif uform in ('не', 'бы', 'ли', 'же'):
                # Частицы "не" и др. никогда не делаем ударной
                variants.append(WordStressVariant(self, -1, 1.0))

                if self.is_rhyming_word:
                    variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68']))
            elif uform == 'а':
                # союз "а" всегда безударный:
                # А была бы ты здорова
                # ^
                variants.append(WordStressVariant(self, -1, 1.0))

                if self.is_rhyming_word:
                    variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68']))
            elif uform == 'и' and self.upos == 'PART':
                # Частицу "и" не делаем ударной:
                # Вот и она... Оставив магазин
                #     ^
                variants.append(WordStressVariant(self, -1, 1.0))

                if self.is_rhyming_word:
                    variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68']))
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
                #if not self.is_rhyming_word:
                variants.append(WordStressVariant(self, -1, COEFF['@71']))
        elif self.upos in ('PRON', 'ADV', 'DET'):
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
            if uform in ['есть', 'раз', 'быть', 'будь']:  # and not self.is_rhyming_word:
                # безударный вариант
                variants.append(WordStressVariant(self, -1, COEFF['@143']))

            # Добавляем исходный вариант с ударением
            variants.append(WordStressVariant(self, self.stress_pos, 1.0))

        # TODO: + сделать вариант смещения позиции ударения в существительном или глаголе
        # ...

        return variants

    def get_first_stress_variant(self):
        return WordStressVariant(self, self.stress_pos, 1.0)


def sum1(arr):
    return sum((x == 1) for x in arr)


class RhymingTail(object):
    def __init__(self, unstressed_prefix, stressed_word, unstressed_postfix_words):
        self.stressed_word = stressed_word
        self.unstressed_postfix_words = unstressed_postfix_words
        self.unstressed_tail = ''.join(w.poetry_word.form for w in unstressed_postfix_words)
        self.prefix = '' if unstressed_prefix is None else unstressed_prefix
        self.ok = stressed_word is not None and stressed_word.new_stress_pos != -1

    def is_ok(self):
        return self.ok

    def is_simple(self):
        return len(self.unstressed_postfix_words) == 0 and len(self.prefix) == 0

    def __repr__(self):
        s = self.prefix + self.stressed_word.stressed_form
        if self.unstressed_tail:
            s += ' ' + self.unstressed_tail
        return s

    def get_unstressed_tail(self):
        return self.unstressed_tail


class LineStressVariant(object):
    def __init__(self, poetry_line, stressed_words, aligner):
        self.poetry_line = poetry_line
        self.stressed_words = stressed_words
        self.stress_signature = list(itertools.chain(*(w.stress_signature for w in stressed_words)))
        self.stress_signature_str = ''.join(map(str, self.stress_signature))
        self.init_rhyming_tail()
        self.score_sequence(aligner)

    def score_sequence(self, aligner):
        self.total_score = reduce(lambda x, y: x*y, [w.score for w in self.stressed_words])
        self.penalties = []

        # 04-08-2022 если клаузулла безударная, то имеем кривое оформление ритмического рисунка.
        if not self.rhyming_tail.is_ok():
            self.total_score *= 0.1
            self.penalties.append('@307')

        # 06-08-2022 если в строке вообще ни одного ударения - это неприемлемо
        if sum((w.new_stress_pos != -1) for w in self.stressed_words) == 0:
            self.total_score *= 0.01
            self.penalties.append('@330')

        # добавка от 15-12-2021: два подряд ударных слога наказываем сильно!
        if '11' in self.stress_signature_str:
            self.total_score *= 0.1
            self.penalties.append('@290')

        # добавка от 16-12-2021: безударное последнее слово наказываем сильно!
        # коррекция от 05.06.2022: так как последним бывает пунктуатор, используем метод для получения рифмуемого слова
        #if self.stressed_words[-1].new_stress_pos == -1:
        #if self.get_last_rhyming_word().new_stress_pos == -1:
        #    self.total_score *= 0.1
        #    self.penalties.append('@297')

        # 01-01-2022 ударную частицу "и" в начале строки наказываем сильно
        # 〚И́(0.500) споко́йно детворе́〛(0.500)
        if self.stressed_words[0].new_stress_pos == 1 and self.stressed_words[0].poetry_word.form.lower() == 'и':
            self.total_score *= 0.1
            self.penalties.append('@303')

        for word1, word2 in zip(self.stressed_words, self.stressed_words[1:]):
            # 28-12-2021 проверяем цепочки согласных в смежных словах
            n_adjacent_consonants = word1.poetry_word.trailing_consonants + word2.poetry_word.leading_consonants
            if n_adjacent_consonants > 5:
                self.total_score *= 0.5
                self.penalties.append('@309')

            # 01-01-2022 Штрафуем за ударный предлог перед существительным:
            # Все по́ дома́м - она и ра́да
            #     ^^^^^^^^
            if word1.poetry_word.upos == 'ADP' and word1.new_stress_pos > 0 and word2.poetry_word.upos in ('NOUN', 'PROPN') and word2.new_stress_pos > 0:
                self.total_score *= 0.5
                self.penalties.append('@317')

        for word1, word2, word3 in zip(self.stressed_words, self.stressed_words[1:], self.stressed_words[2:]):
            # 29-12-2021 Более двух подряд безударных слов - штрафуем
            if word1.new_stress_pos == -1 and word2.new_stress_pos == -1 and word3.new_stress_pos == -1:
                # 03.08.2022 Бывают цепочки из трех слов, среди которых есть частицы вообще без гласных:
                # я́ ж ведь не распла́чусь
                #   ^^^^^^^^^
                # Такие цепочки не штрафуем.
                if count_vowels(word1.poetry_word.form) > 0 and count_vowels(word2.poetry_word.form) > 0 and count_vowels(word3.poetry_word.form) > 0:
                    self.total_score *= 0.1
                    self.penalties.append('@323')

        # 28-12-2021 штрафуем за подряд идущие короткие слова (1-2 буквы)
        #for word1, word2, word3 in zip(stressed_words, stressed_words[1:], stressed_words[2:]):
        #    if word1.is_short_word() and word2.is_short_word() and word3.is_short_word():
        #        self.total_score *= 0.2

        if sum(self.stress_signature) == 1:
            # Всего один ударный слог в строке... Очень странно.
            # 〚Что за недоразуме́нье〛
            # 00000010
            # 07.06.2022 Но если в строке всего одно слово или группа предлог+сущ - это нормально!
            if len(self.poetry_line.pwords) > 2 or (len(self.poetry_line.pwords) == 2 and self.poetry_line.pwords[-2].upos != 'ADP'):
                self.total_score *= 0.1
                self.penalties.append('@335')
        else:
            # 01-01-2022 Детектируем разные сбои ритма
            #if self.stress_signature_str in aligner.bad_signature1:
            #    self.total_score *= 0.1
            #    self.penalties.append('@340')
            pass

    def init_rhyming_tail(self):
        stressed_word = None
        unstressed_prefix = None
        unstressed_postfix_words = []

        # Ищем справа слово с ударением
        i = len(self.stressed_words)-1
        while i >= 0:
            pword = self.stressed_words[i]
            if pword.new_stress_pos != -1 or count_vowels(pword.poetry_word.form) > 1:
                stressed_word = pword

                if pword.poetry_word.form in 'аеёиоуыэюя':
                    # Ситуация, когда рифмуется однобуквенное слово, состоящее из гласной:
                    #
                    # хочу́ отшлё́пать анако́нду
                    # но непоня́тно по чему́
                    # вот у слона́ гора́здо ши́ре
                    # чем у                       <=======
                    if i > 0:
                        unstressed_prefix = self.stressed_words[i-1].poetry_word.form[-1].lower()

                # все слова, кроме пунктуации, справа от данного сформируют безударный хвост клаузуллы
                for i2 in range(i+1, len(self.stressed_words)):
                    if self.stressed_words[i2].poetry_word.upos != 'PUNCT':
                        unstressed_postfix_words.append(self.stressed_words[i2])

                break
            i -= 1

        self.rhyming_tail = RhymingTail(unstressed_prefix, stressed_word, unstressed_postfix_words)

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

    #def get_last_rhyming_word(self):
    #    # Вернем последнее слово в строке, которое надо проверять на рифмовку.
    #    # Финальную пунктуацию игнорируем.
    #    for pword in self.stressed_words[::-1]:
    #        if pword.poetry_word.upos != 'PUNCT':
    #            return pword
    #    return None
    def get_rhyming_tail(self):
        return self.rhyming_tail


def count_vowels(s):
    return sum((c.lower() in 'уеыаоэёяию') for c in s)


def locate_Astress_pos(s):
    stress_pos = -1
    n_vowels = 0
    for c in s:
        if c.lower() in 'уеыаоэёяию':
            n_vowels += 1

        if c in 'АЕЁИОУЫЭЮЯ':
            stress_pos = n_vowels
            break

    return stress_pos


class PoetryLine(object):
    def __init__(self):
        self.text = None
        self.pwords = None

    @staticmethod
    def build(text, udpipe_parser, accentuator):
        pline = PoetryLine()
        pline.text = text
        pline.pwords = []

        text2 = text

        # Отбиваем некоторые симполы пунктуации пробелами, чтобы они гарантировано не слиплись со словом
        # в токенизаторе UDPipe/Stanza.
        for c in '\'.‚,?!:;…-–—«»″”“„‘’`ʹ"˝[]‹›·<>*/=()+®©‛¨×№\u05f4':
            text2 = text2.replace(c, ' ' + c + ' ').replace('  ', ' ')

        parsings = udpipe_parser.parse_text(text2)
        if parsings is None:
            raise ValueError('Could not parse text: ' + text2)

        for parsing in parsings:
            # Если слово в строке всего одно, то частеречная разметка не будет нормально работать.
            # В этом случае мы просто берем все варианты ударения для этого единственного слова.
            nw = sum(t.upos != 'PUNCT' for t in parsing)
            if nw == 1:
                for ud_token in parsing:
                    word = ud_token.form.lower()
                    alt_stress_pos = []

                    if word in accentuator.ambiguous_accents2:
                        ax = accentuator.ambiguous_accents2[word]
                        for a1 in ax:
                            i = locate_Astress_pos(a1)
                            alt_stress_pos.append(i)
                        stress_pos = alt_stress_pos[0]
                    elif word in accentuator.ambiguous_accents:
                        # Слово является омографом. Возьмем в работу все варианты ударения.
                        for stressed_form, tagsets in accentuator.ambiguous_accents[word].items():
                            i = locate_Astress_pos(stressed_form)
                            alt_stress_pos.append(i)
                        stress_pos = alt_stress_pos[0]
                    else:
                        # 22.06.2022 для OOV делаем перебор всех вариантов ударности для одиночного слова в артишоках.
                        # При этом не используется функционал автокоррекции искажения и опечаток в ударяторе.
                        if word not in accentuator.word_accents_dict and any((c in word) for c in 'аеёиоуыэюя'):
                            n_vowels = 0
                            for c in word:
                                if c.lower() in 'уеыаоэёяию':
                                    n_vowels += 1
                                    alt_stress_pos.append(n_vowels)

                            stress_pos = alt_stress_pos[0]
                        else:
                            stress_pos = accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])

                    pword = PoetryWord(ud_token.lemma, ud_token.form, ud_token.upos, ud_token.tags, stress_pos, alt_stress_pos)
                    pline.pwords.append(pword)
            else:
                for ud_token in parsing:
                    word = ud_token.form.lower()
                    stress_pos = accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])
                    alt_stress_pos = []
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
                            # Мы можем оказаться тут из-за ошибки частеречной разметки. Проверим,
                            # входит ли слово в список омографов.
                            if word in accentuator.ambiguous_accents:
                                # Слово является омографом. Возьмем в работу все варианты ударения.
                                for stressed_form, tagsets in accentuator.ambiguous_accents[word].items():
                                    i = locate_Astress_pos(stressed_form)
                                    alt_stress_pos.append(i)
                                stress_pos = alt_stress_pos[0]

                            if stress_pos == -1:
                                msg = 'Could not locate stress position in word "{}"'.format(word)
                                raise ValueError(msg)

                    pword = PoetryWord(ud_token.lemma, ud_token.form, ud_token.upos, ud_token.tags, stress_pos, alt_stress_pos)
                    pline.pwords.append(pword)

        pline.locate_rhyming_word()
        return pline

    @staticmethod
    def build_from_markup(markup_line, parser):
        pline = PoetryLine()
        pline.text = markup_line.replace('\u0301', '')
        pline.pwords = []

        text2 = markup_line
        for c in '.,?!:;…-–—«»”“„‘’`"':
            text2 = text2.replace(c, ' ' + c + ' ').replace('  ', ' ')

        # Надо поискать в исходной размеченной строке наше слово. Одинаковое слово может встретится 2 раза с разной
        # разметкой.
        line_spans = [(span.replace('\u0301', ''), span) for span in re.split(r'[.?,!:;…\-\s]', markup_line)]

        # удаляем расставленные ударения и выполняем полный анализ.
        parsings = parser.parse_text(text2.replace('\u0301', ''))

        for parsing in parsings:
            for ud_token in parsing:
                # определяем позицию гласного, помеченного как ударный.
                stress_pos = -1

                n_vowels = 0
                for ic, c in enumerate(ud_token.form):
                    if c.lower() in 'уеыаоэёяию':
                        # Нашли гласную. попробуем сделать ее ударной (поставить справа юникодный маркер ударения)
                        # и поискать в исходной разметке.
                        n_vowels += 1
                        needle = ud_token.form[:ic+1] + '\u0301' + ud_token.form[ic+1:]
                        # Ищем слева направо в сегментах с разметкой. Если слово найдется в сегменте, то его
                        # исключим из дальнейшего поиска, чтобы справится с ситуацией разного ударения в одинаковых словах.
                        for ispan, (span_form, span_markup) in enumerate(line_spans):
                            if span_form == ud_token.form and span_markup == needle:
                                stress_pos = n_vowels
                                line_spans = line_spans[:ispan] + line_spans[ispan+1:]
                                break

                        if stress_pos == -1:
                            # могут быть MWE с дефисом, которые распались на отдельные слова. Их ищем в исходной
                            # разметке целиком.
                            if re.search(r'\b' + needle.replace('-', '\\-').replace('.', '\\.') + r'\b', markup_line):
                                stress_pos = n_vowels
                                break

                        if stress_pos != -1:
                            break

                pword = PoetryWord(ud_token.lemma, ud_token.form, ud_token.upos, ud_token.tags, stress_pos)
                pline.pwords.append(pword)

        pline.locate_rhyming_word()
        return pline

    def locate_rhyming_word(self):
        # Отмечаем последнее слово в строке, так как оно должно ударяться, за исключением
        # очень редких случаев:
        # ... я же
        # ... ляжет
        located = False
        for pword in self.pwords[::-1]:
            if pword.upos != 'PUNCT':
                pword.is_rhyming_word = True
                located = True
                break
        if not located:
            msg = 'Could not locate rhyming word in line ' + self.text
            raise ValueError(msg)

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
                if len(colloc) == 2:
                    for i1, (w1, w2) in enumerate(zip(lwords, lwords[1:])):
                        if colloc.hit2(w1, w2):
                            # из всех вариантов в variants делаем еще по 1 варианту
                            for variant in variants:
                                v = colloc.produce_stressed_line(variant, aligner)
                                add_variants.append(v)

            if add_variants:
                variants.extend(add_variants)

        # 04-08-2022 добавляем варианты для триграмм типа "бок О бок"
        if any((w in aligner.collocation3_first) for w in lwords) and any((w in aligner.collocation3_second) for w in lwords) and any((w in aligner.collocation3_third) for w in lwords):
            # В строке возможно присутствует одно из особых словосочетаний длиной 3
            add_variants = []
            for colloc in aligner.collocations:
                if len(colloc) == 3:
                    for i1, (w1, w2, w3) in enumerate(zip(lwords, lwords[1:], lwords[2:])):
                        if colloc.hit3(w1, w2, w3):
                            # из всех вариантов в variants делаем еще по 1 варианту
                            for variant in variants:
                                v = colloc.produce_stressed_line(variant, aligner)
                                add_variants.append(v)

            if add_variants:
                variants.extend(add_variants)

        return variants

    def get_first_stress_variants(self, aligner):
        swords = [pword.get_first_stress_variant() for pword in self.pwords]
        return LineStressVariant(self, swords, aligner)


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


# Мы проверяем только эти 5 вариантов чередования ударных и безударных слогов.
# Более сложные случаи отбрасываем, они слишком тяжелы для восприятия.
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

    def __len__(self):
        return len(self.words)

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

    def hit2(self, word1, word2):
        return self.words[0] == word1 and self.words[1] == word2

    def hit3(self, word1, word2, word3):
        return self.words[0] == word1 and self.words[1] == word2 and self.words[2] == word3

    def produce_stressed_line(self, src_line, aligner):
        nw1 = len(src_line.stressed_words) - 1
        nw2 = len(src_line.stressed_words) - 2
        for i1, word1 in enumerate(src_line.stressed_words):
            if word1.poetry_word.form.lower() == self.words[0]:
                if i1 < nw1:
                    word2 = src_line.stressed_words[i1+1]
                    if word2.poetry_word.form.lower() == self.words[1]:
                        new_stressed_words = list(src_line.stressed_words[:i1])

                        if len(self.words) == 2:
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
                        elif len(self.words) == 3:
                            if i1 < nw2:
                                word3 = src_line.stressed_words[i1 + 2]
                                if word3.poetry_word.form.lower() == self.words[2]:

                                    if self.stressed_word_index == 0:
                                        # первое слово становится ударным, второе и третье - безударные
                                        new_stressed_words.append(word1.build_stressed(self.stress_pos))
                                        new_stressed_words.append(word2.build_unstressed())
                                        new_stressed_words.append(word3.build_unstressed())
                                    elif self.stressed_word_index == 1:
                                        # первое и третье слова становятся безударными, второе - ударное
                                        new_stressed_words.append(word1.build_unstressed())
                                        new_stressed_words.append(word2.build_stressed(self.stress_pos))
                                        new_stressed_words.append(word3.build_unstressed())
                                    else:
                                        # первое и второе слова становятся безударными, третье - ударное
                                        new_stressed_words.append(word1.build_unstressed())
                                        new_stressed_words.append(word2.build_unstressed())
                                        new_stressed_words.append(word3.build_stressed(self.stress_pos))

                                    # остаток слов справа от третьего слова
                                    new_stressed_words.extend(src_line.stressed_words[i1 + 3:])
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

        self.collocation3_first = set()
        self.collocation3_second = set()
        self.collocation3_third = set()

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
                elif len(c.words) == 3:
                    # для словосочетаний длиной 3.
                    self.collocation3_first.add(c.words[0])
                    self.collocation3_second.add(c.words[1])
                    self.collocation3_third.add(c.words[2])

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
        self.allow_fuzzy_rhyming = True

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

    def get_spectrum(pline):
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
            raise ValueError("Alignment is not implemented for {}-liners! text={}".format(len(lines), '\n'.join(lines)))

    def check_rhyming(self, rhyming_tail1, rhyming_tail2):
        if not rhyming_tail1.is_ok() or not rhyming_tail2.is_ok():
            return False

        poetry_word1 = rhyming_tail1.stressed_word
        poetry_word2 = rhyming_tail2.stressed_word

        if rhyming_tail1.is_simple() and rhyming_tail2.is_simple():
            # 01.02.2022 проверяем слова с ударениями по справочнику рифмовки
            f1 = poetry_word1.stressed_form
            f2 = poetry_word2.stressed_form
            if (f1, f2) in self.accentuator.rhymed_words or (f2, f1) in self.accentuator.rhymed_words:
                return True

            # Считаем, что слово не рифмуется само с собой
            if poetry_word1.form.lower() == poetry_word2.form.lower():
                return False

        unstressed_tail1 = rhyming_tail1.unstressed_tail
        unstressed_tail2 = rhyming_tail2.unstressed_tail

        r = rhymed2(self.accentuator,
                    poetry_word1.poetry_word.form, poetry_word1.new_stress_pos, [poetry_word1.poetry_word.upos] + poetry_word1.poetry_word.tags, rhyming_tail1.prefix, unstressed_tail1,
                    poetry_word2.poetry_word.form, poetry_word2.new_stress_pos, [poetry_word2.poetry_word.upos] + poetry_word2.poetry_word.tags, rhyming_tail2.prefix, unstressed_tail2)
        if r:
            return True

        if self.allow_fuzzy_rhyming:
            return rhymed_fuzzy2(self.accentuator,
                                poetry_word1.poetry_word.form, poetry_word1.new_stress_pos, [poetry_word1.poetry_word.upos] + poetry_word1.poetry_word.tags, rhyming_tail1.prefix, unstressed_tail1,
                                poetry_word2.poetry_word.form, poetry_word2.new_stress_pos, [poetry_word2.poetry_word.upos] + poetry_word2.poetry_word.tags, rhyming_tail2.prefix, unstressed_tail2)

        return False

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

    def align_AABA(self, lines):
        plines = [PoetryLine.build(line, self.udpipe, self.accentuator) for line in lines]

        # Список вариантов простановки ударения с учётом опциональности ударений для союзов, предлогов и т.д.
        plinevx = [pline.get_stress_variants(self) for pline in plines]

        # Идем по списку вариантов, отображаем на эталонные метры и ищем лучший вариант.
        best_variant = None
        best_score = -1.0
        best_meter = None
        best_ivar = None
        best_rhyme_scheme = None

        total_permutations = reduce(lambda x, y: x * y, [len(z) for z in plinevx])
        if total_permutations > 10000:
            raise ValueError('Too many optional stresses: {}'.format(total_permutations))

        vvx = list(itertools.product(*plinevx))

        for ivar, plinev in enumerate(vvx):
            # plinev это набор из четырех экземпляров LineStressVariant.

            last_pwords = [pline.get_rhyming_tail() for pline in plinev]
            r01 = self.check_rhyming(last_pwords[0], last_pwords[1])
            r13 = self.check_rhyming(last_pwords[1], last_pwords[3])
            r02 = self.check_rhyming(last_pwords[0], last_pwords[2])

            if r01 and r13 and not r02:
                #score1234, mapped_meter = self._align_line_groups([(plinev[0], plinev[1], plinev[2], plinev[3])])
                score1234, mapped_meter = self._align_line_groups([[plinev[0]], [plinev[1]], [plinev[2]], [plinev[3]]])

                score = score1234 * reduce(lambda x, y: x*y, [l.get_score() for l in plinev])
                if score > best_score:
                    best_variant = plinev
                    best_score = score
                    best_meter = mapped_meter
                    best_ivar = ivar
                    best_rhyme_scheme = 'AABA'

        if best_variant is None:
            return PoetryAlignment.build_no_rhyming_result([pline.get_stress_variants(self)[0] for pline in plines])
        else:
            return PoetryAlignment(best_variant, best_score, best_meter, best_rhyme_scheme)

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
            rhyme_scheme = '----'
            rhyming_score = 0.0
            mapped_meter = None

            # проверяем все пары слов
            last_pwords = [pline.get_rhyming_tail() for pline in plinev]
            r01 = self.check_rhyming(last_pwords[0], last_pwords[1])
            r02 = self.check_rhyming(last_pwords[0], last_pwords[2])
            r03 = self.check_rhyming(last_pwords[0], last_pwords[3])
            r12 = self.check_rhyming(last_pwords[1], last_pwords[2])
            r13 = self.check_rhyming(last_pwords[1], last_pwords[3])
            r23 = self.check_rhyming(last_pwords[2], last_pwords[3])

            # 22.04.2022 отдельно детектируем рифмовку AAAA, так как она зачастую выглядит очень неудачно и ее
            # желательно устранять из обучающего датасета.
            if r01 and r12 and r23:
                rhyme_scheme = 'AAAA'
            elif r02 and r13:
                rhyme_scheme = 'ABAB'
            elif r03 and r12:
                rhyme_scheme = 'ABBA'
            # 22-12-2021 добавлена рифмовка AABB
            elif r01 and r23:
                rhyme_scheme = 'AABB'
            # 28-12-2021 добавлена рифмовка "рубаи" AABA
            elif r01 and r03 and not r02:
                rhyme_scheme = 'AABA'
            # 21.05.2022 проверяем неполные рифмовки A-A- и -A-A
            elif r02 and not r13:
                rhyme_scheme = 'A-A-'
            elif not r02 and r13:
                rhyme_scheme = '-A-A'
            else:
                rhyme_scheme = '----'

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
            elif rhyme_scheme == 'A-A-':
                score1234, mapped_meter = self._align_line_groups([(plinev[0], plinev[2]), (plinev[1], plinev[3])])
                rhyming_score = 1.0 - COEFF['@225']*(1.0 - pow(score1234, 0.5))
            elif rhyme_scheme == '-A-A':
                score1234, mapped_meter = self._align_line_groups([(plinev[0], plinev[2]), (plinev[1], plinev[3])])
                rhyming_score = 1.0 - COEFF['@225']*(1.0 - pow(score1234, 0.5))
            elif rhyme_scheme == 'AAAA':
                score1234, mapped_meter = self._align_line_groups([(plinev[0], plinev[1], plinev[2], plinev[3])])
                rhyming_score = 1.0 - COEFF['@225']*(1.0 - pow(score1234, 0.5))
            else:
                if check_rhymes:
                    #continue
                    score1234, mapped_meter = self._align_line_groups([(plinev[0],), (plinev[1],), (plinev[2],), (plinev[3],)])
                    rhyming_score = 0.1*(1.0 - COEFF['@225'] * (1.0 - pow(score1234, 0.5)))
                else:
                    score1234, mapped_meter = self._align_line_groups([(plinev[0], plinev[2]), (plinev[1], plinev[3])])
                    rhyming_score = 0.5*(1.0 - COEFF['@225'] * (1.0 - pow(score1234, 0.5)))

            score = rhyming_score * reduce(lambda x, y: x*y, [l.get_score() for l in plinev])

            # 27.06.2022 предпочитаем максимально зарифмованную разметку, поэтому прежде всего считаем, сколько строк
            # зарифмовано друг с другом, подсчитывая символы '-' в идентификаторе рифмовки.
            is_better = False
            if best_ivar is None:
                is_better = True
            elif best_rhyme_scheme.count('-') > rhyme_scheme.count('-'):
                is_better = True
            elif best_rhyme_scheme.count('-') == rhyme_scheme.count('-') and score > best_score:
                is_better = True

            if is_better:
                best_variant = plinev
                best_score = score
                best_meter = mapped_meter
                best_ivar = ivar
                best_rhyme_scheme = rhyme_scheme

        #if best_rhyme_scheme is None or best_rhyme_scheme == '----':
        #    if check_rhymes:
        #        # Не получилось подобрать рифмовку окончаний строк.
        #        # В этом случае вернем результат с нулевым скором и особым текстом, чтобы
        #        # можно было вывести в лог строки с каким-то дефолтными
        #        return PoetryAlignment.build_no_rhyming_result([pline.get_stress_variants(self)[0] for pline in plines])

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
            rhyme_score = 1.0

            last_pwords = [pline.get_rhyming_tail() for pline in plinev]
            if self.check_rhyming(last_pwords[0], last_pwords[1]):
                rhyme_scheme = 'AA'
            else:
                rhyme_scheme = '--'
                rhyme_score = 0.5

            score_12 = self.map_2signatures(plinev[0], plinev[1])

            # Ищем лучшее отображение метра.
            mapped_meter, mapping_score = self.map_meters(plinev)
            score = rhyme_score * score_12 * mapping_score * reduce(lambda x, y: x*y, [l.get_score() for l in plinev])
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

    def build_from_markup(self, text):
        lines = text.split('\n')

        plines = [PoetryLine.build_from_markup(line, self.udpipe) for line in lines]
        stressed_lines = [pline.get_first_stress_variants(self) for pline in plines]

        mapped_meter, mapping_score = self.map_meters(stressed_lines)
        score = mapping_score * reduce(lambda x, y: x * y, [l.get_score() for l in stressed_lines])

        return PoetryAlignment(stressed_lines, score, mapped_meter, rhyme_scheme='')

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
                if w1.form.lower() == w2.form.lower() and w1.form[0] not in '.!?':
                    return True

            # также штрафуем за паттерн "XXX и XXX"
            for w1, w2, w3 in zip(pline.poetry_line.pwords, pline.poetry_line.pwords[1:], pline.poetry_line.pwords[2:]):
                if w2.form.replace('\u0301', '') in ('и', ',', 'или', 'иль', 'аль', 'да'):
                    if w1.form.replace('\u0301', '').lower() == w3.form.replace('\u0301', '').lower() and w1.form.replace('\u0301', '') not in 'вновь еще ещё снова опять дальше ближе сильнее слабее сильней слабей тише'.split(' '):
                        return True

        return False

    def detect_poor_poetry(self, alignment):
        """Несколько эвристик для обнаружения скучных рифм, которые мы не хотим получать"""

        last_words = [pline.get_rhyming_tail().stressed_word.form.lower() for pline in alignment.poetry_lines]

        # Если два глагольных окончания, причем одно является хвостом другого - это бедная рифма:
        # ждать - подождать
        # смотреть - посмотреть
        # etc.
        rhyme_pairs = []
        if alignment.rhyme_scheme == 'ABAB':
            rhyme_pairs.append((alignment.poetry_lines[0].get_rhyming_tail(), alignment.poetry_lines[2].get_rhyming_tail()))
            rhyme_pairs.append((alignment.poetry_lines[1].get_rhyming_tail(), alignment.poetry_lines[3].get_rhyming_tail()))
        elif alignment.rhyme_scheme == 'ABBA':
            rhyme_pairs.append((alignment.poetry_lines[0].get_rhyming_tail(), alignment.poetry_lines[3].get_rhyming_tail()))
            rhyme_pairs.append((alignment.poetry_lines[1].get_rhyming_tail(), alignment.poetry_lines[2].get_rhyming_tail()))
        elif alignment.rhyme_scheme == 'AABA':
            rhyme_pairs.append((alignment.poetry_lines[0].get_rhyming_tail(), alignment.poetry_lines[1].get_rhyming_tail()))
            rhyme_pairs.append((alignment.poetry_lines[0].get_rhyming_tail(), alignment.poetry_lines[3].get_rhyming_tail()))
        elif alignment.rhyme_scheme == 'AABB':
            rhyme_pairs.append((alignment.poetry_lines[0].get_rhyming_tail(), alignment.poetry_lines[1].get_rhyming_tail()))
            rhyme_pairs.append((alignment.poetry_lines[2].get_rhyming_tail(), alignment.poetry_lines[3].get_rhyming_tail()))

        for tail1, tail2 in rhyme_pairs:
            word1 = tail1.stressed_word
            word2 = tail2.stressed_word
            if word1.poetry_word.upos == 'VERB' and word2.poetry_word.upos == 'VERB':
                # 11-01-2022 если пара слов внесена в специальный список рифмующихся слов, то считаем,
                # что тут все нормально:  ВИТАЮ-ТАЮ
                form1 = word1.poetry_word.form.lower()
                form2 = word2.poetry_word.form.lower()
                if (form1, form2) in self.accentuator.rhymed_words:
                    continue

                if any((form1.endswith(e) and form2.endswith(e)) for e in 'ли ла ло л м шь т тся у те й ю ь лись лась лось лся тся ться я шись в'.split(' ')):
                    return True

                if form1.endswith(form2):
                    return True
                elif form2.endswith(form1):
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

    #x = accents.get_accent('самоцветы')
    #print(x)

    # ================================================

    #alignment = aligner.build_from_markup('Без му́́ки не́т и нау́ки.')
    #print('\n' + alignment.get_stressed_lines() + '\n')

    text = """Во все, что ты верил, не бывши таким,
Растоптали, сожгли, разорвали,
Завидуя в чем-то, пожалуй, другим,
Которым так жизнь не ломали."""

    poem = [z.strip() for z in text.split('\n') if z.strip()]
    alignment = aligner.align(poem, check_rhymes=False)
    # print(alignment)
    print('is_poor={}'.format(aligner.detect_poor_poetry(alignment)))
    print('='*80)

    # ====================================================
    #       ===== АВТОТЕСТЫ РАЗМЕТЧИКА СТИХОВ =====

    true_markups = [
        ("""ми́р разгова́ривает с на́ми
а мы́ по пре́жнему немы́
и до сих по́р не зна́ем са́ми
кто мы́""", 'ABAB'),

        ("""Откупо́рил, налива́ю
Бу́льки ду́шу ворожа́т
Предвкуша́я, пря́мо та́ю
Как ребё́нок ма́ме, ра́д""", "ABAB"),

        ("""Их куку́шка на крюку́
Ме́рит вре́мя старику́
По часо́чку, по деньку́
Ме́лет го́дики в муку́""", "AAAA"),

        ("""Чем се́кс с тобо́й, уж лу́чше телеви́зор
Иль на худо́й коне́ц нажра́ться антифри́за.""", "AA"),

        ("""два́ лингви́ста спо́ря тво́рог иль творо́г
мо́рдами друг дру́га би́ли об поро́г""", "AA"),

        ("""кто́ не лю́бит ка́ши ма́нной —
то́т како́й-то о́чень стра́нный!""", "AA"),

        ("""а моя́ любо́вь на
со́том этаже́
я́ заколеба́юсь
поднима́ться же́""", "-A-A"),

        ("""Хо́ть бы ра́з
Поли́л наро́чно
С не́ба в блю́дце
До́ждь моло́чный!""", "-A-A"),

        ("""До́ждь иде́т,
Пото́ки лью́тся,
Че́рный ко́т
Гляди́т на блю́дце.""", "ABAB"),

        ("""А где хра́мы и попы́
Там к нау́ке все слепы́
Окропя́т, махну́т кади́лом
Так заго́нят всех в моги́лу""", "AABB"),

        ("""Но дурака́м везё́т, и э́то то́чно
Хотя́ никто́ не зна́ет наперё́д
А получи́лось всё насто́лько про́чно
Что никака́я си́ла не порвё́т""", "ABAB"),

        ("""Э́то де́йствие просто́е
        Наблюда́ем ка́ждый го́д
        Да́же ста́рым не́т поко́я
        Чё́рте что на у́м идё́т""", "ABAB"),

        ("""С той поры́, когда́ черну́ха
На душе́, с овчи́нку све́т
Только вспо́мню ту стару́ху
Так хандры́ в поми́не не́т""", "ABAB"),

        ("""И ми́р предста́л как в стра́шной ска́зке
Пусты́ доро́ги про́бок не́т
Все у́лицы наде́ли ма́ски
Тако́й вот бра́т кордебале́т""", "ABAB"),

        ("""Руко́ю ле́вой бо́льше шевели́те
Чтоб ду́мали, что Вы ещё́ всё жи́вы
А э́ти во́семь та́ктов - не дыши́те
Умри́те! Не игра́йте так фальши́во""", "ABAB"),

        ("""Сказа́л серьё́зно так, коле́но преклони́в
Что он влюбле́н давно́, души́ во мне́ не ча́ет
Он в изоля́ции, как у́зник за́мка И́ф
Нас бы́стро в це́ркви виртуа́льной повенча́ют""", "ABAB"),

        ("""Зате́м каре́та бу́дет с тро́йкой лошаде́й
Банке́т, шампа́нское, солье́мся в та́нце ба́льном
Пото́м отпу́стим в небеса́ мы голубе́й
И бу́дем вме́сте, навсегда́! Но... виртуа́льно""", "ABAB"),

        ("""Здра́вствуй До́ня белогри́вый
Получи́лось как-то ло́вко
Е́сли че́стно, некраси́во
Объего́рил сно́ва Во́вку""", "ABAB"),

        ("""Ты́ для нас почти́ что сво́й
        Бы́ло де́ло кры́ли ма́том
        Сче́т на вре́мя им закро́й
        Что́бы ду́мали ребя́та""", "ABAB"),

        ("""И труба́ прое́кту с га́зом
У тебя́ все намази́
Возверне́м реа́льность сра́зу
Ты им ви́рус привези́""", "ABAB"),

        ("""Даны́ кани́кулы – во бла́го
И, что́бы вре́мя не теря́ть
Мы на Парна́с упря́мым ша́гом
Стихи́ отпра́вимся писа́ть""", "ABAB"),

        ("""Ста́л он расспра́шивать сосе́да
Ведь на рыба́лку, хоть с обе́да
Ухо́дит тот, иль у́тром ра́но
И не вопи́т супру́га рья́но""", "AABB"),

        ("""Но де́ло в том, что спе́реди у Да́мы
Для гла́з мужски́х есть ва́жные места́
И чтоб поздне́е не случи́лось дра́мы
Вы покажи́те зу́бки и уста́""", "ABAB"),

        ("""Я мы́ло де́тское себе́ купи́ла
Лицо́ и ру́ки тща́тельно им мы́ла
Вы не пове́рите, мои́ друзья́
В ребе́нка преврати́лась я́""", "AABB"),

        ("""То в поры́ве настрое́нья
Пля́шет в ди́ком упое́нье
То, и во́все вдруг курьё́зы
На стекле́ рису́ет слё́зы""", "AABB"),

        ("""А второ́й сосе́д - банди́т
Он на на́с и не гляди́т
Ве́чно хо́дит под охра́ной
Му́тный би́знес шуруди́т""", "AABA"),

        ("""Где застря́л жени́х мой
Где? Жду́ его́ в трево́ге
Не подо́х ли бе́лый ко́нь
Где́-нибудь в доро́ге""", "-A-A"),

        ("""Прошу́ Вас, не дари́те мне цветы́
И не в почё́те ны́нче самоцве́ты
Ки́ньте в меня́ кусо́чком колбасы́
Идё́т четвё́ртый ча́с мое́й дие́ты""", "-A-A"),

        ("""Быть гли́ною – блаже́ннейший уде́л
Но всё́ ж в рука́х Творца́, покры́тых сла́вой
Нам не пости́чь Его́ вели́ких де́л
Оди́н Госпо́дь твори́ть име́ет пра́во""", "ABAB"),

        ("""Тума́ня ра́зум за́пахом ело́вым
Из нового́дних ска́зочных чуде́с
Друзе́й и ма́му мне напо́мнил ле́с
Не утеша́я, но верну́в к осно́вам""", "ABBA"),

        ("""Увы́, нельзя́ от них спасти́сь
крепка́ петля́ гипно́за
пока моги́лой смо́трит ввы́сь
твоя́ метаморфо́за""", "ABAB"),

        ("""Любо́й рождё́н для по́иска и ри́ска
Любо́му сча́стье хо́чется найти́
Ины́м суха́рик вместо ка́ши в ми́ске
Други́м полё́т по Мле́чному Пути́""", "ABAB"),

        ("""Беспе́чен ле́с, не слы́ша топора́
Дове́рчив, как младе́нец в колыбе́ли
То укача́в, то гла́дя еле - е́ле
Игра́ют с ним весё́лые ветра́""", "ABBA"),

        ("""Быть гли́ною – блаже́ннейший уде́л
Но всё́ ж в рука́х Творца́, покры́тых сла́вой
Нам не пости́чь Его́ вели́ких де́л
Оди́н Госпо́дь твори́ть име́ет пра́во""", "ABAB"),

        ("""Уже́ побыва́ть не вперво́й
На звё́здах разли́чного я́руса
Прия́тно с подзо́рной трубо́й
Стоя́ть под натя́нутым па́русом""", "ABAB"),

        ("""Но что́ же како́е-то чу́вство щемя́щее
        как - будто сего́дня после́дняя встре́ча у на́с
        секу́нды, мину́ты, как пти́цы паря́щие
        то ко́смоса шё́пот, то ве́тра внеза́пного ба́с""", "ABAB"),

        ("""Бо́ль из про́шлых дне́й остра́
        Ла́вры вновь несё́т на блю́де
        От утра́ и до утра́
        Ды́шит та́кже, как и лю́ди""", "ABAB"),

        ("""пролета́ет ле́то
        гру́сти не тая́
        и аналоги́чно
        пролета́ю я́""", "-A-A"),

        ("""хардко́р на у́лице сосе́дней
вчера́ ната́ша умерла́
и никола́ю наконе́ц то
дала́""", "-A-A"),

        ("""он удира́л от нас двора́ми
        едва успе́в сказа́ть свекла́
        филфа́к не ви́дывал тако́го
        ссыкла́""", "-A-A"),

        ("""я проводни́к электрото́ка
        зажгу́ две́ ла́мпочки в носу́
        как только но́жницы в розе́тку
        засу́""", "-A-A"),

        ("""узна́й вы из како́го со́ра
        поро́ю пи́шутся стихи́
        свои́ б засу́нули пода́льше
        имхи́""", "-A-A"),

        ("""аэроста́ты цеппели́ны
        и гво́здь програ́ммы ле́ бурже́
        после́дний вы́дох господи́на
        пэжэ́""", "ABAB"),

        ("""всех тех кого́ хоте́л уво́лить
        от зло́сти проломи́в насти́л
        прора́б в полё́те трёхмину́тном
        прости́л""", "-A-A"),

        ("""оле́г за жи́знью наблюда́ет
        и ду́мает ещё́ б разо́к
        пересмотре́ть её́ снача́ла
        в глазо́к""", "-A-A"),

        ("""ко мне́ во сне́ яви́лась ю́ность
        про жи́знь спроси́ла про семью́
        и до́лго пла́кала в поду́шку
        мою́""", "-A-A"),

        #("""что тру́дно мне дало́сь на сва́дьбе
        #так э́то де́лать ви́д что я́
        #не ви́жу у отца́ неве́сты
        #ружья́""", "-A-A"),

        ("""меж на́ми пробежа́ла и́скра
         а у тебя́ в рука́х кани́стра""", 'AA'),

        ("""риску́я показа́ться гру́бым
         бе́гу за ва́ми с ледору́бом""", "AA"),

        ("""бы́л в се́ксе о́чень виртуо́зен
        но ли́шь в миссионе́рской по́зе""", "AA"),

        ("""из но́рки вы́сунув еба́льце
        мне бурунду́к отгры́з два́ па́льца""", "AA"),

        ("""от ушеспание́льной си́ськи
        гле́б вы́жрал во́дку ро́м и ви́ски""", "AA"),

        ("""фура́жка с ге́рбом ство́л и кси́ва
        лицо́ печа́льно но́ краси́во""", "AA"),

        ("""про меня́ вы лжё́те
        вся́ческую гну́сь
        я́ ж ведь не распла́чусь
        я́ ж ведь расплачу́сь""", "-A-A"),

        ("""на сва́дьбе бы́вшего супру́га
        на би́с спляса́ла гопака́
        криви́лась но́вая вот су́ка
        кака́""", "-A-A"),

        ("""слеза́ прокля́той комсомо́лки
        прожгла́ исто́рию наскво́зь
        и ходорко́вскому на те́мя
        свинцо́вой ка́плей селяви́""", "----"),

        ("""скрипя́ зуба́ми шестерё́нки
        живу́т бок о́ бок без любви́""", "--"),

        ("""я мо́лча загора́л на ка́мне
        а вы́ по па́льцам с молотка́ мне""", "AA"),

        ("""барме́н нале́йте ка мохи́то
        а то́ не тя́нет на грехи́ то""", "AA"),

        ("""голки́пер че́шский не меша́й бы
        забро́сили б ещё́ две́ ша́йбы""", "AA"),

        ("""ми́р электри́чества бичу́я
        иду́ держа́ в руке́ свечу́ я""", "AA"),

        ("""с тобо́ю жи́ли мы́ бок о́ бок
        веще́й нажи́ли пя́ть коро́бок""", "AA"),

        ("""не смо́г ма́г сня́ть вене́ц безбра́чья
        ведь развожу́ кото́в и сра́ч я""", "AA"),

        ("""два́ ме́рса джи́п четы́ре во́львы
        совсе́м сбеси́лись с жи́ру что́ ль вы""", "AA"),

        ("""а то́чно у тебя три́ я́хты
        живе́шь в хруще́вке в ебеня́х ты""", "AA"),

        ("""в а́ли экспре́сс и на ави́то
иска́л любо́вь но нет любви́ то""", "AA"),

        ("""хоте́л в камо́рку я зайти́ но
        чу́ там струга́ют бурати́но""", "AA"),

        ("""ба́б мно́го чи́стых и́ наи́вных
а мы́ скоты́ суе́м хуи́ в них""", "AA"),

        ("""Расскажу́ про молодё́жь
        Не хоте́лось бы, но всё́ ж
        Гра́мот о́трок не чита́ет
        А уда́рился в балдё́ж""", "AABA"),

        ("""я нё́с петро́ву на лине́йке
        и у меня́ смеще́нье гры́ж
        а ведь петро́вой говори́ли
        не жри́ ж""", '-A-A'),

        ("""я с тоско́й гляжу́ на
        де́вушек вдали́
        на лицо́ мне ю́ность
        ко́льщик наколи́""", "-A-A"),

        ("""я распахну́л ду́ши бушла́ты
        но всё равно́ с други́м ушла́ ты""", "AA"),

        ("""к восьмо́му ма́рта одни́ тра́ты
        наде́юсь что́ уйдё́шь сама́ ты""", "AA"),

        ("""не сто́ль криво́й была́ судьба́ бы
        когда́ бы по́нял я су́ть ба́бы""", "AA"),

        ("""гле́б у жены́ проси́л поща́ды
        ну сле́зь с меня́ свари́ борща́ ты""", "AA"),

        ("""чита́ет ста́рая ари́на
        для арапчо́нка на фарси́
        перевели́сь богатыри́ на
        руси́""", "ABAB"),

        #("""ни желтизна́ зубо́в ни ба́с и
        #ни даже чё́рные усы́
        #зухру́ весно́й смуща́ли то́лько
        #весы́""", "-A-A"),

        ("""Скала́. Хребе́т. Гора́. Или утё́с
        Не вертика́ль. Но всё́ же о́чень кру́то
        Како́й… поры́в её́ туда́ занё́с
        Одну́. И без страхо́вки почему́-то""", "ABAB"),

        ("""взя́л вместо во́дки молока́ я
        присни́тся же хуйня́ така́я""", "AA"),

        ("""два́ па́рных отыска́л носка́ я
        встреча́й краса́вчика тверска́я""", "AA"),

        ("""сня́ть се́лфи с ка́мнем на мосту́ ли
        иль сто́я под крюко́м на сту́ле""", "AA"),

        ("""не оттого́ ль мне одино́ко
        что лу́к е́м только и чесно́к а""", "AA"),

        ("""оле́г не дре́ль но почему́ то
        всё вре́мя све́рлит мо́зг кому́ то""", "AA"),

        ("""взя́л динами́т поджё́г фити́ль но
        так неуме́ло инфанти́льно""", "AA"),

        ("""хоть в фэ́ э́с бэ́ то вы́ могли́ бы
        не говори́ть что мы́ тали́бы""", "AA"),

        ("""лежа́т на кла́дбище не те́ ли
        чей ду́х бы́л здра́в в здоро́вом те́ле""", "AA"),

        ("""лети́т аню́та с парашю́том
        а не́т без парашю́та тьфу́ ты""", "AA"),

        ("""мы собира́лись плы́ть к вега́нам
        но ку́к сказа́л а нафига́ нам""", "AA"),

        ("""я ебану́л тебя́ весло́м бы
        но зна́ю дороги́е пло́мбы""", "AA"),

        ("""от вымира́нья ми́р спаса́я
        творю́ в посте́ли чудеса́ я""", "AA"),

        ("""передаё́м приве́т ано́нам
        а впро́чем на́до ли оно́ нам""", "AA"),

        ("""люблю́ ее́ и жму́сь побли́же
        да и прохла́дно не бали́ же""", "AA"),

        ("""не в жё́ны взя́л а для заба́вы
        на передо́к окса́н слаба́ вы""", "AA"),

        ("""бетхо́вен от насто́йки клю́квы
        и ба́ха спро́сит а не глю́к вы""", "AA"),

        ("""о всемогу́щий си́лы да́й нам
        в борьбе́ с засте́ночным раммшта́йном""", "AA"),

        ("""Но, уже́ тоску́я
        В ро́т тебя́ кладу́ я
        Так пробле́мы не́т
        Ста́л я людое́д""", "AABB"),

        ("""топоро́м не на́до
        балова́ть в лесу́
        заруби́л себе́ я
        э́то на носу́""", "-A-A"),

        ("""о граждани́н спи́рт пью́щий кто́ ты
исто́чник ра́дости ль иль рво́ты""", "AA"),

        ("""и заче́м вчера́ я
        вле́зла в э́тот спо́р
        на́до перепря́тать
        тру́пы и топо́р""", "-A-A"),

        ("""дре́ль изоле́нту пассати́жи
        отло́жь ма́ш с му́жем накати́ же""", "AA"),

        ("""гле́б закупи́л побо́льше брю́та
        с него́ изя́щнее хрю́ хрю́ то""", "AA"),

        ("""хоте́л взя́ть льго́тные креди́ты
        сказа́ли на́хуй мол иди́ ты""", "AA"),

        ("""жи́знь пролети́т мы все умре́м на
        и кто кути́л и кто жи́л скро́мно""", "AA"),

        ("""не понима́ю нихера́ я
        заче́м бо́х за́пер две́ри ра́я""", "AA"),

        ("""а на пове́стке дня́ две́ те́мы
        бюдже́т и почему́ в пизде́ мы""", "AA"),

        ("""вам спря́тать удало́сь в трико́ то
        ча́сть ту́ши ни́же антреко́та""", "AA"),

        ("""чте́ц что то мя́млил нам фальце́том
        та́к и не по́нял что в конце́ там""", "AA"),

        ("""гле́б пиздану́лся с эвере́ста
        у нас вопро́с заче́м поле́з то""", "AA"),

        ("""Я сра́зу осозна́л, что встре́тились не зря́ мы
        И вско́ре сбы́лось всё, о чём мечта́л
        Дово́льно ско́ро я к жилью́ прекра́сной да́мы
        Доро́жку вечера́ми протопта́л""", "ABAB"),

        ("""не бу́дь у ба́нь и у кофе́ен
        пиаротде́лов и прессслу́жб
        мы все б завши́вели и пи́ли
        из лу́ж б""", "-A-A"),

        ("""по суперма́ркету с теле́жкой
        броди́ла же́нщина сукку́б
        и донима́ла консульта́нтов
        муку́ б""", "-A-A"),

        ("""необосно́ванная ве́ра
        в своё́ владе́ние дзюдо́
        не так вредна́ в проце́ссе дра́ки
        как до́""", "-A-A"),

        ("""когда бухга́лтерша напьё́тся
        всегда́ танцу́ет нам стрипти́з
        и не в трусы́ к ней де́ньги ле́зут
        а и́з""", '-A-A'),

        ("""моя́ програ́мма вам помо́жет
        повы́сить у́ровень в дзюдо́
        вот фо́то после поеди́нка
        вот до́""", "-A-A"),

        ("""капу́сты посади́ла по́ле
        и а́исту свила́ гнездо́
        но интуи́ция мне ше́пчет
        не то́""", "----"),

        ("""гля́нул я на лю́сю
        и отве́л глаза́
        мы́слей нехоро́ших
        па́костных из за́""", "-A-A"),

        ("""хочу́ отшлё́пать анако́нду
        но непоня́тно по чему́
        вот у слона́ гора́здо ши́ре
        чем у́""", "-A-A"),

        ("""люблю́ натру́женные ру́ки
        и ко́сы ру́сые твои́
        а если вду́маться не бо́льно
        то и́""", "-A-A"),

        ("""когда уви́дела принце́сса
        моё́ фами́льное гнездо́
        то поняла́ несоверше́нство
        гнё́зд до́""", "-A-A"),

        ("""вот от люде́й мол на́ два ме́тра
        а мне поня́тно не вполне́
        счита́ть за челове́ка му́жа
        иль не́""", "-A-A"),

        ("""кро́тику из бу́син
        сде́лаю глаза́
        загляни́ дружо́чек
        горизо́нты за́""", "-A-A"),

        ("""никто́ не хо́чет с никола́ем
        повспомина́ть корпорати́в
        все предлага́ют почему́ то
        идти́ в""", "-A-A"),

        ("""оле́г реа́льностью уби́тый
        всё и́щет и́стину в вине́
        попереме́нно то нахо́дит
        то не́""", '-A-A'),

        ("""вот ра́ньше просто отрека́лись
        тепе́рь отла́йкиваются
        но в де́йстве э́том пра́вда жи́зни
        не вся́""", "----"),

        ("""ты сли́шком мно́го е́шь клетча́тки
        хле́б с отрубя́ми на обе́д
        не ви́жу ша́хматы и кста́ти
        где пле́д""", "-A-A"),

        ("""в трущо́бах во́лки приуны́ли
        воро́ны на ветвя́х скорбя́т
        верну́ли но́ль но́ль три́ проми́лли
        наза́д""", "ABAB"),

        ("""ты не пришё́л сказа́л что про́бки
        сказа́л ветра́ сказа́л снега́
        а я прекра́сно понима́ю
        пурга́""", "-A-A"),

        ("""на вконта́кт отвлё́кся
        и блину́ хана́
        сжё́г на сковоро́дке
        чу́чело блина́""", "-A-A"),

        #("""мы вме́сте с тобо́й занима́лись цигу́н
        # и вме́сте ходи́ли на йо́гу
        # но ты́ вдруг сказа́ла мне вместо сёгу́н
        # сё́гун""", "ABAB"),
]


    for true_markup, true_scheme in true_markups:
        poem = [z.strip() for z in true_markup.split('\n') if z.strip()]
        alignment = aligner.align(poem, check_rhymes=True)
        #print(alignment)
        #print('is_poor={}'.format(aligner.detect_poor_poetry(alignment)))
        #print('='*80)
        #print(alignment.get_unstressed_lines())
        #for pline in alignment.poetry_lines:
        #    print(pline.stress_signature_str)
        pred_markup = alignment.get_stressed_lines()
        expected_markup = '\n'.join(poem)
        if pred_markup != expected_markup:
            print('Markup test failed')
            print('Expected:\n{}\n\nOutput:\n{}'.format(expected_markup, pred_markup))
            exit(0)

        if alignment.rhyme_scheme != true_scheme:
            print('Rhyming scheme mismatch')
            print('Poem:\n{}\nExpected scheme={}\nPredicted scheme={}'.format(true_markup, true_scheme, alignment.rhyme_scheme))
            exit(0)


    print('All tests passed OK.')
