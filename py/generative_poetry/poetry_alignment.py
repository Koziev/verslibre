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
06.12.2022 полная переработка алгоритма расстановк ударений: оптимизация, подготовка к использованию спеллчекера
09.12.2022 Тесты ударятора вынесены в отдельный файл.
"""

import collections
import itertools
import traceback
from functools import reduce
import os
import io
import math
import jellyfish
import re
from typing import List, Set, Dict, Tuple, Optional

from poetry.phonetic import Accents, rhymed2, rhymed_fuzzy2, render_xword
from generative_poetry.metre_classifier import get_syllables
from generative_poetry.whitespace_normalization import normalize_whitespaces


# Коэффициенты для штрафов за разные отступления от идеальной метрики.
COEFF = dict()
COEFF['@68'] = 0.95  # 0.5
COEFF['@68_2'] = 0.98  # 0.95
COEFF['@71'] = 1.0
COEFF['@75'] = 0.98  # 0.9
COEFF['@77'] = 1.0
COEFF['@77_2'] = 1.0
COEFF['@79'] = 1.0
COEFF['@126'] = 0.98
COEFF['@225'] = 0.95
COEFF['@143'] = 0.9


def mul(items):
    return reduce((lambda x, y: x * y), items)


class Defect(object):
    def __init__(self, penalty, description):
        self.penalty = penalty
        self.description = description

    def __repr__(self):
        return self.description + '({})'.format(self.penalty)

    def serialize(self):
        return {'penalty': self.penalty, 'description': self.description}


class Defects(object):
    def __init__(self):
        self.items = []

    def __repr__(self):
        s = '({:5.2f})'.format(self.get_cumulative_factor())
        if self.items:
            s += ' {}'.format('; '.join(map(str, self.items)))

        return s

    def add_defect(self, defect):
        self.items.append(defect)

    def has_defects(self):
        return len(self.items) > 0

    def get_cumulative_factor(self):
        if self.items:
            return mul((1.0-defect.penalty) for defect in self.items)
        else:
            return 1.0

    def serialize(self):
        return {'cumulative_factor': self.get_cumulative_factor(), 'items': [d.serialize() for d in self.items]}


class MetreMappingResult(object):
    def __init__(self, prefix):
        self.score = 1.0
        self.word_mappings = []
        self.stress_shift_count = 0
        self.prefix = prefix
        self.cursor = 0

    @staticmethod
    def build_for_empty_line():
        r = MetreMappingResult(prefix=0)
        return r

    @staticmethod
    def build_from_source(src_mapping, new_cursor):
        new_mapping = MetreMappingResult(src_mapping.prefix)
        new_mapping.score = src_mapping.score
        new_mapping.word_mappings = list(src_mapping.word_mappings)
        new_mapping.stress_shift_count = src_mapping.stress_shift_count
        new_mapping.cursor = new_cursor
        return new_mapping

    def add_word_mapping(self, word_mapping):
        self.word_mappings.append(word_mapping)
        self.score *= word_mapping.get_total_score()
        if word_mapping.stress_shift:
            self.stress_shift_count += 1

    def finalize(self):
        # ищем цепочки безударных слогов (000...) длиннее 3х подряд, и штрафуем.
        signature = list(itertools.chain(*[m.word.stress_signature for m in self.word_mappings]))
        s = ''.join(map(str, signature))
        for m in re.findall(r'0{4,}', s):
            l = len(m)
            factor = 0.1  #math.exp((2-l)*0.5)
            self.score *= factor

        return

    def __repr__(self):
        if self.word_mappings:
            sx = []

            for word_mapping in self.word_mappings:
                sx.append(str(word_mapping))

            sx.append('〚' + '{:6.2g}'.format(self.score).strip() + '〛')
            return ' '.join(sx)
        else:
            return '««« EMPTY »»»'

    def get_score(self):
        stress_shift_factor = 1.0 if self.stress_shift_count < 2 else pow(0.5, self.stress_shift_count)
        return self.score * stress_shift_factor # * self.src_line_variant_score


class WordMappingResult(object):
    def __init__(self, word, TP, FP, TN, FN, stress_shift, additional_score_factor):
        self.word = word
        self.TP = TP
        self.FP = FP
        self.TN = TN
        self.FN = FN
        self.metre_score = pow(0.1, FP) * pow(0.95, FN) * additional_score_factor
        self.total_score = self.metre_score * word.get_score()
        self.stress_shift = stress_shift

    def get_total_score(self):
        return self.total_score

    def __repr__(self):
        s = self.word.get_stressed_form()
        if self.total_score != 1.0:
            s += '[' + '{:5.2g}'.format(self.total_score).strip() + ']'
        return s


class MetreMappingCursor(object):
    def __init__(self, metre_signature: List[int], prefix: int):
        self.prefix = prefix
        self.metre_signature = metre_signature
        self.length = len(metre_signature)

    def get_stress(self, cursor) -> int:
        """Возвращает ударность, ожидаемую в текущей позиции"""
        if self.prefix:
            if cursor == 0:
                return 0
            else:
                return self.metre_signature[(cursor - self.prefix) % self.length]
        else:
            return self.metre_signature[cursor % self.length]

    def map(self, stressed_words_chain, aligner):
        start_results = [MetreMappingResult(self.prefix)]
        final_results = []
        self.map_chain(prev_node=stressed_words_chain, prev_results=start_results, aligner=aligner, final_results=final_results)
        final_results = sorted(final_results, key=lambda z: -z.get_score())
        return final_results

    def map_chain(self, prev_node, prev_results, aligner, final_results):
        for cur_slot in prev_node.next_nodes:
            cur_results = self.map_word(stressed_word_group=cur_slot.stressed_words, results=prev_results, aligner=aligner)
            if cur_slot.next_nodes:
                self.map_chain(prev_node=cur_slot, prev_results=cur_results, aligner=aligner, final_results=final_results)
            else:
                for result in cur_results:
                    result.finalize()
                    final_results.append(result)

    def map_word(self, stressed_word_group, results: [MetreMappingResult], aligner):
        new_results = []

        for prev_result in results:
            for word_mapping, new_cursor in self.map_word1(stressed_word_group, prev_result, aligner):
                next_metre_mapping = MetreMappingResult.build_from_source(prev_result, new_cursor)
                next_metre_mapping.add_word_mapping(word_mapping)
                new_results.append(next_metre_mapping)

        new_results = sorted(new_results, key=lambda z: -z.get_score())

        return new_results

    def map_word1(self, stressed_word_group, result: MetreMappingResult, aligner):
        result_mappings = []

        for stressed_word in stressed_word_group:
            cursor = result.cursor
            TP, FP, TN, FN = 0, 0, 0, 0
            for word_sign in stressed_word.stress_signature:
                metre_sign = self.get_stress(cursor)
                if metre_sign:
                    if word_sign:
                        # Ударение должно быть и оно есть
                        TP += 1
                    else:
                        # ударение должно быть, но его нет
                        FN += 1
                else:
                    if word_sign:
                        # Ударения не должно быть, но оно есть
                        FP += 1
                    else:
                        # Ударения не должно быть, и его нет
                        TN += 1
                cursor += 1

            # Проверим сочетание ударения в предыдущем слове и в текущем, в частности - оштрафуем за два ударных слога подряд
            additional_score_factor = 1.0
            if len(stressed_word.stress_signature) > 0:
                if len(result.word_mappings) > 0:
                    prev_mapping = result.word_mappings[-1]
                    if prev_mapping.word.stress_signature:
                        if prev_mapping.word.stress_signature[-1] == 1:  # предыдущее слово закончилось ударным слогом
                            if stressed_word.stress_signature[0] == 1:
                                # большой штраф за два ударных подряд
                                additional_score_factor = 0.1

            mapping1 = WordMappingResult(stressed_word, TP, FP, TN, FN, False, additional_score_factor=additional_score_factor)
            result_mappings.append((mapping1, cursor))

        return result_mappings


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

    def get_score(self):
        return self.score

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
        self.tags2 = dict(s.split('=') for s in tags)
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
            elif clower in 'бвгджзклмнпрстфхцчшщ':
                if n_vowels == 0:
                    self.leading_consonants += 1
                else:
                    self.trailing_consonants += 1
        self.n_vowels = n_vowels

    def get_attr(self, tag_name):
        return self.tags2.get(tag_name, '')

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

    def get_stress_variants(self, aligner, allow_stress_shift):
        variants = []

        nvowels = sum((c in 'уеыаоэёяию') for c in self.form.lower())
        uform = self.form.lower()

        if False:  #aligner.accentuator.is_oov(uform) and nvowels > 1:
            # 17.08.2022 для OOV слов просто перебираем все варианты ударения.
            # в будущем тут можно вызвать модельку в ударяторе, которая выдаст вероятности ударения на каждой из гласных.
            vowel_count = 0
            for i, c in enumerate(uform):
                if c in 'уеыаоэёяию':
                    vowel_count  += 1
                    variants.append(WordStressVariant(self, vowel_count, 1.0))
        elif uform == 'начала' and self.upos == 'NOUN':
            # У этого слова аж 3 варианта ударения, два для глагола и 1 для существительного.
            # Если тэггер считает, что у нас именно существительное - не используем вариативность глагола.
            # TODO вынести такую логику как-то в словарь ambiguous_accents_2.yaml, чтобы не приходилось тут
            # хардкодить.
            variants.append(WordStressVariant(self, 2, 1.0))
        elif uform in aligner.accentuator.ambiguous_accents2:
            # Вариативное ударение в рамках одной грамматической формы пОнял-понЯл
            for stress_pos in aligner.accentuator.ambiguous_accents2[uform]:
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
            if self.is_rhyming_word:
                # уколи сестра мне
                # в глаз чего нибудь     <======
                # за бревном не вижу
                # к коммунизму путь
                variants.append(WordStressVariant(self, self.stress_pos, 1.0))
            else:
                variants.append(WordStressVariant(self, self.stress_pos, 0.5))
        elif nvowels == 1 and self.upos in ('NOUN', 'NUM', 'ADJ'):
            # Односложные слова типа "год" или "два" могут стать безударными:
            # В год Петуха́ учи́тесь кукаре́кать.
            #   ^^^
            # Оле́г два дня́ прожи́л без во́дки.
            #      ^^^
            variants.append(WordStressVariant(self, self.stress_pos, 1.0))
            variants.append(WordStressVariant(self, -1, 0.7))
        elif nvowels == 1 and self.upos == 'VERB':
            # 21.08.2022 разрешаем становится безударными односложным глаголам.
            variants.append(WordStressVariant(self, self.stress_pos, 1.0))
            variants.append(WordStressVariant(self, -1, 0.7))
        elif uform == 'нет':  # and not self.is_rhyming_word:
            # частицу (или глагол для Stanza) "нет" с ударением штрафуем
            variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68_2']))

            # а вариант без ударения - с нормальным скором:
            variants.append(WordStressVariant(self, -1, COEFF['@71']))
        elif self.upos in ('ADP', 'CCONJ', 'SCONJ', 'PART', 'INTJ', 'AUX'): # and not self.is_rhyming_word:
            if uform in ('о', 'у', 'из', 'от', 'под', 'подо', 'за', 'при', 'до', 'про', 'для', 'ко', 'со', 'во', 'на', 'по', 'об', 'обо', 'без', 'над', 'пред') and self.upos == 'ADP':
                # эти предлоги никогда не делаем ударными
                variants.append(WordStressVariant(self, -1, 1.0))

                # Но если это последнее слово в строке, то допускается вариант:
                # необосно́ванная ве́ра
                # в своё́ владе́ние дзюдо́
                # не так вредна́ в проце́ссе дра́ки
                # как до
                if self.is_rhyming_word:
                    variants.append(WordStressVariant(self, self.stress_pos, 1.0))
                else:
                    variants.append(WordStressVariant(self, self.stress_pos, 0.1))
            elif uform in ('не', 'бы', 'ли', 'же', 'ни', 'ка'):
                # Частицы "не" и др. никогда не делаем ударной
                variants.append(WordStressVariant(self, -1, 1.0))

                if self.is_rhyming_word:
                    variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68']))
            elif uform in ('а', 'и', 'или', 'но'):
                # союзы "а", "и", "но" обычно безударный:
                # А была бы ты здорова
                # ^
                variants.append(WordStressVariant(self, -1, 1.0))

                if self.is_rhyming_word:
                    # ударный вариант в рифмуемой позиции
                    variants.append(WordStressVariant(self, self.stress_pos, 0.70))  # ударный вариант
                    #    variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68']))
                else:
                    variants.append(WordStressVariant(self, self.stress_pos, 0.20))  # ударный вариант
            elif uform == 'и' and self.upos == 'PART':
                # Частицу "и" не делаем ударной:
                # Вот и она... Оставив магазин
                #     ^
                variants.append(WordStressVariant(self, -1, 1.0))

                if self.is_rhyming_word:
                    variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68']))
            elif uform in ('были', 'было'):
                variants.append(WordStressVariant(self, -1, 1.0))
                variants.append(WordStressVariant(self, self.stress_pos, 1.0))
            elif nvowels == 0:
                # Предлоги без единой гласной типа "в"
                variants.append(WordStressVariant(self, -1, 1.0))
            else:
                if count_vowels(uform) < 3:
                    # Предлоги, союзы, частицы предпочитаем без ударения, если в них меньше трех гласных.
                    # поэтому базовый вариант добавляем с дисконтом:
                    if uform in ['лишь', 'вроде', 'если', 'чтобы', 'когда', 'просто', 'мимо', 'даже', 'всё', 'хотя', 'едва', 'нет', 'пока']:
                        variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68_2']))
                    else:
                        variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68']))

                    # а вариант без ударения - с нормальным скором:
                    #if not self.is_rhyming_word:
                    variants.append(WordStressVariant(self, -1, COEFF['@71']))
                else:
                    # Частицы типа "НЕУЖЕЛИ" с 3 и более гласными
                    variants.append(WordStressVariant(self, self.stress_pos, 1.0))
        elif self.upos in ('PRON', 'ADV', 'DET'):
            # Для односложных местоимений (Я), наречий (ТУТ, ГДЕ) и слов типа МОЙ, ВСЯ, если они не последние в строке,
            # добавляем вариант без ударения с дисконтом.
            if nvowels == 1:
                variants.append(WordStressVariant(self, self.stress_pos, 1.0))  # COEFF['@75']
                # вариант без ударения
                variants.append(WordStressVariant(self, -1, 1.0))  # COEFF['@77']
            else:
                if uform in ['эти', 'эту', 'это', 'мои', 'твои', 'моих', 'твоих', 'моим', 'твоим', 'моей', 'твоей',
                             'мою', 'твою', 'его', 'ему', 'нему', 'ее', 'её', 'себе', 'меня', 'тебя', 'свою', 'свои', 'своим', 'они', 'она',
                             'уже', 'этом', 'тебе']:
                    # Безударный вариант для таких двусложных прилагательных
                    variants.append(WordStressVariant(self, -1, COEFF['@77_2']))

                variants.append(WordStressVariant(self, self.stress_pos, COEFF['@79']))
        else:
            if uform in ['есть', 'раз', 'быть', 'будь', 'был']:  # and not self.is_rhyming_word:
                # безударный вариант
                variants.append(WordStressVariant(self, -1, COEFF['@143']))

            # Добавляем исходный вариант с ударением
            variants.append(WordStressVariant(self, self.stress_pos, 1.0))

        if allow_stress_shift:
            # Сдвигаем ударение вопреки решению на основе частеречной разметки
            uform = self.form.lower()

            if count_vowels(uform) > 1:
                has_different_stresses = uform in aligner.accentuator.ambiguous_accents and uform not in aligner.accentuator.ambiguous_accents2
                if has_different_stresses:
                    # Можем попробовать взять другой вариант ударности слова, считая,
                    # что имеем дело с ошибкой частеречной разметки.
                    sx = list(aligner.accentuator.ambiguous_accents[uform].keys())

                    for stress_form in sx:
                        stress_pos = -1
                        n_vowels = 0
                        for c in stress_form:
                            if c.lower() in 'уеыаоэёяию':
                                n_vowels += 1

                            if c in 'АЕЁИОУЫЭЮЯ':
                                stress_pos = n_vowels
                                break

                        if not any((variant.new_stress_pos == stress_pos) for variant in variants):
                            # Нашли новый вариант ударности этого слова.
                            # Попробуем использовать его вместо выбранного с помощью частеречной разметки.
                            new_stressed_word = WordStressVariant(self, stress_pos, 0.99)
                            variants.append(new_stressed_word)

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


class StressVariantsSlot(object):
    def __init__(self):
        self.stressed_words = None
        self.next_nodes = None

    def __repr__(self):
        s = ''

        if self.stressed_words:
            s += '[ ' + ' | '.join(map(str, self.stressed_words)) + ' ]'
        else:
            s += '∅'

        if self.next_nodes:
            if len(self.next_nodes) == 1:
                s += ' ↦ '
                s += str(self.next_nodes[0])
            else:
                s += ' ⇉ ⦃'
                for i, n in enumerate(self.next_nodes, start=1):
                    s += ' 〚{}〛 {}'.format(i, str(n))
                s += '⦄'

        return s

    @staticmethod
    def build_next(poetry_words, aligner, allow_stress_shift):
        next_nodes = []

        pword = poetry_words[0]

        # Проверяем особые случаи трехсловных словосочетаний, в которых ударение ВСЕГДА падает особым образом:
        # бок О бок
        if len(poetry_words) > 2:
            lword1 = pword.form.lower()
            lword2 = poetry_words[1].form.lower()
            lword3 = poetry_words[2].form.lower()
            key = (lword1, lword2, lword3)
            collocs = aligner.collocations.get(key, None)
            if collocs is not None:
                for colloc in collocs:
                    if colloc.stressed_word_index == 0:
                        # первое слово становится ударным, второе и третье - безударные
                        stressed_word1 = WordStressVariant(poetry_words[0], new_stress_pos=colloc.stress_pos, score=1.0)
                        stressed_word2 = WordStressVariant(poetry_words[1], new_stress_pos=-1, score=1.0)
                        stressed_word3 = WordStressVariant(poetry_words[2], new_stress_pos=-1, score=1.0)
                    elif colloc.stressed_word_index == 1:
                        # первое слово становится безударным, второе - ударное, третье - безударное
                        stressed_word1 = WordStressVariant(poetry_words[0], new_stress_pos=-1, score=1.0)
                        stressed_word2 = WordStressVariant(poetry_words[1], new_stress_pos=colloc.stress_pos, score=1.0)
                        stressed_word3 = WordStressVariant(poetry_words[2], new_stress_pos=-1, score=1.0)
                    else:
                        # первое и второе слово безударные, третье - ударное
                        stressed_word1 = WordStressVariant(poetry_words[0], new_stress_pos=-1, score=1.0)
                        stressed_word2 = WordStressVariant(poetry_words[1], new_stress_pos=-1, score=1.0)
                        stressed_word3 = WordStressVariant(poetry_words[2], new_stress_pos=colloc.stress_pos, score=1.0)

                    next_node = StressVariantsSlot()
                    next_node.stressed_words = [stressed_word1]

                    next_node2 = StressVariantsSlot()
                    next_node2.stressed_words = [stressed_word2]
                    next_node.next_nodes = [next_node2]

                    next_node3 = StressVariantsSlot()
                    next_node3.stressed_words = [stressed_word3]
                    next_node2.next_nodes = [next_node3]

                    if len(poetry_words) > 3:
                        next_node3.next_nodes = StressVariantsSlot.build_next(poetry_words[3:], aligner, allow_stress_shift=allow_stress_shift)

                    next_nodes.append(next_node)

                return [next_node]

        # Проверяем особый случай двусловных словосочетаний, в которых ударение ВСЕГДА падает не так, как обычно:
        # друг др^уга
        if len(poetry_words) > 1:
            lword1 = pword.form.lower()
            lword2 = poetry_words[1].form.lower()
            key = (lword1, lword2)
            collocs = aligner.collocations.get(key, None)
            if collocs is not None:
                for colloc in collocs:
                    if colloc.stressed_word_index == 0:
                        # первое слово становится ударным, второе - безударное
                        stressed_word1 = WordStressVariant(poetry_words[0], new_stress_pos=colloc.stress_pos, score=1.0)
                        stressed_word2 = WordStressVariant(poetry_words[1], new_stress_pos=-1, score=1.0)
                    else:
                        # первое слово становится безударным, второе - ударное
                        stressed_word1 = WordStressVariant(poetry_words[0], new_stress_pos=-1, score=1.0)
                        stressed_word2 = WordStressVariant(poetry_words[1], new_stress_pos=colloc.stress_pos, score=1.0)

                    next_node = StressVariantsSlot()
                    next_node.stressed_words = [stressed_word1]

                    next_node2 = StressVariantsSlot()
                    next_node2.stressed_words = [stressed_word2]
                    next_node.next_nodes = [next_node2]

                    if len(poetry_words) > 2:
                        next_node2.next_nodes = StressVariantsSlot.build_next(poetry_words[2:], aligner, allow_stress_shift=allow_stress_shift)

                    next_nodes.append(next_node)

                return next_nodes

        # Самый типичный путь - получаем варианты ударения для слова с их весами.
        next_node = StressVariantsSlot()
        next_node.stressed_words = pword.get_stress_variants(aligner, allow_stress_shift=allow_stress_shift)
        if len(poetry_words) > 1:
            next_node.next_nodes = StressVariantsSlot.build_next(poetry_words[1:], aligner, allow_stress_shift)
        next_nodes.append(next_node)
        return next_nodes

    @staticmethod
    def build(poetry_words, aligner, allow_stress_shift):
        start = StressVariantsSlot()
        start.next_nodes = StressVariantsSlot.build_next(poetry_words, aligner, allow_stress_shift)
        return start


class LineStressVariant(object):
    def __init__(self, poetry_line, stressed_words, aligner):
        self.poetry_line = poetry_line
        self.stressed_words = stressed_words
        self.stress_signature = list(itertools.chain(*(w.stress_signature for w in stressed_words)))
        self.stress_signature_str = ''.join(map(str, self.stress_signature))
        self.init_rhyming_tail()
        if aligner is not None:
            self.score_sequence(aligner)

    @staticmethod
    def build_empty_line():
        return LineStressVariant('', [], None)

    def score_sequence(self, aligner):
        self.total_score = reduce(lambda x, y: x*y, [w.score for w in self.stressed_words])
        self.penalties = []

        # 09-08-2022 штрафуем за безударное местоимение после безударного предлога с гласной:
        # я вы́рос в ми́ре где драко́ны
        # съеда́ли на́ших дочере́й
        # а мы расти́ли помидо́ры
        # и ке́тчуп де́лали для них
        #                 ^^^^^^^
        if len(self.stressed_words) >= 2 and \
                self.stressed_words[-1].new_stress_pos == -1 and \
                self.stressed_words[-2].new_stress_pos == -1 and \
                self.stressed_words[-2].poetry_word.upos == 'ADP' and \
                count_vowels(self.stressed_words[-2].poetry_word.form) and \
                self.stressed_words[-1].poetry_word.upos == 'PRON':
            self.total_score *= 0.5
            self.penalties.append('@545')

        # 04-08-2022 если клаузулла безударная, то имеем кривое оформление ритмического рисунка.
        if not self.rhyming_tail.is_ok():
            self.total_score *= 0.1
            self.penalties.append('@550')

        # 06-08-2022 если в строке вообще ни одного ударения - это неприемлемо
        if sum((w.new_stress_pos != -1) for w in self.stressed_words) == 0:
            self.total_score *= 0.01
            self.penalties.append('@555')

        # добавка от 15-12-2021: два подряд ударных слога наказываем сильно!
        #if '11' in self.stress_signature_str:
        #    self.total_score *= 0.1
        #    self.penalties.append('@560')

        # 01-01-2022 ударную частицу "и" в начале строки наказываем сильно
        # 〚И́(0.500) споко́йно детворе́〛(0.500)
        if self.stressed_words[0].new_stress_pos == 1 and self.stressed_words[0].poetry_word.form.lower() == 'и':
            self.total_score *= 0.1
            self.penalties.append('@573')

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
                #if count_vowels(word1.poetry_word.form) > 0 and count_vowels(word2.poetry_word.form) > 0 and count_vowels(word3.poetry_word.form) > 0:
                #    self.total_score *= 0.1
                #    self.penalties.append('@323')
                pass

        # 28-12-2021 штрафуем за подряд идущие короткие слова (1-2 буквы)
        #for word1, word2, word3 in zip(stressed_words, stressed_words[1:], stressed_words[2:]):
        #    if word1.is_short_word() and word2.is_short_word() and word3.is_short_word():
        #        self.total_score *= 0.2

        if sum(self.stress_signature) == 1:
            # Всего один ударный слог в строке с > 2 слогов... Очень странно.
            # 〚Что за недоразуме́нье〛
            # 00000010
            # 07.06.2022 Но если в строке всего одно слово или группа предлог+сущ - это нормально!
            if len(self.poetry_line.pwords) > 2 or (len(self.poetry_line.pwords) == 2 and self.poetry_line.pwords[-2].upos != 'ADP'):
                if sum(count_vowels(w.poetry_word.form) for w in self.stressed_words) > 2:
                    self.total_score *= 0.1
                    self.penalties.append('@335')
        else:
            # 01-01-2022 Детектируем разные сбои ритма
            #if self.stress_signature_str in aligner.bad_signature1:
            #    self.total_score *= 0.1
            #    self.penalties.append('@340')
            pass

        # Три безударных слога в конце - это очень странно:
        # Ходи́ть по то́нкому льду.		0101000
        if self.stress_signature_str.endswith('000'):
            self.total_score *= 0.1
            self.penalties.append('@626')

    def init_rhyming_tail(self):
        stressed_word = None
        unstressed_prefix = None
        unstressed_postfix_words = []

        # Ищем справа слово с ударением
        i = len(self.stressed_words)-1
        while i >= 0:
            pword = self.stressed_words[i]
            if pword.new_stress_pos != -1:  # or pword.poetry_word.n_vowels > 1:
                stressed_word = pword

                if re.match(r'^[аеёиоуыэюя]$', pword.poetry_word.form, flags=re.I) is not None:
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
        if self.poetry_line == '':
            return ''

        s = '〚' + ' '.join(w.__repr__() for w in self.stressed_words) + '〛'
        if self.total_score != 1.0:
            s += '(≈{:5.3f})'.format(self.total_score)
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

    def __len__(self):
        return len(self.pwords)

    @staticmethod
    def build(text, udpipe_parser, accentuator):
        poetry_line = PoetryLine()
        poetry_line.text = text
        poetry_line.pwords = []

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
                        alt_stress_pos.extend(ax)
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
                    poetry_line.pwords.append(pword)
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
                            px = accentuator.ambiguous_accents2[word]
                            stress_pos = px[0]
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

                    form2 = accentuator.yoficate(ud_token.form)
                    pword = PoetryWord(ud_token.lemma, form2, ud_token.upos, ud_token.tags, stress_pos, alt_stress_pos)

                    # НАЧАЛО ОТЛАДКИ
                    if not isinstance(poetry_line, PoetryLine):
                        print('DEBUG@1027')
                    # КОНЕЦ ОТЛАДКИ

                    poetry_line.pwords.append(pword)

        poetry_line.locate_rhyming_word()
        return poetry_line

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

    # def get_stress_variants(self, aligner):
    #     wordx = [pword.get_stress_variants(aligner) for pword in self.pwords]
    #     variants = [LineStressVariant(self, swords, aligner) for swords in itertools.product(*wordx)]
    #
    #     # 23-01-2022 добавляем варианты, возникающие из-за особых ударений в словосочетаниях типа "пО полю"
    #     lwords = [w.form.lower() for w in self.pwords]
    #     if any((w in aligner.collocation2_first) for w in lwords) and any((w in aligner.collocation2_second) for w in lwords):
    #         # В строке возможно присутствует одно из особых словосочетаний длиной 2
    #         for colloc in aligner.collocations:
    #             add_variants = []
    #             if len(colloc) == 2:
    #                 for i1, (w1, w2) in enumerate(zip(lwords, lwords[1:])):
    #                     if colloc.hit2(w1, w2):
    #                         # из всех вариантов в variants делаем еще по 1 варианту
    #                         for variant in variants:
    #                             v = colloc.produce_stressed_line(variant, aligner)
    #                             add_variants.append(v)
    #
    #             if add_variants:
    #                 variants.extend(add_variants)
    #
    #     # 04-08-2022 добавляем варианты для триграмм типа "бок О бок"
    #     if any((w in aligner.collocation3_first) for w in lwords) and any((w in aligner.collocation3_second) for w in lwords) and any((w in aligner.collocation3_third) for w in lwords):
    #         # В строке возможно присутствует одно из особых словосочетаний длиной 3
    #         add_variants = []
    #         for colloc in aligner.collocations:
    #             if len(colloc) == 3:
    #                 for i1, (w1, w2, w3) in enumerate(zip(lwords, lwords[1:], lwords[2:])):
    #                     if colloc.hit3(w1, w2, w3):
    #                         # из всех вариантов в variants делаем еще по 1 варианту
    #                         for variant in variants:
    #                             v = colloc.produce_stressed_line(variant, aligner)
    #                             add_variants.append(v)
    #
    #         if add_variants:
    #             variants.extend(add_variants)
    #
    #     return variants

    def get_first_stress_variants(self, aligner):
        swords = [pword.get_first_stress_variant() for pword in self.pwords]
        return LineStressVariant(self, swords, aligner)


class PoetryAlignment(object):
    def __init__(self, poetry_lines, score, meter, rhyme_scheme, metre_mappings):
        self.poetry_lines = poetry_lines
        self.score = score
        self.meter = meter
        self.rhyme_scheme = rhyme_scheme
        self.error_text = None
        self.metre_mappings = metre_mappings

    def __repr__(self):
        s = '{} {}({:5.3f}):\n'.format(self.meter, self.rhyme_scheme, self.score)
        s += '\n'.join(map(str, self.poetry_lines))
        return s

    @staticmethod
    def build_n4(alignments4, total_score):
        poetry_lines = []
        for a in alignments4:
            poetry_lines.extend(a.poetry_lines)
        return PoetryAlignment(poetry_lines, total_score, alignments4[0].meter, alignments4[0].rhyme_scheme,
                               metre_mappings=[a.metre_mappings for a in alignments4])

    @staticmethod
    def build_no_rhyming_result(poetry_lines):
        a = PoetryAlignment(poetry_lines, 0.0, None, None, metre_mappings=None)
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

    def key(self):
        return tuple(self.words)

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


class RapAlignment(object):
    def __init__(self):
        self.rhymes = []
        self.defects = []
        self.denominator = 0

    def get_total_score(self):
        return len(self.rhymes) / (self.denominator+1.0e-6) - sum(weight for descr, weight in self.defects)

    def __repr__(self):
        return f'hits={self.rhymes} denominator={self.denominator}'


def ngrams(s, n):
    return set(u''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1 & shingles2))/float(len(shingles1 | shingles2) + 1e-6)



class PoetryStressAligner(object):
    def __init__(self, udpipe, accentuator, data_dir):
        self.udpipe = udpipe
        self.accentuator = accentuator

        self.collocations = collections.defaultdict(list)

        with io.open(os.path.join(data_dir, 'collocation_accents.dat'), 'r', encoding='utf-8') as rdr:
            for line in rdr:
                line = line.strip()
                if line.startswith('#') or len(line) == 0:
                    continue

                c = CollocationStress.load_collocation(line)

                self.collocations[c.key()].append(c)

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
        max_score = 0.0

        for (prefix1, prefix2) in [('', ''), ('0', ''), ('', '0')]:
            sign1 = prefix1 + sline1.stress_signature_str
            sign2 = prefix2 + sline2.stress_signature_str

            len1 = len(sign1)
            len2 = len(sign2)

            if len1 <= len2:
                sign2 = sign2[:len1]
            else:
                sign1 = sign1[:len2]

            if sign1.count('1') < 2 or sign2.count('1') < 2:
                continue

            diff = jellyfish.hamming_distance(sign1, sign2)
            score = math.exp(-diff * 1.0)
            max_score = max(score, max_score)

        return max_score

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
        # <<<<<<<<<>>>>>>>>
        #d = jellyfish.damerau_levenshtein_distance(sline1.stress_signature_str, sline2.stress_signature_str)
        #score = math.exp(-d*1.0)

        # Добавка от 31-12-2021: если во второй строке больше ударных слогов, чем в первой - штрафуем сильно.
        #nstressed_1 = sum(sline1.stress_signature)
        #nstressed_2 = sum(sline2.stress_signature)
        #if nstressed_2 > nstressed_1:
        #    score *= pow(0.5, (nstressed_2-nstressed_1))

        #spectrum1 = self.get_spectrum(sline1)
        #spectrum2 = self.get_spectrum(sline2)

        # 01-01-2022 Разные корявые рифмовки детектируем по словарю сигнатур.
        #if (sline1.stress_signature_str, sline2.stress_signature_str) in self.bad_alignments2:
        #    score *= 0.1

        return score

    def align(self, lines0, check_rhymes=True):
        # Иногда для наглядности можем выводить сгенерированные стихи вместе со значками ударения.
        # Эти значки мешают работе алгоритма транскриптора, поэтому уберем их сейчас.
        lines = [line.replace('\u0301', '') for line in lines0]
        nlines = len(lines)
        if nlines == 1:
            return self.align1(lines)
        elif nlines == 2:
            return self.align2(lines, check_rhymes)
        elif nlines == 4:
            return self.align4(lines, check_rhymes)
        else:
            if nlines in (9, 14, 19):
                n4 = nlines // 4
                is_x4 = True
                j = 0
                for i in range(1, n4):
                    j += 4
                    if lines[j] != '':
                        is_x4 = False
                        break
                    j += 1

                if is_x4:
                    # Считаем, что перед нами несколько блоков по 4 строки, разделенных пустой строкой
                    return self.align_n4(lines, check_rhymes)

            return self.align_nonstandard_blocks(lines)
            #raise ValueError("Alignment is not implemented for {}-liners! text={}".format(len(lines), '\n'.join(lines)))

    def align_n4(self, lines, check_rhymes):
        total_score = 1.0
        blocks = []
        block = []
        block_alignments = []
        for line in lines + ['']:
            if line:
                block.append(line)
            else:
                blocks.append(block)
                block = []

        for block in blocks:
            assert(len(block) == 4)

            alignment = self.align4(block, check_rhymes=check_rhymes)
            if alignment is None:
                total_score = 0.0
                break

            block_score = alignment.score
            if self.detect_poor_poetry(alignment):
                block_score *= 0.1

            # TODO: штрафовать за разные виды метра в блоках?

            total_score *= block_score
            block_alignments.append(alignment)

        return PoetryAlignment.build_n4(block_alignments, total_score)

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
            # 09-08-2022 отключил эту проверку, так как в is_poor_rhyming делает более качественная проверка!
            #if poetry_word1.form.lower() == poetry_word2.form.lower():
            #    return False

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
        #line1 = lines[0]
        #for line2 in lines[1:]:
        #    if (line1.stress_signature_str, line2.stress_signature_str) in self.bad_alignments2:
        #        # Отсеиваем заведомо несопоставимые пары
        #        return 0.0
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
        res = self.align4(lines, check_rhymes=True)
        if res.rhyme_scheme == 'AABA':
            return res
        else:
            #plines = [PoetryLine.build(line, self.udpipe, self.accentuator) for line in lines]
            #return PoetryAlignment.build_no_rhyming_result([pline.get_stress_variants(self)[0] for pline in plines])
            return None

    def align1(self, lines):
        """Разметка однострочника."""
        pline1 = PoetryLine.build(lines[0], self.udpipe, self.accentuator)

        # 08.11.2022 добавлена защита от взрыва числа переборов для очень плохих генераций.
        if sum((pword.n_vowels > 1) for pword in pline1.pwords) >= 8:
            raise ValueError('Line is too long: "{}"'.format(pline1))

        best_score = 0.0
        best_metre_name = None
        best_mapping = None

        for allow_stress_shift in [False, True]:
            # Заранее сгенерируем для каждого слова варианты спеллчека и ударения...
            # stressed_words_groups = [pword.get_stress_variants(self, allow_stress_shift=True) for pword in pline1.pwords]
            stressed_words_chain = StressVariantsSlot.build(poetry_words=pline1.pwords, aligner=self, allow_stress_shift=allow_stress_shift)

            for metre_name, metre_signature in meters:
                cursor = MetreMappingCursor(metre_signature, prefix=0)
                for metre_mapping in cursor.map(stressed_words_chain, self):
                    if metre_mapping.get_score() > best_score:
                        best_score = metre_mapping.get_score()
                        best_metre_name = metre_name
                        best_mapping = metre_mapping
            if best_score > 0.4:
                break

        # Возвращаем найденный лучший вариант разметки и его оценку

        stressed_words = [m.word for m in best_mapping.word_mappings]
        new_stress_line = LineStressVariant(pline1, stressed_words, self)
        best_variant = [new_stress_line]

        return PoetryAlignment(best_variant, best_score, best_metre_name, rhyme_scheme='', metre_mappings=[best_mapping])

    def align2(self, lines, check_rhymes):
        """ Разметка двустрочника """
        plines = [PoetryLine.build(line, self.udpipe, self.accentuator) for line in lines]

        # 08.11.2022 добавлена защита от взрыва числа переборов для очень плохих генераций.
        for pline in plines:
            if sum((pword.n_vowels>1) for pword in pline.pwords) >= 8:
                raise ValueError('Line is too long: "{}"'.format(pline))

        best_score = 0.0
        best_metre = None
        best_rhyme_scheme = None
        best_variant = None

        for allow_stress_shift in [False, True]:
            #if best_score > 0.3:
            #    break

            # Заранее сгенерируем для каждого слова варианты спеллчека и ударения...
            stressed_words_groups = [StressVariantsSlot.build(poetry_words=pline.pwords, aligner=self, allow_stress_shift=allow_stress_shift) for pline in plines]

            # Для каждой строки перебираем варианты разметки и оставляем по 2 варианта в каждом метре.
            for metre_name, metre_signature in meters:
                best_scores = dict()

                # В каждой строке перебираем варианты расстановки ударений.
                for ipline, pline in enumerate(plines):
                    best_scores[ipline] = dict()

                    for prefix in [0, 1]:
                        cursor = MetreMappingCursor(metre_signature, prefix=prefix)
                        metre_mappings = cursor.map(stressed_words_groups[ipline], self)

                        for metre_mapping in metre_mappings[:2]:  # берем только 2(?) лучших варианта
                            stressed_words = [m.word for m in metre_mapping.word_mappings]
                            new_stress_line = LineStressVariant(pline, stressed_words, self)

                            if new_stress_line.get_rhyming_tail().is_ok():
                                tail_str = new_stress_line.get_rhyming_tail().__repr__()
                                score = metre_mapping.get_score()
                                if tail_str not in best_scores[ipline]:
                                    prev_score = -1e3
                                else:
                                    prev_score = best_scores[ipline][tail_str][0].get_score()
                                if score > prev_score:
                                    best_scores[ipline][tail_str] = (metre_mapping, new_stress_line)

                # Теперь для каждой исходной строки имеем несколько вариантов расстановки ударений.
                # Перебираем сочетания этих вариантов, проверяем рифмовку и оставляем лучший вариант для данного метра.
                stressed_lines2 = [list() for _ in range(2)]
                for iline, items2 in best_scores.items():
                    stressed_lines2[iline].extend(items2.values())

                vvx = list(itertools.product(*stressed_lines2))
                for ivar, plinev in enumerate(vvx):
                    # plinev это набор из двух экземпляров кортежей (MetreMappingResult, LineStressVariant).

                    # Определяем рифмуемость
                    rhyme_scheme = None
                    rhyme_score = 1.0

                    last_pwords = [pline[1].get_rhyming_tail() for pline in plinev]
                    if self.check_rhyming(last_pwords[0], last_pwords[1]):
                        rhyme_scheme = 'AA'
                    else:
                        rhyme_scheme = '--'
                        rhyme_score = 0.75

                    total_score = rhyme_score * mul([pline[0].get_score() for pline in plinev])
                    if total_score > best_score:
                        best_score = total_score
                        best_metre = metre_name
                        best_rhyme_scheme = rhyme_scheme
                        best_variant = plinev

        if best_variant is None:
            # В этом случае вернем результат с нулевым скором и особым текстом, чтобы
            # можно было вывести в лог строки с каким-то дефолтными
            return PoetryAlignment.build_no_rhyming_result([pline.get_stress_variants(self)[0] for pline in plines])
        else:
            # Возвращаем найденный вариант разметки и его оценку
            best_lines = [v[1] for v in best_variant]
            metre_mappings = [v[0] for v in best_variant]
            return PoetryAlignment(best_lines, best_score, best_metre, rhyme_scheme=best_rhyme_scheme, metre_mappings=metre_mappings)

    def align4(self, lines, check_rhymes):
        plines = [PoetryLine.build(line, self.udpipe, self.accentuator) for line in lines]

        # 08.11.2022 добавлена защита от взрыва числа переборов для очень плохих генераций.
        for pline in plines:
            if sum((pword.n_vowels>1) for pword in pline.pwords) >= 8:
                raise ValueError('Line is too long: "{}"'.format(pline))

        best_score = 0.0
        best_metre = None
        best_rhyme_scheme = None
        best_variant = None

        for allow_stress_shift in [False, True]:
            #if best_score > 0.3:
            #    break

            # Заранее сгенерируем для каждого слова варианты спеллчека и ударения...
            stressed_words_groups = [StressVariantsSlot.build(poetry_words=pline.pwords, aligner=self, allow_stress_shift=allow_stress_shift) for pline in plines]

            # Для каждой строки перебираем варианты разметки и оставляем по ~2 варианта в каждом метре.
            for metre_name, metre_signature in meters:
                best_scores = dict()

                # В каждой строке перебираем варианты расстановки ударений.
                for ipline, pline in enumerate(plines):
                    best_scores[ipline] = dict()

                    for prefix in [0, 1]:
                        cursor = MetreMappingCursor(metre_signature, prefix=prefix)
                        for metre_mapping in cursor.map(stressed_words_groups[ipline], self):
                            stressed_words = [m.word for m in metre_mapping.word_mappings]
                            new_stress_line = LineStressVariant(pline, stressed_words, self)

                            if new_stress_line.get_rhyming_tail().is_ok():
                                tail_str = new_stress_line.get_rhyming_tail().__repr__()
                                score = metre_mapping.get_score()
                                if tail_str not in best_scores[ipline]:
                                    prev_score = -1e3
                                else:
                                    prev_score = best_scores[ipline][tail_str][0].get_score()
                                if score > prev_score:
                                    best_scores[ipline][tail_str] = (metre_mapping, new_stress_line)

                # Теперь для каждой исходной строки имеем несколько вариантов расстановки ударений.
                # Перебираем сочетания этих вариантов, проверяем рифмовку и оставляем лучший вариант для данной метра.
                stressed_lines2 = [list() for _ in range(4)]
                for iline, items2 in best_scores.items():
                    stressed_lines2[iline].extend(items2.values())

                vvx = list(itertools.product(*stressed_lines2))
                for ivar, plinev in enumerate(vvx):
                    # plinev это набор из двух экземпляров кортежей (MetreMappingResult, LineStressVariant).

                    # Различные дефекты ритма
                    metre_defects_score = 1.0
                    # 11.12.2022 сдвиг одной строки на 1 позицию
                    nprefixa = collections.Counter(pline[0].prefix for pline in plinev)
                    if nprefixa.get(0) == 1 or nprefixa.get(1) == 1:
                        metre_defects_score *= 0.1

                    nsyllaba = collections.Counter(len(pline[1].stress_signature) for pline in plinev)
                    if len(nsyllaba) > 2:
                        # Есть более 2 длин строк в слогах
                        metre_defects_score *= 0.1
                    else:
                        for nsyllab, num in nsyllaba.most_common():
                            if num == 3:
                                # В одной строке число слогов отличается от 3х других строк:
                                #
                                # У Лукоморья дуб зеленый,   - 9 слогов
                                # Под ним живут русалки,     - 7 слогов
                                # А в будках - покемоны,     - 7 слогов
                                # Китайские пугалки.         - 7 слогов
                                metre_defects_score *= 0.1

                    # Определяем рифмуемость
                    rhyme_scheme = None
                    last_pwords = [pline[1].get_rhyming_tail() for pline in plinev]

                    r01 = self.check_rhyming(last_pwords[0], last_pwords[1])
                    r02 = self.check_rhyming(last_pwords[0], last_pwords[2])
                    r03 = self.check_rhyming(last_pwords[0], last_pwords[3])
                    r12 = self.check_rhyming(last_pwords[1], last_pwords[2])
                    r13 = self.check_rhyming(last_pwords[1], last_pwords[3])
                    r23 = self.check_rhyming(last_pwords[2], last_pwords[3])

                    rhyme_score = 1.0
                    if r01 and r12 and r23:
                        # 22.04.2022 отдельно детектируем рифмовку AAAA, так как она зачастую выглядит очень неудачно и ее
                        # желательно устранять из обучающего датасета.
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
                        rhyme_score = 0.75
                    elif not r02 and r13:
                        rhyme_scheme = '-A-A'
                        rhyme_score = 0.75
                    elif r12 and not r01 and not r23:
                        rhyme_scheme = '-AA-'
                        rhyme_score = 0.75
                    else:
                        rhyme_scheme = '----'
                        rhyme_score = 0.50

                    total_score = metre_defects_score * rhyme_score * mul([pline[0].get_score() for pline in plinev])
                    if total_score > best_score:
                        best_score = total_score
                        best_metre = metre_name
                        best_rhyme_scheme = rhyme_scheme
                        best_variant = plinev

        if best_variant is None:
            # В этом случае вернем результат с нулевым скором и особым текстом, чтобы
            # можно было вывести в лог строки с каким-то дефолтными
            return PoetryAlignment.build_no_rhyming_result([pline.get_first_stress_variants(self) for pline in plines])
        else:
            # Возвращаем найденный вариант разметки и его оценку
            best_lines = [v[1] for v in best_variant]
            metre_mappings = [v[0] for v in best_variant]
            return PoetryAlignment(best_lines, best_score, best_metre, rhyme_scheme=best_rhyme_scheme, metre_mappings=metre_mappings)

    def align_nonstandard_block(self, lines):
        plines = [PoetryLine.build(line, self.udpipe, self.accentuator) for line in lines]

        # 08.11.2022 добавлена защита от взрыва числа переборов для очень плохих генераций.
        for pline in plines:
            if sum((pword.n_vowels>1) for pword in pline.pwords) >= 8:
                raise ValueError('Line is too long: "{}"'.format(pline))

        best_score = 0.0
        best_metre = None
        best_rhyme_scheme = None
        best_variant = None

        allow_stress_shift = True

        # Заранее сгенерируем для каждого слова варианты спеллчека и ударения...
        stressed_words_groups = [StressVariantsSlot.build(poetry_words=pline.pwords, aligner=self, allow_stress_shift=allow_stress_shift) for pline in plines]

        # Для каждой строки перебираем варианты разметки и оставляем по ~2 варианта в каждом метре.
        for metre_name, metre_signature in meters:
            best_scores = dict()

            # В каждой строке перебираем варианты расстановки ударений.
            for ipline, pline in enumerate(plines):
                best_scores[ipline] = dict()

                prefix = 0
                cursor = MetreMappingCursor(metre_signature, prefix=prefix)
                for metre_mapping in cursor.map(stressed_words_groups[ipline], self):
                    stressed_words = [m.word for m in metre_mapping.word_mappings]
                    new_stress_line = LineStressVariant(pline, stressed_words, self)

                    if new_stress_line.get_rhyming_tail().is_ok():
                        tail_str = new_stress_line.get_rhyming_tail().__repr__()
                        score = metre_mapping.get_score()
                        if tail_str not in best_scores[ipline]:
                            prev_score = -1e3
                        else:
                            prev_score = best_scores[ipline][tail_str][0].get_score()
                        if score > prev_score:
                            best_scores[ipline][tail_str] = (metre_mapping, new_stress_line)

            # Теперь для каждой исходной строки имеем несколько вариантов расстановки ударений.
            # Перебираем сочетания этих вариантов, проверяем рифмовку и оставляем лучший вариант для данной метра.
            stressed_lines2 = [list() for _ in range(len(best_scores))]
            for iline, items2 in best_scores.items():
                stressed_lines2[iline].extend(items2.values())

            # дальше все нехорошо сделано, так как число вариантов будет запредельное.
            # надо переделать на какую-то динамическую схему подбора.
            # пока тупо ограничим перебор первыми вариантами.
            vvx = list(itertools.product(*stressed_lines2))[:1000]
            for ivar, plinev in enumerate(vvx):
                # plinev это набор из двух экземпляров кортежей (MetreMappingResult, LineStressVariant).

                # Различные дефекты ритма
                metre_defects_score = 1.0
                # 11.12.2022 сдвиг одной строки на 1 позицию
                nprefixa = collections.Counter(pline[0].prefix for pline in plinev)
                if nprefixa.get(0) == 1 or nprefixa.get(1) == 1:
                    metre_defects_score *= 0.1

                # Определяем рифмуемость
                rhyme_scheme = None
                last_pwords = [pline[1].get_rhyming_tail() for pline in plinev]
                # TODO ???

                total_score = metre_defects_score * mul([pline[0].get_score() for pline in plinev])
                if total_score > best_score:
                    best_score = total_score
                    best_metre = metre_name
                    best_rhyme_scheme = rhyme_scheme
                    best_variant = plinev

        if best_variant is None:
            # В этом случае вернем результат с нулевым скором и особым текстом, чтобы
            # можно было вывести в лог строки с каким-то дефолтными
            return None, None, None  #PoetryAlignment.build_no_rhyming_result([pline.get_first_stress_variants(self) for pline in plines])
        else:
            # Возвращаем найденный вариант разметки и его оценку
            best_lines = [v[1] for v in best_variant]
            metre_mappings = [v[0] for v in best_variant]
            return best_lines, metre_mappings, best_metre

    def align_nonstandard_blocks(self, lines):
        stressed_lines = []
        metre_mappings = []
        total_score = 1.0
        best_metre = None

        # Разобьем весь текст на блоки по границам пустых строк
        block_lines = []
        lines2 = lines + ['']
        for line in lines2:
            if len(line) == 0:
                if block_lines:
                    stressed_lines_i, metre_mappings_i, metre_i = self.align_nonstandard_block(lines)
                    best_metre = metre_i
                    stressed_lines.extend(stressed_lines_i)
                    metre_mappings.extend(metre_mappings_i)

                # добаваляем пустую строку, разделявшую блоки.
                stressed_lines.append(LineStressVariant.build_empty_line())
                metre_mappings.append(MetreMappingResult.build_for_empty_line())
                block_lines = []
            else:
                block_lines.append(line)

        return PoetryAlignment(stressed_lines[:-1], total_score, best_metre, rhyme_scheme='', metre_mappings=metre_mappings[:-1])

    def build_from_markup(self, text):
        lines = text.split('\n')

        plines = [PoetryLine.build_from_markup(line, self.udpipe) for line in lines]
        stressed_lines = [pline.get_first_stress_variants(self) for pline in plines]

        mapped_meter, mapping_score = self.map_meters(stressed_lines)
        score = mapping_score * reduce(lambda x, y: x * y, [l.get_score() for l in stressed_lines])
        rhyme_scheme = ''

        # 13.08.2022 определяем схему рифмовки
        if len(lines) == 2:
            rhyme_scheme = '--'
            claus1 = stressed_lines[0].get_rhyming_tail()
            claus2 = stressed_lines[1].get_rhyming_tail()
            if self.check_rhyming(claus1, claus2):
                rhyme_scheme = 'AA'
        elif len(lines) == 4:
            rhyme_scheme = '----'
            claus1 = stressed_lines[0].get_rhyming_tail()
            claus2 = stressed_lines[1].get_rhyming_tail()
            claus3 = stressed_lines[2].get_rhyming_tail()
            claus4 = stressed_lines[3].get_rhyming_tail()

            r12 = self.check_rhyming(claus1, claus2)
            r13 = self.check_rhyming(claus1, claus3)
            r14 = self.check_rhyming(claus1, claus4)
            r23 = self.check_rhyming(claus2, claus3)
            r24 = self.check_rhyming(claus2, claus4)
            r34 = self.check_rhyming(claus3, claus4)

            if r12 and r23 and r34:
                rhyme_scheme = 'AAAA'
            elif r13 and r24 and not r12 and not r34:
                rhyme_scheme = 'ABAB'
            elif r12 and not r23 and r34:
                rhyme_scheme = 'AABB'
            elif r12 and not r23 and not r23 and r14:
                rhyme_scheme = 'AABA'
            elif r14 and r23 and not r12 and not r34:
                rhyme_scheme = 'ABBA'
            elif not r12 and r13 and not r34:
                rhyme_scheme = 'A-A-'
            elif not r12 and r24 and not r23:
                rhyme_scheme = '-A-A'

        return PoetryAlignment(stressed_lines, score, mapped_meter, rhyme_scheme=rhyme_scheme, metre_mappings=None)

    def detect_repeating(self, alignment, strict=False):
        # Повтор последних слов в разных строках
        last_words = [pline.get_rhyming_tail().stressed_word.form.lower() for pline in alignment.poetry_lines]
        for i1, word1 in enumerate(last_words):
            for word2 in last_words[i1+1:]:
                if word1 == word2:
                    return True

        return self.detect_repeating_in_line(alignment, strict)

    def detect_repeating_in_line(self, alignment, strict=False):
        # Иногда генеративная модель выдает повторы существительных типа "любовь и любовь" в одной строке.
        # Такие генерации выглядят криво.
        # Данный метод детектирует повтор леммы существительного в строке.
        # 22.10.2022 добавлен учет глаголов и прилагательных
        for pline in alignment.poetry_lines:
            n_lemmas = collections.Counter()
            for pword in pline.poetry_line.pwords:
                if pword.upos in ('NOUN', 'PROPN', 'ADJ', 'VERB'):
                    n_lemmas[pword.lemma] += 1
                elif strict and pword.upos in ('ADV', 'PRON', 'SYM'):
                    n_lemmas[pword.lemma] += 1
            if n_lemmas and n_lemmas.most_common(1)[0][1] > 1:
                return True

            if strict:
                # Повтор слова длиннее 4 букв тоже считаем плохим
                n_forms = collections.Counter()
                for pword in pline.poetry_line.pwords:
                    if len(pword.form) >= 5:
                        n_forms[pword.upos + ':' + pword.form.lower().replace('ё', 'е')] += 1
                if n_forms and n_forms.most_common(1)[0][1] > 1:
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

            # 01-11-2022 повтор формы существительного
            nouns = collections.Counter(w.form.lower() for w in pline.poetry_line.pwords if w.upos in ('NOUN', 'PROPN'))
            if len(nouns) > 0:
                if nouns.most_common(1)[0][1] > 1:
                    return True

            # 01-11-2022 наличие в одной строке вариантов с "ё" и с "е" считаем повтором
            forms = [w.form.lower() for w in pline.poetry_line.pwords]
            for v1, v2 in [('поёт', 'поет')]:
                if v1 in forms and v2 in forms:
                    return True

            # 14.08.2023 попадаются повторы из личной формы глагола и деепричастия:
            # Когда лишь любишь ты любя.
            #            ^^^^^^^^^^^^^^
            #print('DEBUG@1992')
            vlemmas = set()
            conv_lemmas = set()
            for w in pline.poetry_line.pwords:
                if w.upos == 'VERB':
                    #print('DEBUG@1997 w.form=', w.form,  w.lemma)
                    if w.get_attr('VerbForm') == 'Conv':
                        conv_lemmas.add(w.lemma)
                    else:
                        vlemmas.add(w.lemma)

            if any((v in conv_lemmas) for v in vlemmas):
                #print('DEBUG@2004')
                return True

        return False

    def detect_poor_poetry(self, alignment):
        """Несколько эвристик для обнаружения скучных рифм, которые мы не хотим получать"""

        last_words = [pline.get_rhyming_tail().stressed_word.form.lower() for pline in alignment.poetry_lines]

        # Проверяем банальный повтор слова
        # 18.08.2023 проверяем случаи "легко-нелегко"
        for i1, word1 in enumerate(last_words[:-1]):
            for word2 in last_words[i1+1:]:
                form1 = word1.lower().replace('ё', 'е')
                form2 = word2.lower().replace('ё', 'е')
                if form1 == form2:
                    return True
                if form1 == 'не'+form2 or 'не'+form1 == form2:
                    return True

        # Если два глагольных окончания, причем одно является хвостом другого - это бедная рифма:
        # ждать - подождать
        # смотреть - посмотреть
        # etc.
        rhyme_pairs = []
        if alignment.rhyme_scheme in ('ABAB', 'A-A-', '-A-A'):
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
        elif alignment.rhyme_scheme in ('AAAA', '----'):
            rhyme_pairs.append((alignment.poetry_lines[0].get_rhyming_tail(), alignment.poetry_lines[1].get_rhyming_tail()))
            rhyme_pairs.append((alignment.poetry_lines[1].get_rhyming_tail(), alignment.poetry_lines[2].get_rhyming_tail()))
            rhyme_pairs.append((alignment.poetry_lines[2].get_rhyming_tail(), alignment.poetry_lines[3].get_rhyming_tail()))

        for tail1, tail2 in rhyme_pairs:
            word1 = tail1.stressed_word
            word2 = tail2.stressed_word

            form1 = word1.poetry_word.form.lower()
            form2 = word2.poetry_word.form.lower()

            if word1.poetry_word.upos == 'VERB' and word2.poetry_word.upos == 'VERB':
                # 11-01-2022 если пара слов внесена в специальный список рифмующихся слов, то считаем,
                # что тут все нормально:  ВИТАЮ-ТАЮ
                if (form1, form2) in self.accentuator.rhymed_words:
                    continue

                #if any((form1.endswith(e) and form2.endswith(e)) for e in 'ли ла ло л м шь т тся у те й ю ь лись лась лось лся тся ться я шись в'.split(' ')):
                if any((form1.endswith(e) and form2.endswith(e)) for e in 'ли ла ло л тся те лись лась лось лся тся ться шись'.split(' ')):
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

    def detect_rhyme_repeatance(self, alignment):
        """Обнаруживаем повтор слова в рифмовке"""
        last_words = [pline.get_rhyming_tail().stressed_word.form.lower() for pline in alignment.poetry_lines]

        # Проверяем банальный повтор слова
        for i1, word1 in enumerate(last_words[:-1]):
            for word2 in last_words[i1+1:]:
                form1 = word1.lower().replace('ё', 'е')
                form2 = word2.lower().replace('ё', 'е')
                if form1 == form2:
                    return True

        return False

    def analyze_defects(self, alignment):
        defects = Defects()

        # Анализируем количество слогов в строках
        nvs = [count_vowels(line.get_unstressed_line()) for line in alignment.poetry_lines]

        nv1 = nvs[0]
        nv_deltas = [(nv - nv1) for nv in nvs]

        if len(set(nv_deltas)) != 1:
            # Не все строки имеют равное число слогов. Выясняем подробности.
            are_good_nvs = False

            if alignment.rhyme_scheme in ('ABAB', 'A-A-', '-A-A'):
                if nv_deltas[0] == nv_deltas[2] and nv_deltas[1] == nv_deltas[3]:
                    delta = abs(nv_deltas[0] - nv_deltas[2])  # чем сильнее отличается число шагов, тем больше штраф
                    defects.add_defect(Defect(penalty=0.95*delta, description='@1858'))
                    are_good_nvs = True
                elif nv_deltas[0] == nv_deltas[1] and nv_deltas[1] == nv_deltas[2]:
                    # Первые три строки имеют равно число слогов, а последняя - отличается
                    delta = abs(nv_deltas[0] - nv_deltas[3])  # чем сильнее отличается число шагов, тем больше штраф
                    defects.add_defect(Defect(penalty=0.90*delta, description='@1863'))
                    are_good_nvs = True
            elif alignment.rhyme_scheme in ('ABBA', '-AA-'):
                if (nv_deltas[0] == nv_deltas[3]) and (nv_deltas[1] == nv_deltas[2]):
                    delta = abs(nv_deltas[0] - nv_deltas[1])
                    defects.add_defect(Defect(penalty=0.90 * delta, description='@1868'))
                    are_good_nvs = True
            else:
                nvz = collections.Counter(nv_deltas)
                if len(nvz) == 2:
                    m = nvz.most_common()
                    delta = max(nv_deltas) - min(nv_deltas)
                    if m[0][1] == 2 and m[1][1] == 2:
                        defects.add_defect(Defect(penalty=0.85 * delta, description='@1876'))
                        are_good_nvs = True
                    elif nv_deltas[3] == m[1][0]:
                        # Последняя строка в катрене имеет отличное число слогов от трех других.
                        defects.add_defect(Defect(penalty=0.80 * delta, description='@1880'))
                        are_good_nvs = True

            if not are_good_nvs:
                defects.add_defect(Defect(0.5, description='@1884'))

        return defects

    def markup_rap_line(self, line):
        opt_words = ['лишь', 'вроде', 'если', 'чтобы', 'когда', 'просто', 'мимо', 'даже', 'всё', 'хотя', 'едва', 'нет',
                     'эти', 'эту', 'это', 'мои', 'твои', 'моих', 'твоих', 'моим', 'твоим', 'моей', 'твоей',
                     'мою', 'твою', 'его', 'ее', 'её', 'себе', 'тебя', 'свою', 'свои', 'своим', 'они', 'она',
                     'уже', 'есть', 'раз', 'быть']
        res_tokens = []
        parsings = self.udpipe.parse_text(line)
        if parsings is None:
            raise RuntimeError()

        for parsing in parsings:
            for ud_token in parsing:
                stress_pos = 0
                word = ud_token.form.lower()
                nvowels = sum((c in 'уеыаоэёяию') for c in word)

                if ud_token.upos in ('PRON', 'ADV', 'DET') and nvowels == 1:
                    # Односложные наречия, местоимения и т.д.
                    is_optional_stress = True
                elif ud_token.upos in ('PUNCT', 'ADP', 'PART', 'SCONJ', 'CCONJ', 'INTJ'):
                    is_optional_stress = True
                elif word in opt_words:
                    is_optional_stress = True
                else:
                    is_optional_stress = False

                if not is_optional_stress:
                    stress_pos = self.accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])

                cx = []
                vowel_counter = 0
                for c in ud_token.form:
                    cx.append(c)
                    if c.lower() in 'уеыаоэёяию':
                        vowel_counter += 1
                        if vowel_counter == stress_pos:
                            cx.append('\u0301')
                token2 = ''.join(cx)
                res_tokens.append({'form': ud_token.form, 'upos': ud_token.upos, 'tags': ud_token.tags, 'stress_pos': stress_pos, 'rendition': token2})

        return res_tokens

    def get_rhyming_tail(self, rap_tokens):
        while rap_tokens:
            if re.search(r'\w', rap_tokens[-1]['form']) is not None:
                return rap_tokens[-1]
            rap_tokens = rap_tokens[:-1]
        return None

    def align_rap(self, rap_lines):
        markups = [self.markup_rap_line(line) for line in rap_lines]
        res = RapAlignment()

        total_word_vocab = collections.Counter()
        total_word_count = 0

        # Рифмовка последних слов соседних строк или через одну строку.
        for iline1, line1 in enumerate(markups):
            if len(line1) > 0:
                # Поищем дефекты генерации по этой строке.

                line_words = [t['form'].lower().replace('ё', 'е') for t in line1 if re.match(r'^\w+$', t['form']) is not None]
                total_word_vocab.update(line_words)
                total_word_count += len(line_words)

                # 1) Если строка состоит из повторов одного слова (не считая пунктуации):
                # Котик котик,
                # ^^^^^^^^^^^^
                word_counter = collections.Counter(line_words)
                if sum(word_counter.values()) > 1 and len(word_counter.items()) == 1:
                    res.defects.append(('строка = повтор одного слова "{}"'.format(word_counter.most_common()[0][0]), 1))

                last_token1 = self.get_rhyming_tail(line1)
                for line2 in markups[iline1+1: iline1+3]:
                    # TODO учесть безударные слова в конце...
                    last_token2 = self.get_rhyming_tail(line2)

                    if last_token1 and last_token2:
                        if last_token1['stress_pos'] > 0 and last_token2['stress_pos'] > 0:
                            if last_token1['form'].lower().replace('ё', 'е') != last_token2['form'].lower().replace('ё', 'е'):
                                r = rhymed_fuzzy2(self.accentuator,
                                                  last_token1['form'], last_token1['stress_pos'], [last_token1['upos']]+last_token1['tags'], '', '',
                                                  last_token2['form'], last_token2['stress_pos'], [last_token2['upos']]+last_token2['tags'], '', '')

                                if not r:
                                    # Попробуем менее четкое сравнение, только для рэпа, типа грусть-вернусь
                                    xword1, clausula1 = render_xword(self.accentuator, last_token1['form'], last_token1['stress_pos'], [last_token1['upos']]+last_token1['tags'], '', '',)
                                    xword2, clausula2 = render_xword(self.accentuator, last_token2['form'], last_token2['stress_pos'], [last_token2['upos']]+last_token2['tags'], '', '')

                                    stressed_vowel1 = re.search(r'\^([аеёиоуыэюя])', xword1)
                                    stressed_vowel1 = stressed_vowel1.group(1) if stressed_vowel1 else '<1>'

                                    stressed_vowel2 = re.search(r'\^([аеёиоуыэюя])', xword2)
                                    stressed_vowel2 = stressed_vowel2.group(1) if stressed_vowel2 else '<2>'

                                    if stressed_vowel1 == stressed_vowel2:
                                        d = jellyfish.levenshtein_distance(clausula1, clausula2)
                                        if d < 2:
                                            r = True

                                if r:
                                    res.rhymes.append((last_token1['form'], last_token1['rendition'], last_token2['form'], last_token2['rendition']))

                res.denominator += 1  # ведем учет проверок, чтобы тексты разной длины привести к общему знаменателю

        # Ищем дегенеративный инференс:
        #
        # У котика, у котика,
        # Розовый животик.
        # У котика, у котика,
        # Маленький животик.
        #
        # У котика, у котика,
        # Маленький животик.
        # У котика, у котика,
        # Маленький животик.
        if len(total_word_vocab) < total_word_count//2:
            res.defects.append(('дегенерация@2247', 100.0))


        return res

