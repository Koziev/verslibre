"""
15-12-2021 Введен сильный штраф за 2 подряд ударных слога
17-12-2021 Регулировка ударности некоторых наречий и частиц, удаление лишних пробелов вокруг дефиса при выводе строки с ударениями
18-12-2021 Коррекция пробелов вынесена в отдельный модуль whitespace_normalization
"""

import itertools
from functools import reduce
import os
import io
import jellyfish
import hyperopt
import re
from hyperopt import hp, tpe, STATUS_OK, Trials

from poetry.phonetic import Accents, rhymed
from generative_poetry.udpipe_parser import UdpipeParser
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
COEFF['@225'] = 0.9
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

        self.stressed_form = ''.join(output)

    def get_stressed_form(self):
        return self.stressed_form

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
    def __init__(self, form, upos, tags, stress_pos):
        self.form = form
        self.upos = upos
        self.tags = tags
        self.stress_pos = stress_pos

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

    def get_stress_variants(self):
        variants = []

        nvowels = sum((c in 'уеыаоэёяию') for c in self.form.lower())
        uform = self.form.lower()

        if self.upos in ('ADP', 'CCONJ', 'SCONJ', 'PART', 'INTJ'):
            # Предлоги, союзы, частицы предпочитаем без ударения,
            # поэтому базовый вариант добавляем с дисконтом:
            if uform in ['лишь', 'вроде', 'если', 'чтобы', 'когда', 'просто', 'мимо', 'даже', 'всё', 'хотя', 'едва', 'нет']:
                variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68_2']))
            else:
                variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68']))

            # а вариант без ударения - с нормальным скором:
            variants.append(WordStressVariant(self, -1, COEFF['@71']))
        elif self.upos in ('PRON', 'ADV', 'DET'):
            # Для односложных местоимений (Я), наречий (ТУТ, ГДЕ) и слов типа МОЙ, ВСЯ добавляем вариант без ударения с дисконтом
            if nvowels == 1:
                variants.append(WordStressVariant(self, self.stress_pos, COEFF['@75']))
                # вариант без ударения
                variants.append(WordStressVariant(self, -1, COEFF['@77']))
            else:
                if uform in ['эти', 'эту', 'это', 'мои', 'твои', 'моих', 'твоих', 'моим', 'твоим', 'моей', 'твоей', 'мою', 'твою', 'его', 'ее', 'её']:
                    # Безударный вариант для таких двусложных прилагательных
                    variants.append(WordStressVariant(self, -1, COEFF['@77_2']))

                variants.append(WordStressVariant(self, self.stress_pos, COEFF['@79']))
        else:
            if uform in ['есть', 'раз']:
                # безударный вариант
                variants.append(WordStressVariant(self, self.stress_pos, COEFF['@143']))

            # Добавляем исходный вариант с ударением
            variants.append(WordStressVariant(self, self.stress_pos, 1.0))

        # TODO: + сделать вариант смещения позиции ударения в существительном или глаголе
        # ...

        return variants


def sum1(arr):
    return sum((x == 1) for x in arr)


class LineStressVariant(object):
    def __init__(self, poetry_line, stressed_words):
        self.poetry_line = poetry_line
        self.stressed_words = stressed_words
        self.stress_signature = list(itertools.chain(*(w.stress_signature for w in stressed_words)))
        self.stress_signature_str = ''.join(map(str, self.stress_signature))
        self.total_score = reduce(lambda x, y: x*y, [w.score for w in stressed_words])
        # добавка от 15-12-2021: два подряд ударных слога наказываем сильно!
        if '11' in self.stress_signature_str:
            self.total_score *= 0.1
        # добавка от 16-12-2021: безударное последнее слово наказываем сильно!
        if self.stressed_words[-1].new_stress_pos == -1:
            self.total_score *= 0.1


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

    def map_meter(self, signature):
        l = len(signature)
        n = len(self.stress_signature) // l
        if (len(self.stress_signature) % l) > 0:
            n += 1

        expanded_sign = (signature * n)[:len(self.stress_signature)]

        hits = sum((x == y) for x, y in zip(expanded_sign, self.stress_signature) if (x==1 or y==1))
        score = float(hits) / max(sum1(expanded_sign), sum1(self.stress_signature), 1e-6)

        if score < 1.0 and signature[0] == 1 and self.stress_signature[0] == 0:
            # Попробуем сместить вправо, добавив в начало один неударный слог.
            expanded_sign2 = (0,) + expanded_sign[:-1]
            hits2 = sum((x == y) for x, y in zip(expanded_sign2, self.stress_signature) if (x==1 or y==1))
            score2 = COEFF['@126'] * float(hits2) / max(sum1(expanded_sign2), sum1(self.stress_signature), 1e-6)
            if score2 > score:
                return score2

        return score

    def split_to_syllables(self):
        output_tokens = []
        for word in self.stressed_words:
            if len(output_tokens) > 0:
                output_tokens.append('|')
            output_tokens.extend(word.split_to_syllables())
        return output_tokens


def count_vowels(s):
    return sum((c.lower() in 'уеыаоэёяию') for c in s)


class PoetryLine(object):
    def __init__(self):
        self.text = None
        self.pwords = None
        self.stress_signature = None

    @staticmethod
    def build(text, udpipe_parser, accentuator):
        pline = PoetryLine()
        pline.text = text
        pline.pwords = []

        parsings = udpipe_parser.parse_text(text)
        for parsing in parsings:
            for ud_token in parsing:
                word = ud_token.form.lower()
                stress_pos = accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])
                if count_vowels(word) > 0 and stress_pos == -1:
                    msg = 'Could not locate stress position in word "{}"'.format(word)
                    raise ValueError(msg)

                pword = PoetryWord(ud_token.form, ud_token.upos, ud_token.tags, stress_pos)
                pline.pwords.append(pword)

        return pline

    def __repr__(self):
        return ' '.join([pword.__repr__() for pword in self.pwords])

    def get_stress_variants(self):
        wordx = [pword.get_stress_variants() for pword in self.pwords]
        variants = [LineStressVariant(self, swords) for swords in itertools.product(*wordx)]
        return variants

    def get_last_rhyming_word(self):
        for pword in self.pwords[::-1]:
            if pword.upos != 'PUNCT':
                return pword
        return None


class PoetryAlignment(object):
    def __init__(self, poetry_lines, score, meter, rhyme_scheme):
        self.poetry_lines = poetry_lines
        self.score = score
        self.meter = meter
        self.rhyme_scheme = rhyme_scheme

    def __repr__(self):
        s = '{} {}({:5.3f}):\n'.format(self.meter, self.rhyme_scheme, self.score)
        s += '\n'.join(map(str, self.poetry_lines))
        return s

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


class PoetryStressAligner(object):
    def __init__(self, udpipe, accentuator):
        self.udpipe = udpipe
        self.accentuator = accentuator

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

    def map_2signatures(self, sline1, sline2):
        d = jellyfish.damerau_levenshtein_distance(sline1.stress_signature_str, sline2.stress_signature_str)
        score = 1.0 - float(d) / max(len(sline1.stress_signature), len(sline2.stress_signature), 1e-6)
        return score

    def align(self, lines0, check_rhymes=True):
        # Иногда для наглядности можем выводить сгенерированные стихи вместе со значками ударения.
        # Эти значки мешают работе алгоритма транскриптора, поэтому уберем их сейчас.
        lines = [line.replace('\u0301', '') for line in lines0]

        if len(lines) == 2:
            return self.align2(lines)
        elif len(lines) == 4:
            return self.align4(lines, check_rhymes)
        else:
            raise NotImplementedError()

    def check_rhyming(self, poetry_word1, poetry_word2):
        return rhymed(self.accentuator,
                      poetry_word1.form, [poetry_word1.upos] + poetry_word1.tags,
                      poetry_word2.form, [poetry_word2.upos] + poetry_word2.tags)

    def align4(self, lines, check_rhymes):
        plines = [PoetryLine.build(line, self.udpipe, self.accentuator) for line in lines]

        # Проверим, что эти 4 строки имеют рифмовку
        if check_rhymes:
            rhyme_scheme = None
            last_pwords = [pline.get_last_rhyming_word() for pline in plines]
            if self.check_rhyming(last_pwords[0], last_pwords[2]) and self.check_rhyming(last_pwords[1], last_pwords[3]):
                # Считаем, что слово не рифмуется само с собой, чтобы не делать для отсечения
                # таких унылых рифм отдельную проверку в другой части кода.
                if last_pwords[0].form.lower() != last_pwords[2].form.lower() and last_pwords[1].form.lower() != last_pwords[3].form.lower():
                    rhyme_scheme = 'ABAB'
            elif self.check_rhyming(last_pwords[0], last_pwords[3]) and self.check_rhyming(last_pwords[1], last_pwords[2]):
                if last_pwords[0].form.lower() != last_pwords[3].form.lower() and last_pwords[1].form.lower() != last_pwords[2].form.lower():
                    rhyme_scheme = 'ABBA'

            if rhyme_scheme is None:
                return None
        else:
            rhyme_scheme = 'ABAB'

        # Список вариантов простановки ударения с учётом опциональности ударений для союзов, предлогов и т.д.
        plinevx = [pline.get_stress_variants() for pline in plines]

        # Идем по списку вариантов, отображаем на эталонные метры и ищем лучший вариант.
        best_variant = None
        best_score = 0.0
        best_meter = None
        best_ivar = None

        vvx = list(itertools.product(*plinevx))
        if len(vvx) > 10000:
            raise ValueError('Too many optional stresses: {}'.format(len(vvx)))

        for ivar, plinev in enumerate(vvx):
            # plinev это набор из четырех экземпляров LineStressVariant.

            rhyming_score = 0.0
            if rhyme_scheme == 'ABAB':
                # Оцениваем, насколько хорошо соответствуют сигнатуры строк для схемы рифмовки ABAB
                score_13 = self.map_2signatures(plinev[0], plinev[2])
                score_24 = self.map_2signatures(plinev[1], plinev[3])
                rhyming_score = 1.0 - COEFF['@225']*(1.0 - score_13*score_24)
            elif rhyme_scheme == 'ABBA':
                score_14 = self.map_2signatures(plinev[0], plinev[3])
                score_23 = self.map_2signatures(plinev[1], plinev[2])
                rhyming_score = 1.0 - COEFF['@225']*(1.0 - score_14*score_23)
            else:
                raise NotImplementedError()

            # Ищем лучшее отображение метра.
            mapped_meter, mapping_score = self.map_meters(plinev)
            score = rhyming_score * mapping_score * reduce(lambda x, y: x*y, [l.get_score() for l in plinev])
            if score > best_score:
                best_variant = plinev
                best_score = score
                best_meter = mapped_meter
                best_ivar = ivar

        # Возвращаем найденный вариант разметки и его оценку
        return PoetryAlignment(best_variant, best_score, best_meter, rhyme_scheme)

    def align2(self, lines):
        plines = [PoetryLine.build(line, self.udpipe, self.accentuator) for line in lines]
        plinevx = [pline.get_stress_variants() for pline in plines]

        rhyme_scheme = None
        last_pwords = [pline.get_last_rhyming_word() for pline in plines]
        if self.check_rhyming(last_pwords[0], last_pwords[1]):
            # Считаем, что слово не рифмуется само с собой, чтобы не делать для отсечения
            # таких унылых рифм отдельную проверку в другой части кода.
            if last_pwords[0].form.lower() != last_pwords[1].form.lower():
                rhyme_scheme = 'AA'

        if rhyme_scheme is None:
            return None

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

            score_12 = self.map_2signatures(plinev[0], plinev[1])

            # Ищем лучшее отображение метра.
            mapped_meter, mapping_score = self.map_meters(plinev)
            score = score_12 * mapping_score * reduce(lambda x, y: x*y, [l.get_score() for l in plinev])
            if score > best_score:
                best_variant = plinev
                best_score = score
                best_meter = mapped_meter
                best_ivar = ivar

        # Возвращаем найденный вариант разметки и его оценку
        return PoetryAlignment(best_variant, best_score, best_meter, rhyme_scheme=rhyme_scheme)


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

    udpipe = UdpipeParser()
    udpipe.load(models_dir)

    accents = Accents()
    accents.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
    accents.after_loading(stress_model_dir=os.path.join(tmp_dir, 'stress_model'))

    aligner = PoetryStressAligner(udpipe, accents)

    # ================================================
    if False:
        good_poems = []
        with io.open(os.path.join(data_dir, 'poetry', 'эталонные_четверостишья.txt'), 'r', encoding='utf-8') as rdr:
            lines = []
            for line in rdr:
                s = line.strip()
                if s:
                    lines.append(s)
                else:
                    if len(lines) == 4:
                        good_poems.append(lines)

                    lines = []

        # НАЧАЛО ОТЛАДКИ
        poem = """У меня жена седая.
        Боль всю душу забрала.
        И она, как молодая,
        В дом пришла и умерла."""
        poem = [z.strip() for z in poem.split('\n') if z.strip()]
        #good_poems = [poem]
        objective(COEFF)
        # КОНЕЦ ОТЛАДКИ

        space = dict((coef_name, hp.uniform(coef_name, 0.0, 1.0)) for coef_name in COEFF.keys())

        trials = Trials()
        best = hyperopt.fmin(fn=objective,
                             space=space,
                             algo=HYPEROPT_ALGO,
                             max_evals=100,
                             trials=trials,
                             verbose=0)
        print('Best parameters: {}'.format(best))

        COEFF = best
        poem = """У меня жена седая.
        Боль всю душу забрала.
        И она, как молодая,
        В дом пришла и умерла."""
        poem = [z.strip() for z in poem.split('\n') if z.strip()]
        alignment = aligner.align(poem)
        print(alignment)

        exit(0)


    # ================================================

    poem = """Тебя и нет - ты вместе с нами
В тебе искать иных решений
А ты, Россия, вместе с нами
Не зная сложностей, сомнений"""
    poem = [z.strip() for z in poem.split('\n') if z.strip()]

    alignment = aligner.align(poem)
    print(alignment)
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
