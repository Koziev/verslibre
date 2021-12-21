# -*- coding: utf-8 -*-

"""
Фонетический словарь (ударятор).
При выполнении словарь будет загружен из текстовых файлов, подготовлен и сохранен в pickle-файле
для последующей быстрой подгрузки.

LC -> 27.06.2021
"""

import json
import yaml
import pickle
import os
import io
import six
import codecs
import operator
import logging
import re

from nltk.stem.snowball import RussianStemmer

#from russ.stress.model import StressModel
from transcriptor_models.stress_model.stress_model import StressModel

import rusyllab

from poetry.corpus_analyser import CorpusWords


class Accents:
    def __init__(self):
        self.ambiguous_accents = None
        self.word_accents_dict = None
        self.yo_words = None
        self.rhymed_words = set()

    def sanitize_word(self, word):
        return word.lower() #.replace(u'ё', u'е')

    def load(self, data_dir, all_words):
        # пары слов, который будем считать рифмующимися
        with io.open(os.path.join(data_dir, 'rhymed_words.txt'), 'r', encoding='utf-8') as rdr:
            for line in rdr:
                s = line.strip()
                if s and not s.startswith('#'):
                    i = s.index(' ')
                    word1 = s[:i].strip()
                    word2 = s[i+1:].strip()
                    self.rhymed_words.add((word1, word2))

        # однозначаная ёфикация
        path = os.path.join(data_dir, 'solarix_yo.txt')
        logging.info('Loading words with ё from "%s"', path)
        self.yo_words = dict()
        with io.open(path, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                word = line.strip()
                key = word.replace('ё', 'е')
                self.yo_words[key] = word

        # Информация о словах, которые для разных грамматических форм могут давать разное ударение.
        path = os.path.join(data_dir, 'ambiguous_accents.yaml')
        logging.info('Loading ambiguous accents information from "%s"', path)
        d = yaml.safe_load(io.open(path, 'r', encoding='utf-8').read())
        self.ambiguous_accents = d
        logging.info('%d items in ambiguous_accents', len(self.ambiguous_accents))

        self.word_accents_dict = dict()

        if True:
            path = os.path.join(data_dir, 'single_accent.dat')
            logging.info('Loading stress information from "%s"', path)
            with io.open(path, 'r', encoding='utf-8') as rdr:
                for line in rdr:
                    tx = line.split('\t')
                    if len(tx) == 2:
                        word, accent = tx[0], tx[1]
                        n_vowels = 0
                        for c in accent:
                            if c.lower() in 'уеыаоэёяию':
                                n_vowels += 1
                                if c.isupper():
                                    stress = n_vowels
                                    self.word_accents_dict[word] = stress

        if True:
            path2 = os.path.join(data_dir, 'accents.txt')
            logging.info('Loading stress information from "%s"', path2)
            with codecs.open(path2, 'r', 'utf-8') as rdr:
                for line in rdr:
                    tx = line.strip().split(u'#')
                    if len(tx) == 2:
                        forms = tx[1].split(u',')
                        for form in forms:
                            word = self.sanitize_word(form.replace('\'', '').replace('`', ''))
                            if all_words is None or word in all_words:
                                if '\'' in form:
                                    accent_pos = form.index('\'')
                                    nb_vowels_before = self.get_vowel_count(form[:accent_pos])
                                    if word not in self.word_accents_dict:
                                        self.word_accents_dict[word] = nb_vowels_before
                                elif 'ё' in form:
                                    accent_pos = form.index('ё')
                                    nb_vowels_before = self.get_vowel_count(form[:accent_pos])+1
                                    if word not in self.word_accents_dict:
                                        self.word_accents_dict[word] = nb_vowels_before

        if True:
            stress_char = u'́'
            stress2_char = u'̀'
            p3 = os.path.join(data_dir, 'ruwiktionary-accents.txt')
            logging.info('Loading stress information from "%s"', p3)
            with codecs.open(p3, 'r', 'utf-8') as rdr:
                for iline, line in enumerate(rdr):
                    word = line.strip()
                    if '-' not in word:
                        nword = word.replace(stress_char, '').replace('ё', 'е')\
                            .replace('\'', '').replace('ѝ', 'и').replace('ѐ', 'е').replace(stress2_char, '')
                        if len(nword) > 2:
                            if stress_char in word:
                                accent_pos = word.index(stress_char)
                                nb_vowels_before = self.get_vowel_count(word[:accent_pos])
                                if nword not in self.word_accents_dict:
                                    self.word_accents_dict[nword] = nb_vowels_before
                            elif '\'' in word:
                                accent_pos = word.index('\'')
                                nb_vowels_before = self.get_vowel_count(word[:accent_pos])
                                if nword not in self.word_accents_dict:
                                    self.word_accents_dict[nword] = nb_vowels_before
                            elif 'ё' in word:
                                accent_pos = word.index('ё')
                                nb_vowels_before = self.get_vowel_count(word[:accent_pos])
                                if nword not in self.word_accents_dict:
                                    self.word_accents_dict[nword] = nb_vowels_before

        if True:
            path = os.path.join(data_dir, 'words_accent.json')
            logging.info('Loading stress information from "%s"', path)
            d = json.loads(open(path).read())
            for word, a in d.items():
                if '-' not in word:
                    nword = self.sanitize_word(word)
                    if nword not in self.word_accents_dict:
                        self.word_accents_dict[nword] = a

        with io.open(os.path.join(data_dir, 'true_accents.txt'), 'r', encoding='utf-8') as rdr:
            for line in rdr:
                word = line.strip()
                if word:
                    nword = self.sanitize_word(word)
                    if nword in self.ambiguous_accents:
                        del self.ambiguous_accents[nword]
                    accent_char = re.search('([АЕЁИОУЭЮЯ])', word).groups(0)[0]
                    accent_pos = word.index(accent_char)
                    nb_vowels_before = self.get_vowel_count(word[:accent_pos], abbrevs=False) + 1
                    self.word_accents_dict[nword] = nb_vowels_before

        logging.info('%d items in word_accents_dict', len(self.word_accents_dict))

        if False:
            # Делаем отладочный листинг загруженных ударений
            with codecs.open('../tmp/accents.txt', 'w', 'utf-8') as wrt:
                for word, accent in sorted(six.iteritems(self.word_accents_dict), key=operator.itemgetter(0)):
                    wrt.write(u'{}\t{}\n'.format(word, accent))

    def save_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.ambiguous_accents, f)
            pickle.dump(self.word_accents_dict, f)
            pickle.dump(self.yo_words, f)
            pickle.dump(self.rhymed_words, f)

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            self.ambiguous_accents = pickle.load(f)
            self.word_accents_dict = pickle.load(f)
            self.yo_words = pickle.load(f)
            self.rhymed_words = pickle.load(f)

    def after_loading(self, stress_model_dir):
        self.stemmer = RussianStemmer()
        #self.stress_model = StressModel.load()
        self.stress_model = StressModel(stress_model_dir)
        self.predicted_accents = dict()

    def conson(self, c1):
        # Оглушение согласной
        if c1 == 'б':
            return 'п'
        elif c1 == 'в':
            return 'ф'
        elif c1 == 'г':
            return 'к'
        elif c1 == 'д':
            return 'т'
        elif c1 == 'ж':
            return 'ш'
        elif c1 == 'з':
            return 'с'

        return c1

    def yoficate(self, word):
        return self.yo_words.get(word, word)

    def pronounce_full(self, word):
        # Фонетическая транскрипция всего слова
        # Сейчас сделана затычка в виде вызова транскриптора окончания, но
        # это неправильно с точки зрения обработки ударных/безударных гласных.
        return self.pronounce(self.yoficate(word))

    def pronounce(self, s):
        # Фонетическая транскрипция фрагмента слова, НАЧИНАЯ С УДАРНОЙ ГЛАСНОЙ
        #                                            ^^^^^^^^^^^^^^^^^^^^^^^^^
        if s.endswith('жь'):
            # РОЖЬ -> РОЖ
            s = s[:-1]
        elif s.endswith('шь'):
            # МЫШЬ -> МЫШ
            s = s[:-1]

        # СОЛНЦЕ -> СОНЦЕ
        s = s.replace('лнц', 'нц')

        # СЧАСТЬЕ -> ЩАСТЬЕ
        s = s.replace('сч', 'щ')

        # БРАТЬСЯ -> БРАЦА
        s = s.replace('ться', 'ца')

        # БОЯТСЯ -> БОЯЦА
        s = s.replace('тся', 'ца')

        # БРАТЦЫ -> БРАЦА
        s = s.replace('тц', 'ц')

        # ЖИР -> ЖЫР
        s = s.replace('жи', 'жы')

        # ШИП -> ШЫП
        s = s.replace('ши', 'шы')

        # МОЦИОН -> МОЦЫОН
        s = s.replace('ци', 'цы')

        # ЖЁСТКО -> ЖОСТКО
        s = s.replace('жё', 'жо')

        # ОКОНЦЕ -> ОКОНЦЭ
        s = s.replace('це', 'цэ')

        # двойные согласные:
        # СУББОТА -> СУБОТА
        s = re.sub(r'([бвгджзклмнпрстфхцчшщ])\1', r'\1', s)

        # оглушение:
        # СКОБКУ -> СКОПКУ
        n = len(s)
        new_s = []
        for c1, c2 in zip(s, s[1:]):
            if c2 in 'кпстфх':
                new_s.append(self.conson(c1))
            else:
                new_s.append(c1)

        # последнюю согласную оглушаем всегда:
        # ГОД -> ГОТ
        new_s.append(self.conson(s[-1]))

        s = ''.join(new_s)

        # огрушаем последнюю согласную с мягким знаком:
        # ВПРЕДЬ -> ВПРЕТЬ
        if s[-1] == 'ь' and s[-2] in 'бвгдз':
            s = s[:-2] + self.conson(s[-2]) + 'ь'

        if self.get_vowel_count(s) > 1:
            for ic, c in enumerate(s):
                if c in "уеыаоэёяию":
                    # нашли первую, ударную гласную
                    new_s = s[:ic+1]
                    for c2 in s[ic+1:]:
                        # безударные О меняем на А (потом надо бы ввести фонетический алфавит)
                        if c2 == 'о':
                            new_s += 'а'
                        else:
                            new_s += c2

                    s = new_s
                    break

            # ослабляем безударную Е
            # МУСУЛЬМАНЕ = МУСУЛЬМАНИ
            #if s[-1] == 'е':
            #    s = s[:-1] + 'и'

        return s

    def get_vowel_count(self, word0, abbrevs=True):
        word = self.sanitize_word(word0)
        vowels = "уеыаоэёяию"
        vowel_count = 0

        for ch in word:
            if ch in vowels:
                vowel_count += 1

        if vowel_count == 0 and len(word0) > 1 and abbrevs:
            # аббревиатуры из согласных: ГКЧП
            # Считаем, что там число гласных=длине: Гэ-Ка-Че-Пэ
            return len(word0)

        return vowel_count

    def predict_ambiguous_accent(self, word, ud_tags):
        best_accented = None
        best_matching = 0
        ud_tagset = set(ud_tags)
        for accented, tagsets in self.ambiguous_accents[word].items():
            for tagset in tagsets:
                tx = set(tagset.split('|'))
                nb_matched = len(ud_tagset.intersection(tx))
                if nb_matched > best_matching:
                    best_matching = nb_matched
                    best_accented = accented

        if best_accented is None:
            #print('ERROR@166 Could not choose accent for word="{}" ud_tags="{}"'.format(word, '|'.join(ud_tags)))
            #exit(0)
            return -1

        n_vowels = 0
        for c in best_accented:
            if c.lower() in 'уеыаоэёяию':
                n_vowels += 1
                if c.isupper():
                    return n_vowels

        raise NotImplementedError()

    def predict_stressed_charpos(self, word):
        """ Вернет индекс ударной буквы (это будет гласная, конечно). Отсчет от 0 """
        if word in self.word_accents_dict:
            vi = self.word_accents_dict[word]
            nv = 0
            for ic, c in enumerate(word):
                if c in "уеыаоэёяию":
                    nv += 1

                    if nv == vi:
                        return ic

        if re.match(r'^[бвгджзклмнпрстфхцчшщ]{2,}$', word):
            # Считаем, что в аббревиатурах, состоящих из одних согласных,
            # ударение падает на последний "слог":
            # ГКЧП -> Гэ-Ка-Че-П^э
            return len(word)

        i = self.stress_model.predict(word)
        return i

    def predict_stress(self, word):
        if word in self.predicted_accents:
            return self.predicted_accents[word]

        if re.match(r'^[бвгджзклмнпрстфхцчшщ]{2,}$', word):
            # Считаем, что в аббревиатурах, состоящих из одних согласных,
            # ударение падает на последний "слог":
            # ГКЧП -> Гэ-Ка-Че-П^э
            return len(word)

        #print('DEBUG@146 word={}'.format(word))
        i = self.stress_model.predict(word)
        #if len(i) != 1:
        #    return -1
        #i = i[0]

        # получили индекс символа с ударением.
        # нам надо посчитать гласные слева (включая ударную).
        nprev = self.get_vowel_count(word[:i])
        accent = nprev + 1
        self.predicted_accents[word] = accent
        return accent

    def get_accent0(self, word0, ud_tags=None):
        word = self.yoficate(self.sanitize_word(word0))
        if 'ё' in word:
            # считаем, что "ё" всегда ударная (исключение - слово "ёфикация" и однокоренные)
            n_vowels = 0
            for c in word0:
                if c in 'уеыаоэёяию':
                    n_vowels += 1
                    if c == 'ё':
                        return n_vowels

        if ud_tags and self.ambiguous_accents and word in self.ambiguous_accents:
            return self.predict_ambiguous_accent(word, ud_tags)

        if word in self.word_accents_dict:
            return self.word_accents_dict[word]

        return self.predict_stress(word)

    def get_accent(self, word0, ud_tags=None):
        word = self.sanitize_word(word0)
        word = self.yoficate(word)

        if 'ё' in word:
            # считаем, что "ё" всегда ударная (исключение - слово "ёфикация" и однокоренные)
            n_vowels = 0
            for c in word:
                if c in 'уеыаоэёяию':
                    n_vowels += 1
                    if c == 'ё':
                        return n_vowels

        if ud_tags and self.ambiguous_accents and word in self.ambiguous_accents:
            return self.predict_ambiguous_accent(word, ud_tags)

        if word in self.word_accents_dict:
            return self.word_accents_dict[word]

        # Некоторые грамматические формы в русском языке имеют
        # фиксированное ударение.
        pos1 = word.find(u'ейш') # сильнейший, наимудрейшие
        if pos1 != -1:
            stress_pos = self.get_vowel_count(word[:pos1]) + 1
            return stress_pos

        # Есть продуктивные приставки типа АНТИ или НЕ
        for prefix in u'анти не недо прото'.split():
            if word.startswith(prefix):
                word1 = word[len(prefix):]
                if len(word1) > 2:
                    if word1 in self.word_accents_dict:
                        return self.word_accents_dict[word1]

        # Иногда можно взять ударение из стема: "ПОЗИТРОННЫЙ" -> "ПОЗИТРОН"
        if False:
            stem = self.stemmer.stem(word)
            if stem in self.word_accents_dict:
                return self.word_accents_dict[stem]

        if True:
            return self.predict_stress(word)

        vowel_count = self.get_vowel_count(word)
        return (vowel_count + 1) // 2

    def get_phoneme(self, word):
        word = self.sanitize_word(word)

        word_end = word[-3:]
        vowel_count = self.get_vowel_count(word)
        accent = self.get_accent(word)

        return word_end, vowel_count, accent

    def render_accenture(self, word):
        accent = self.get_accent(word)

        accenture = []
        n_vowels = 0
        stress_found = False
        for c in word:
            s = None
            if c in 'уеыаоэёяию':
                n_vowels += 1
                s = '-'

            if n_vowels == accent and not stress_found:
                s = '^'
                stress_found = True

            if s:
                accenture.append(s)

        return ''.join(accenture)

    def do_endings_match(self, word1, vowels1, accent1, word2):
        if len(word1) >= 3 and len(word2) >= 3:
            # Если ударный последний слог, то проверим совпадение этого слога
            if accent1 == vowels1:
                # TODO - надо проверять не весь слог, а буквы, начиная с гласной
                # ...
                syllabs1 = rusyllab.split_word(word1)
                syllabs2 = rusyllab.split_word(word2)
                return syllabs1[-1] == syllabs2[-1]
            else:
                # В остальных случаях - проверим совместимость последних 3х букв
                end1 = word1[-3:]
                end2 = word2[-3:]

                # БЕДНА == ГРУСТНА
                if re.match(r'[бвгджзклмнпрстфхцчшщ]на', end1) and re.match(r'[бвгджзклмнпрстфхцчшщ]на', end2):
                    return True

                if re.match(r'[бвгджзклмнпрстфхцчшщ][ая]я', end1) and re.match(r'[бвгджзклмнпрстфхцчшщ][ая]я', end2):
                    return True

                if re.match(r'[бвгджзклмнпрстфхцчшщ][ую]ю', end1) and re.match(r'[бвгджзклмнпрстфхцчшщ][ую]ю', end2):
                    return True

                return end1 == end2

        return False


def get_stressed_vowel(word, stress):
    v_counter = 0
    for c in word:
        if c in "уеыаоэёяию":
            v_counter += 1
            if v_counter == stress:
                return c

    return None


def get_stressed_syllab(syllabs, stress):
    v_counter = 0
    for syllab in syllabs:
        for c in syllab:
            if c in "уеыаоэёяию":
                v_counter += 1
                if v_counter == stress:
                    return syllab

    return None


def are_rhymed_syllables(syllab1, syllab2):
    # Проверяем совпадение последних букв слога, начиная с гласной
    r1 = re.match(r'^.+([уеыаоэёяию].*)$', syllab1)
    r2 = re.match(r'^.+([уеыаоэёяию].*)$', syllab2)
    if r1 and r2:
        # это последние буквы слога с гласной.
        s1 = r1.group(1)
        s2 = r2.group(1)

        # при проверке соответствия надо учесть фонетическую совместимость гласных (vowel2base)
        return are_phonetically_equal(s1, s2)

    return False


def extract_ending_vc(s):
    # вернет последние буквы слова, среди которых минимум 1 гласная и 1 согласная

    # МАМА
    #   ^^
    r = re.search('([бвгджзйклмнпрстфхцчшщ][уеыаоэёяию]+)$', s)
    if r:
        return r.group(1)

    # СТОЛБ
    #   ^^^
    # СТОЙ
    #   ^^
    r = re.search('([уеыаоэёяию][бвгджзйклмнпрстфхцчшщ]+)$', s)
    if r:
        return r.group(1)

    # КРОВЬ
    #   ^^^
    r = re.search('([уеыаоэёяию][бвгджзйклмнпрстфхцчшщ]+ь)$', s)
    if r:
        return r.group(1)


    # ЛАДЬЯ
    #   ^^^
    r = re.search('([бвгджзйклмнпрстфхцчшщ]ь[уеыаоэёяию]+)$', s)
    if r:
        return r.group(1)

    return ''


vowel2base = {'я': 'а', 'ю': 'у', 'е': 'э'}
vowel2base0 = {'я': 'а', 'ю': 'у'}


def are_phonetically_equal(s1, s2):
    # Проверяем фонетическую эквивалентность двух строк, учитывая пары гласных типа А-Я etc
    # Каждая из строк содержит часть слова, начиная с ударной гласной (или с согласной перед ней).
    if len(s1) == len(s2):
        if s1 == s2:
            return True

        vowels = "уеыаоэёяию"
        total_vowvels1 = sum((c in vowels) for c in s1)

        n_vowel = 0
        for ic, (c1, c2) in enumerate(zip(s1, s2)):
            if c1 in vowels:
                n_vowel += 1
                if n_vowel == 1:
                    # УДАРНАЯ ГЛАСНАЯ
                    if total_vowvels1 == 1 and ic == len(s1)-1:
                        # ОТЕЛЯ <==> ДАЛА
                        if c1 != c2:
                            return False
                    else:
                        cc1 = vowel2base0.get(c1, c1)
                        cc2 = vowel2base0.get(c2, c2)
                        if cc1 != cc2:
                            return False

                        tail1 = s1[ic+1:]
                        tail2 = s2[ic+1:]
                        if tail1 in ('жной', 'жный', 'жнай') and tail2 in ('жной', 'жный', 'жнай'):
                            return True
                else:
                    cc1 = vowel2base.get(c1, c1)
                    cc2 = vowel2base.get(c2, c2)
                    if cc1 != cc2:
                        return False
            elif c1 != c2:
                return False

        return True

    return False


def extract_ending_prononciation_after_stress(accents, word, stress):
    ending = None
    v_counter = 0
    for i, c in enumerate(word):
        if c in "уеыаоэёяию":
            v_counter += 1
            if v_counter == stress:
                if i == len(word) - 1:
                    # Ударная гласная в конце слова, берем последние 2 или 3 буквы
                    # ГУБА
                    #   ^^
                    ending = extract_ending_vc(word)
                else:
                    ending = word[i:]

                if len(ending) < len(word):
                    c2 = word[-len(ending)-1]
                    if c2 in 'цшщ' and ending[0] == 'и':
                        # меняем ЦИ -> ЦЫ
                        ending = 'ы' + ending[1:]

                break

    if not ending:
        # print('ERROR@385 word1={} stress1={}'.format(word1, stress1))
        return ''

    ending = accents.pronounce(ending)
    if ending.startswith('ё'):
        ending = 'о' + ending[1:]

    return ending


def rhymed(accents, word1, ud_tags1, word2, ud_tags2):
    word1 = accents.yoficate(accents.sanitize_word(word1))
    word2 = accents.yoficate(accents.sanitize_word(word2))

    if (word1.lower(), word2.lower()) in accents.rhymed_words or (word2.lower(), word1.lower()) in accents.rhymed_words:
        return True

    stress1 = accents.get_accent(word1, ud_tags1)
    vow_count1 = accents.get_vowel_count(word1)
    pos1 = vow_count1 - stress1

    stress2 = accents.get_accent(word2, ud_tags2)
    vow_count2 = accents.get_vowel_count(word2)
    pos2 = vow_count2 - stress2

    # смещение ударной гласной от конца слова должно быть одно и то же
    # для проверяемых слов.
    if pos1 == pos2:
        # Теперь все буквы, начиная с ударной гласной
        ending1 = extract_ending_prononciation_after_stress(accents, word1, stress1)
        ending2 = extract_ending_prononciation_after_stress(accents, word2, stress2)

        return are_phonetically_equal(ending1, ending2)

    return False


if __name__ == '__main__':
    data_folder = '../../data/poetry/dict'
    tmp_dir = '../../tmp'

    e = extract_ending_vc('мама')
    assert(e == 'ма')

    e = extract_ending_vc('стой')
    assert(e == 'ой')

    e = extract_ending_vc('ставь')
    assert(e == 'авь')

    e = extract_ending_vc('столб')
    assert(e == 'олб')

    e = extract_ending_vc('эпидемии')
    assert(e == 'мии')

    e = extract_ending_vc('профанации')
    assert(e == 'ции')

    e = extract_ending_vc('марок')
    assert(e == 'ок')

    e = extract_ending_vc('марки')
    assert(e == 'ки')

    r = are_phonetically_equal('ов', 'во')
    assert(r is False)

    accents = Accents()

    # ТЕСТЫ ТРАНСКРИПТОРА
    s = accents.pronounce('муж')
    assert(s == 'муш')

    s = accents.pronounce('сильно')
    assert(s == 'сильна')

    s = accents.pronounce('впредь')
    assert(s == 'фпреть')

    s = accents.pronounce('рожь')
    assert(s == 'рош')

    s = accents.pronounce('мышь')
    assert(s == 'мыш')

    s = accents.pronounce('солнце')
    assert(s == 'сонцэ')

    s = accents.pronounce('счастье')
    assert(s == 'щастье')

    s = accents.pronounce('браться')
    assert(s == 'браца')

    s = accents.pronounce('братца')
    assert(s == 'браца')

    s = accents.pronounce('жир')
    assert(s == 'жыр')

    s = accents.pronounce('шип')
    assert(s == 'шып')

    s = accents.pronounce('жёстко')
    assert(s == 'жостка')

    #s = accents.pronounce('суббота')
    #assert(s == 'субота')

    s = accents.pronounce('скобку')
    assert(s == 'скопку')

    s = accents.pronounce('сказки')
    assert(s == 'скаски')

    s = accents.pronounce('ковка')
    assert(s == 'кофка')

    s = accents.pronounce('загса')
    assert(s == 'закса')

    s = accents.pronounce('гадка')
    assert(s == 'гатка')

    s = accents.pronounce('год')
    assert(s == 'гот')

    s = accents.pronounce('мама')
    assert(s == 'мама')

    s = accents.pronounce('монстр')
    assert(s == 'монстр')

    #s = accents.pronounce('моцион')
    #assert(s == 'моцыон')

    #s = accents.pronounce('мусульмане')
    #assert(s == 'мусульмани')

    if True:
        # Делаем бинарный файл с данными для работы акцентуатора.
        accents.load(data_folder, None)
        accents.save_pickle(os.path.join(tmp_dir, 'accents.pkl'))

    # =======================================
    # ТЕСТЫ СОХРАНЕННОГО СЛОВАРЯ
    # =======================================

    accents.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
    accents.after_loading(stress_model_dir='../../tmp/stress_model')

    i = accents.get_accent('голубое')
    assert(i == 3)

    i = accents.get_accent('детям')
    assert(i == 1)

    #i = accents.get_accent('сочнейшего')
    #assert(i == 2)

    i = accents.get_accent('землей')
    assert(i == 2)

    i = accents.get_accent('дождей')
    assert(i == 2)

    i = accents.get_accent('конек')
    assert(i == 2)

    i = accents.get_accent('остаётся')
    assert(i == 3)

    i = accents.get_accent('остается')
    assert(i == 3)

    i = accents.get_accent('годы')
    assert(i == 1)

    i = accents.get_accent('кин')
    assert(i == 1)

    i = accents.get_accent('кино')
    assert(i == 2)

    i = accents.get_accent('сын')
    assert(i == 1)

    i = accents.get_accent('груди', ['Case=Loc'])
    assert(i == 2)

    # ======================================================================================
    # Проверяем процедуру проверки рифмованности двух слов с учетом морфологических тегов
    # ======================================================================================

    r = rhymed(accents, 'Европа', [], 'поклёпа', [])
    assert(r is True)

    r = rhymed(accents, 'берёзу', [], 'прозу', [])
    assert(r is True)

    r = rhymed(accents, 'отеля', [], 'дала', [])
    assert(r is False)

    r = rhymed(accents, 'бороду', [], 'сторону', [])
    assert(r is False)

    r = rhymed(accents, 'землей', [], 'коней', [])
    assert(r is False)

    r = rhymed(accents, 'сильно', [], 'обильна', [])
    assert(r is True)

    r = rhymed(accents, 'впредь', [], 'корпеть', [])
    assert(r is True)

    r = rhymed(accents, 'является', [], 'остается', [])
    assert(r is False)

    r = rhymed(accents, 'любой', 'ADJ|Case=Gen', 'русской', 'ADJ|Case=Gen')
    assert(r is False)

    r = rhymed(accents, 'браться', [], 'братца', [])
    assert(r is True)

    r = rhymed(accents, 'сказки', [], 'хаски', [])
    assert(r is True)

    r = rhymed(accents, 'загса', [], 'бакса', [])
    assert(r is True)

    r = rhymed(accents, 'догадка', [], 'ватка', [])
    assert(r is True)

    r = rhymed(accents, 'тогда', [], 'губа', [])
    assert(r is False)

    r = rhymed(accents, 'на', [], 'губа', [])
    assert(r is False)

    r = rhymed(accents, 'лозанья', [], 'баранья', [])
    assert(r is True)

    r = rhymed(accents, 'аттракцион', [], 'моцион', [])
    assert(r is True)

    r = rhymed(accents, 'эпидемиями', [], 'котлеточками', [])
    assert(r is False)

    r = rhymed(accents, 'деточки', [], 'конфеточки', [])
    assert(r is True)

    r = rhymed(accents, 'кровь', [], 'морковь', [])
    assert(r is True)

    r = rhymed(accents, 'катька', [], 'лопатка', [])
    assert(r is False)

    r = rhymed(accents, 'века', 'Case=Gen'.split('|'), 'аптека', [])
    assert(r is True)

    r = rhymed(accents, 'века', 'Case=Nom'.split('|'), 'аптека', [])
    assert(r is False)

    r = rhymed(accents, 'коровы', [], 'совы', 'Case=Nom'.split('|'))
    assert(r is True)

    r = rhymed(accents, 'побеседуем', [], 'дуем', [])
    assert(r is False)

    r = rhymed(accents, 'кури', [], 'куры', [])
    assert(r is False)

    r = rhymed(accents, 'зеркала', 'NOUN|Case=Nom'.split('|'), 'начала', 'VERB|Gender=Fem'.split('|'))
    assert(r is True)

    r = rhymed(accents, 'салфетка', 'NOUN|Case=Nom'.split('|'), 'клетка', 'NOUN|Case=Nom'.split('|'))
    assert(r is True)

    r = rhymed(accents, 'горе', 'NOUN|Case=Dat'.split('|'), 'январе', [])
    assert(r is True)

    r = rhymed(accents, 'горе', 'NOUN|Case=Nom'.split('|'), 'январе', [])
    assert(r is False)

    r = rhymed(accents, 'боров', 'Case=Nom'.split('|'), 'кров', [])
    assert(r is False)

    r = rhymed(accents, 'коров', [], 'кров', [])
    assert(r is True)

    r = rhymed(accents, 'впереди', [], 'детсады', [])
    assert(r is False)

    r = rhymed(accents, 'кляп', [], 'пап', [])
    assert(r is True)

    r = rhymed(accents, 'кляп', [], 'ляпа', [])
    assert(r is False)

    r = rhymed(accents, 'смелее', [], 'большое', [])
    assert(r is False)

    r = rhymed(accents, 'смелее', [], 'сильнее', [])
    assert(r is True)

    r = rhymed(accents, 'сильная', [], 'синильная', [])
    assert(r is True)

    r = rhymed(accents, 'умильная', [], 'сдельная', [])
    assert(r is False)

    r = rhymed(accents, 'груза', [], 'союза', [])
    assert(r is True)

    r = rhymed(accents, 'снежной', [], 'нежный', [])
    assert(r is True)

    # Проверяем вспомогательную процедуру определения ударения для случаев, когда
    # нучно учитывать морфологические признаки слова, чтобы снять неоднозначность.
    i1 = accents.predict_ambiguous_accent('воды', 'NOUN|Case=Nom'.split('|'))
    assert(i1 == 1)
    i2 = accents.predict_ambiguous_accent('воды', 'NOUN|Case=Gen'.split('|'))
    assert(i2 == 2)

    print('-'*40)
    for w in 'голубое кошка собака яма катапультами'.split():
        ax = accents.render_accenture(w)
        print('{} --> {}'.format(w, ax))

    print('-'*40)

    for word in 'борода землей проливного коней'.split():
        a = accents.get_accent(word)
        print(a)
        s = ''
        n_vowels = 0
        stress_found = False
        for c in word:
            if c in 'уеыаоэёяию':
                n_vowels += 1
            if n_vowels == a and not stress_found:
                s += '^'
                stress_found = True
            s += c
        print(s)
