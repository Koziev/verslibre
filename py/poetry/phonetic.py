"""
Фонетический словарь (ударятор).
При выполнении словарь будет загружен из текстовых файлов, подготовлен и сохранен в pickle-файле
для последующей быстрой подгрузки.

19.05.2022 Добавлено автоисправление некоторых орфографических ошибок типа "тишына", "стесняцца"
07.06.2022 Добавлено автоисправление децкий ==> детский
02.08.2022 Исправление опечатки - твердый знак вместо мягкого "пъянки"
11.08.2022 Добавлена продуктивная приставка "супер-"
12.08.2022 Добавлена продуктивная приставка "лже-"
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
        self.ambiguous_accents2 = None
        self.word_accents_dict = None
        self.yo_words = None
        self.rhymed_words = set()
        self.allow_rifmovnik = False

    def sanitize_word(self, word):
        return word.lower() #.replace(u'ё', u'е')

    def load(self, data_dir, all_words):
        # Рифмовник для нечеткой рифмы
        with open(os.path.join(data_dir, 'rifmovnik.small.upgraded.json'), 'r') as f:
            rhyming_data = json.load(f)
            self.rhyming_dict = dict((key, values) for key, values in rhyming_data['dictionary'].items() if len(values) > 0)

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
                word = line.strip().lower()
                key = word.replace('ё', 'е')
                self.yo_words[key] = word

        # Информация о словах, которые для разных грамматических форм могут давать разное ударение.
        path = os.path.join(data_dir, 'ambiguous_accents.yaml')
        logging.info('Loading ambiguous accents information from "%s"', path)
        d = yaml.safe_load(io.open(path, 'r', encoding='utf-8').read())

        # 03.08.2022
        # В словаре встречаются вхождения "Case=Every"
        # Раскроем, чтобы было явное перечисление всех падежей.
        d2 = dict()
        for entry_name, entry_data in d.items():
            entry_data2 = dict()
            for form, tagsets in entry_data.items():
                tagsets2 = []
                for tagset in tagsets:
                    if 'Case=Every' in tagset:
                        for case in ['Nom', 'Gen', 'Ins', 'Acc', 'Dat', 'Loc']:
                            tagset2 = tagset.replace('Case=Every', 'Case={}'.format(case))
                            tagsets2.append(tagset2)
                    else:
                        tagsets2.append(tagset)

                entry_data2[form] = tagsets2

            d2[entry_name] = entry_data2

        self.ambiguous_accents = d2

        # 14.02.2022 сделаем проверку содержимого, чтобы не словить ошибку в рантайме.
        for word, wdata in self.ambiguous_accents.items():
            for stressed_form, tagsets in wdata.items():
                if not any((c in 'АЕЁИОУЫЭЮЯ') for c in stressed_form):
                    print('Missing stressed vowel in "ambiguous_accents.yaml" for word={}'.format(word))
                    exit(0)

        logging.info('%d items in ambiguous_accents', len(self.ambiguous_accents))

        # Некоторые слова допускают разное ударение для одной грамматической формы: пОнял-понЯл
        self.ambiguous_accents2 = yaml.safe_load(io.open(os.path.join(data_dir, 'ambiguous_accents_2.yaml'), 'r', encoding='utf-8').read())

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
                                    self.word_accents_dict[word.lower()] = stress
                                    break

        if True:
            path2 = os.path.join(data_dir, 'accents.txt')
            logging.info('Loading stress information from "%s"', path2)
            with codecs.open(path2, 'r', 'utf-8') as rdr:
                for line in rdr:
                    tx = line.strip().split('#')
                    if len(tx) == 2:
                        forms = tx[1].split(',')
                        for form in forms:
                            word = self.sanitize_word(form.replace('\'', '').replace('`', ''))
                            if all_words is None or word in all_words:
                                if '\'' in form:
                                    accent_pos = form.index('\'')
                                    nb_vowels_before = self.get_vowel_count(form[:accent_pos], abbrevs=False)
                                    if word not in self.word_accents_dict:
                                        self.word_accents_dict[word] = nb_vowels_before
                                elif 'ё' in form:
                                    accent_pos = form.index('ё')
                                    nb_vowels_before = self.get_vowel_count(form[:accent_pos], abbrevs=False)+1
                                    if word not in self.word_accents_dict:
                                        self.word_accents_dict[word] = nb_vowels_before

        if True:
            stress_char = '́'
            stress2_char = '̀'
            p3 = os.path.join(data_dir, 'ruwiktionary-accents.txt')
            logging.info('Loading stress information from "%s"', p3)
            with codecs.open(p3, 'r', 'utf-8') as rdr:
                for iline, line in enumerate(rdr):
                    word = line.strip()
                    if '-' not in word:
                        nword = word.replace(stress_char, '').replace('\'', '').replace('ѝ', 'и').replace('ѐ', 'е').replace(stress2_char, '').lower()
                        if len(nword) > 2:
                            if stress_char in word:
                                accent_pos = word.index(stress_char)
                                nb_vowels_before = self.get_vowel_count(word[:accent_pos], abbrevs=False)
                                if nword not in self.word_accents_dict:
                                    self.word_accents_dict[nword] = nb_vowels_before
                            elif '\'' in word:
                                accent_pos = word.index('\'')
                                nb_vowels_before = self.get_vowel_count(word[:accent_pos], abbrevs=False)
                                if nword not in self.word_accents_dict:
                                    self.word_accents_dict[nword] = nb_vowels_before
                            elif 'ё' in word:
                                accent_pos = word.index('ё')
                                nb_vowels_before = self.get_vowel_count(word[:accent_pos], abbrevs=False)
                                stress_pos = nb_vowels_before + 1
                                if nword not in self.word_accents_dict:
                                    self.word_accents_dict[nword] = stress_pos

        if True:
            path = os.path.join(data_dir, 'words_accent.json')
            logging.info('Loading stress information from "%s"', path)
            d = json.loads(open(path).read())
            for word, a in d.items():
                if '-' not in word:
                    nword = self.sanitize_word(word)
                    if nword not in self.word_accents_dict:
                        self.word_accents_dict[nword] = a

        true_accent_entries = dict()
        with io.open(os.path.join(data_dir, 'true_accents.txt'), 'r', encoding='utf-8') as rdr:
            for line in rdr:
                word = line.strip()
                if word:
                    nword = self.sanitize_word(word)
                    if nword in self.ambiguous_accents:
                        del self.ambiguous_accents[nword]
                    m = re.search('([АЕЁИОУЭЮЯЫ])', word)
                    if m is None:
                        logging.error('Invalid item "%s" in "true_accents.txt"', word)
                        exit(0)

                    accent_char = m.groups(0)[0]
                    accent_pos = word.index(accent_char)
                    nb_vowels_before = self.get_vowel_count(word[:accent_pos], abbrevs=False) + 1

                    # Детектируем переопеределение ударения в слове. Такие слова с неоднозначным ударением
                    # надо переносить в ambiguous_accents_2.yaml
                    if nword in true_accent_entries and true_accent_entries[nword] != word:
                        logging.error('Controversial redefenition of stress position for word "%s" in "true_accents.txt": %s and %s', nword, true_accent_entries[nword], word)
                        exit(0)

                    self.word_accents_dict[nword] = nb_vowels_before
                    true_accent_entries[nword] = word

        logging.info('%d items in word_accents_dict', len(self.word_accents_dict))

        if False:
            # Делаем отладочный листинг загруженных ударений
            with codecs.open('../tmp/accents.txt', 'w', 'utf-8') as wrt:
                for word, accent in sorted(six.iteritems(self.word_accents_dict), key=operator.itemgetter(0)):
                    wrt.write(u'{}\t{}\n'.format(word, accent))

    def save_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.ambiguous_accents, f)
            pickle.dump(self.ambiguous_accents2, f)
            pickle.dump(self.word_accents_dict, f)
            pickle.dump(self.yo_words, f)
            pickle.dump(self.rhymed_words, f)
            pickle.dump(self.rhyming_dict, f)

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            self.ambiguous_accents = pickle.load(f)
            self.ambiguous_accents2 = pickle.load(f)
            self.word_accents_dict = pickle.load(f)
            self.yo_words = pickle.load(f)
            self.rhymed_words = pickle.load(f)
            self.rhyming_dict = pickle.load(f)

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
        if s is None or len(s) == 0:
            return ''

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

        # БЕЗБРАЧЬЯ
        if 'чь' in s:
            s = s.replace('чья', 'ча')
            s = s.replace('чье', 'чэ')
            s = s.replace('чьё', 'чо')
            s = s.replace('чью', 'чё')

        # двойные согласные:
        # СУББОТА -> СУБОТА
        s = re.sub(r'([бвгджзклмнпрстфхцчшщ])\1', r'\1', s)

        # оглушение:
        # СКОБКУ -> СКОПКУ
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
        if len(s) >= 2 and s[-1] == 'ь' and s[-2] in 'бвгдз':
            s = s[:-2] + self.conson(s[-2]) + 'ь'

        if self.get_vowel_count(s, abbrevs=False) > 1:
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
        vowels = "уеыаоэёяиюaeoy" # "уеыаоэёяию"   28.12.2021 добавил гласные из латиницы
        vowel_count = 0

        for ch in word:
            if ch in vowels:
                vowel_count += 1

        if vowel_count == 0 and len(word0) > 1 and abbrevs:
            # аббревиатуры из согласных: ГКЧП
            # Считаем, что там число гласных=длине: Гэ-Ка-Че-Пэ
            return len(word0)

        return vowel_count

    def is_oov(self, word):
        return 'ё' not in word and word not in self.word_accents_dict and word not in self.ambiguous_accents and word not in self.ambiguous_accents2

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

        msg = 'Could not predict stress position in word="{}" tags="{}"'.format(word, ' '.join(ud_tags) if ud_tags else '[]')
        raise ValueError(msg)

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
        nprev = self.get_vowel_count(word[:i], abbrevs=False)
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

        vowel_count = self.get_vowel_count(word)
        if vowel_count == 1:
            # Для слов, содержащих единственную гласную, сразу возвращаем позицию ударения на этой гласной
            return 1

        if 'ё' in word:
            if word in self.word_accents_dict:
                return self.word_accents_dict[word]

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

        # 19.05.2022 в порошках и т.п. по законам жанра допускаются намеренные ошибки типа "ошыбка".
        # Попробуем скорректировать такие ошибки.
        corrections = [('тьса', 'тся'),  # рвЕтьса
                       ('тьса', 'ться'),  # скрытьса
                       ('ться', 'тся'),
                       ('юцца', 'ются'),
                       ('цца', 'ться'),
                       ('юца', 'ются'),  # бьюца
                       ('шы', 'ши'), ('жы', 'жи'), ('цы', 'ци'), ('щю', 'щу'), ('чю', 'чу'),
                       ('ща', 'сча'),
                       ('щя', 'ща'),  # щями
                       ("чя", "ча"),  # чящя
                       ("жэ", "же"),  # художэственный
                       ('цэ', 'це'), ('жо', 'жё'), ('шо', 'шё'), ('чо', 'чё'), ('́чьк', 'чк'),
                       ('що', 'щё'),  # вощоный
                       ('щьк', 'щк'),
                       ('цк', 'тск'),  # 07.06.2022 децкий ==> детский
                       ('цца', 'тся'),  # 04.08.2022 "льюцца"
                       ('ъе', 'ьё'),  # бъется
                       ('ье', 'ъе'),  # сьЕли
                       ('сн', 'стн'),  # грусный
                       ('цц', 'тц'),  # браццы
                       ('цц', 'дц'),  # триццать
                       ('чт', 'чьт'),  # прячте
                       ('тьн', 'тн'),  # плОтьник
                       ('зд', 'сд'),  # здачу
                       ('тса', 'тся'),  # гнУтса
                       ]

        for m2 in corrections:
            if m2[0] in word:
                word2 = word.replace(m2[0], m2[1])
                if word2 in self.word_accents_dict:
                    return self.word_accents_dict[word2]

        # восстанавливаем мягкий знак в "стоиш" "сможеш"  "сбереч"
        # встретимса
        #        ^^^
        e_corrections = [('иш', 'ишь'),  # стоиш
                         ('еш', 'ешь'),  # сможеш
                         ('еч', 'ечь'),  # сбереч
                         ('мса', 'мся'),  # встретимса
                         ]
        for e1, e2 in e_corrections:
            if word.endswith(e1):
                word2 = word[:-len(e1)] + e2
                if word2 in self.word_accents_dict:
                    return self.word_accents_dict[word2]



        # убираем финальный "ь" после шипящих:
        # клавишь
        if re.search(r'[чшщ]ь$', word):
            word2 = word[:-1]
            if word2 in self.word_accents_dict:
                return self.word_accents_dict[word2]

        # повтор согласных сокращаем до одной согласной:
        # щщупать
        if len(word) > 1:
            cn = re.search(r'(.)\1', word, flags=re.I)
            if cn:
                c1 = cn.group(1)[0]
                word2 = re.sub(c1+'{2,}', c1, word, flags=re.I)
                if word2 in self.word_accents_dict:
                    return self.word_accents_dict[word2]

        # Некоторые грамматические формы в русском языке имеют
        # фиксированное ударение.
        pos1 = word.find('ейш') # сильнейший, наимудрейшие
        if pos1 != -1:
            stress_pos = self.get_vowel_count(word[:pos1], abbrevs=False) + 1
            return stress_pos

        # Есть продуктивные приставки типа АНТИ или НЕ
        for prefix in 'спец сверх недо анти полу электро магнито не прото микро макро нано квази само слабо одно двух трех четырех пяти шести семи восьми девяти десяти одиннадцати двенадцати тринадцати четырнадцати пятнадцати шестнадцати семнадцати восемнадцати девятнадцати двадцати тридцами сорока пятидесяти шестидесяти семидесяти восьмидесяти девяносто сто тысяче супер лже мета'.split():
            if word.startswith(prefix):
                word1 = word[len(prefix):]  # отсекаем приставку
                if len(word1) > 2:  # нас интересуют составные слова
                    if word1 in self.word_accents_dict:
                        return self.get_vowel_count(prefix, abbrevs=False) + self.word_accents_dict[word1]

        # Иногда можно взять ударение из стема: "ПОЗИТРОННЫЙ" -> "ПОЗИТРОН"
        if False:
            stem = self.stemmer.stem(word)
            if stem in self.word_accents_dict:
                return self.word_accents_dict[stem]

        if vowel_count == 0:
            # знаки препинания и т.д., в которых нет ни одной гласной.
            return -1

        # 02.08.2022 Исправление опечатки - твердый знак вместо мягкого "пъянки"
        if 'ъ' in word:
            word1 = word.replace('ъ', 'ь')
            if word1 in self.word_accents_dict:
                return self.word_accents_dict[word1]

        if True:
            return self.predict_stress(word)

        return (vowel_count + 1) // 2

    def get_phoneme(self, word):
        word = self.sanitize_word(word)

        word_end = word[-3:]
        vowel_count = self.get_vowel_count(word, abbrevs=False)
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

    # мягкий знак и после него йотированная гласная:
    #
    # семья
    #    ^^
    if re.search(r'ь[ёеюя]$', s):
        return s[-1]

    # гласная и следом - йотированная гласная:
    #
    # моя
    #  ^^
    if re.search(r'[аеёиоуыэюя][ёеюя]$', s):
        return s[-1]

    # неглиже
    #      ^^
    r = re.search('([жшщ])е$', s)
    if r:
        return r.group(1) + 'э'

    # хороши
    #     ^^
    r = re.search('([жшщ])и$', s)
    if r:
        return r.group(1) + 'ы'

    # иногда встречается в пирожках неорфографичная форма:
    # щя
    # ^^
    r = re.search('([жшщ])я$', s)
    if r:
        return r.group(1) + 'а'

    # иногда встречается в пирожках неорфографичная форма:
    # трепещю
    #      ^^
    r = re.search('([жшщ])ю$', s)
    if r:
        return r.group(1) + 'ю'

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


def transcript_unstressed(chars):
    if chars is None or len(chars) == 0:
        return ''

    phonems = []
    for c in chars:
        if c == 'о':
            phonems.append('а')
        elif c == 'и':
            phonems.append('ы')
        elif c == 'ю':
            phonems.append('у')
        elif c == 'я':
            phonems.append('а')
        elif c == 'ё':
            phonems.append('о')
        elif c == 'е':
            phonems.append('э')
        else:
            phonems.append(c)

    if phonems[-1] == 'ж':
        phonems[-1] = 'ш'
    if phonems[-1] == 'в':
        phonems[-1] = 'ф'
    elif phonems[-1] == 'б':
        # оглушение частицы "б"
        # не бу́дь у ба́нь и у кофе́ен
        # пиаротде́лов и прессслу́жб    <=====
        # мы все б завши́вели и пи́ли
        # из лу́ж б                    <=====
        phonems[-1] = 'п'

    res = ''.join(phonems)
    return res




def extract_ending_prononciation_after_stress(accents, word, stress, ud_tags, unstressed_prefix, unstressed_tail):
    unstressed_prefix_transcription = accents.pronounce(unstressed_prefix)  # transcript_unstressed(unstressed_prefix)
    unstressed_tail_transcription = accents.pronounce(unstressed_tail)  #transcript_unstressed(unstressed_tail)

    if len(word) == 1:
        return unstressed_prefix_transcription + word + unstressed_tail_transcription

    ending = None
    v_counter = 0
    for i, c in enumerate(word.lower()): # 25.06.2022 приводим к нижнему регистру
        if c in "уеыаоэёяию":
            v_counter += 1
            if v_counter == stress:
                if i == len(word) - 1 and len(unstressed_tail) == 0:
                    # Ударная гласная в конце слова, берем последние 2 или 3 буквы
                    # ГУБА
                    #   ^^
                    ending = extract_ending_vc(word)

                    # 01.02.2022 неударная "о" перед ударной гласной превращается в "а":  своя ==> сваЯ
                    if len(ending) >= 2 and ending[-2] == 'о' and ending[-1] in 'аеёиоуыэюя':
                        ending = ending[:-2] + 'а' + ending[-1]

                else:
                    ending = word[i:]
                    if ud_tags is not None and ('ADJ' in ud_tags or 'DET' in ud_tags) and ending == 'ого':
                        # Меняем "люб-ОГО" на "люб-ОВО"
                        ending = 'ово'

                    if ending.startswith('е'):
                        # 01.02.2022 меняем ударную "е" на "э": летом==>л'Этом
                        ending = 'э' + ending[1:]
                    elif ending.startswith('я'):
                        # 01.02.2022 меняем ударную "я" на "а": мячик==>м'Ачик
                        ending = 'а' + ending[1:]
                    elif ending.startswith('ё'):
                        # 01.02.2022 меняем ударную "ё" на "о": мёдом==>м'Одом
                        ending = 'о' + ending[1:]
                    elif ending.startswith('ю'):
                        # 01.02.2022 меняем ударную "ю" на "у": люся==>л'Уся
                        ending = 'у' + ending[1:]
                    elif ending.startswith('и'):
                        # 01.02.2022 меняем ударную "и" на "ы": сливы==>сл'Ывы, живы==>жЫвы
                        ending = 'ы' + ending[1:]

                if len(ending) < len(word):
                    c2 = word[-len(ending)-1]
                    if c2 in 'цшщ' and ending[0] == 'и':
                        # меняем ЦИ -> ЦЫ
                        ending = 'ы' + ending[1:]

                # if ending.endswith('ь'):  # убираем финальный мягкий знак: "ВОЗЬМЁШЬ"
                #     ending = ending[:-1]
                #
                # if ending.endswith('д'):  # оглушаем последнюю "д": ВЗГЛЯД
                #     ending = ending[:-1] + 'т'
                # elif ending.endswith('ж'):  # оглушаем последнюю "ж": ЁЖ
                #     ending = ending[:-1] + 'ш'
                # elif ending.endswith('з'):  # оглушаем последнюю "з": МОРОЗ
                #     ending = ending[:-1] + 'с'
                # #elif ending.endswith('г'):  # оглушаем последнюю "г": БОГ
                # #    ending = ending[:-1] + 'х'
                # elif ending.endswith('б'):  # оглушаем последнюю "б": ГРОБ
                #     ending = ending[:-1] + 'п'
                # elif ending.endswith('в'):  # оглушаем последнюю "в": КРОВ
                #     ending = ending[:-1] + 'ф'

                break

    if not ending:
        # print('ERROR@385 word1={} stress1={}'.format(word1, stress1))
        return ''

    ending = accents.pronounce(ending)
    if ending.startswith('ё'):
        ending = 'о' + ending[1:]

    return unstressed_prefix_transcription + ending + unstressed_tail_transcription


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
        # 28.06.2022 особо рассматриваем случай рифмовки с местоимением "я": друзья-я
        if word2 == 'я':
            return word1.endswith('я')

        # Теперь все буквы, начиная с ударной гласной
        ending1 = extract_ending_prononciation_after_stress(accents, word1, stress1, ud_tags1, '', '')
        ending2 = extract_ending_prononciation_after_stress(accents, word2, stress2, ud_tags2, '', '')

        return are_phonetically_equal(ending1, ending2)

    return False


def rhymed2(accentuator, word1, stress1, ud_tags1, unstressed_prefix1, unstressed_tail1, word2, stress2, ud_tags2, unstressed_prefix2, unstressed_tail2):
    word1 = accentuator.yoficate(accentuator.sanitize_word(word1))
    word2 = accentuator.yoficate(accentuator.sanitize_word(word2))

    if not unstressed_tail1 and not unstressed_tail2:
        if (word1.lower(), word2.lower()) in accentuator.rhymed_words or (word2.lower(), word1.lower()) in accentuator.rhymed_words:
            return True

    vow_count1 = accentuator.get_vowel_count(word1)
    pos1 = vow_count1 - stress1 + accentuator.get_vowel_count(unstressed_tail1, abbrevs=False)

    vow_count2 = accentuator.get_vowel_count(word2)
    pos2 = vow_count2 - stress2 + accentuator.get_vowel_count(unstressed_tail2, abbrevs=False)

    # смещение ударной гласной от конца слова должно быть одно и то же
    # для проверяемых слов.
    if pos1 == pos2:
        # 22.05.2022 Особо рассматриваем рифмовку с местоимением "я":
        # пролета́ет ле́то
        # гру́сти не тая́
        # и аналоги́чно
        # пролета́ю я́
        if word2 == 'я' and len(word1) > 1 and word1[-2] in 'аеёиоуэюяь' and word1[-1] == 'я':
            return True

        # Получаем клаузулы - все буквы, начиная с ударной гласной
        ending1 = extract_ending_prononciation_after_stress(accentuator, word1, stress1, ud_tags1, unstressed_prefix1, unstressed_tail1)
        ending2 = extract_ending_prononciation_after_stress(accentuator, word2, stress2, ud_tags2, unstressed_prefix2, unstressed_tail2)

        # Фонетическое сравнение клаузул.
        return are_phonetically_equal(ending1, ending2)

    return False



fuzzy_ending_pairs = [
    (r'\^утам', r'\^уты'),  # парашютам - тьфу ты

    (r'\^ады', r'\^аты'),  # пощады - борща ты

    (r'\^овый', r'\^овай'),  # трёхочковый - волочковой

    (r'\^ымым', r'\^ымам'),  # одерж^имым - ж^имом

    (r'\^ыськы', r'\^ыскы'),  # с^иськи - в^иски

    (r'\^альцэ', r'\^альца'),  # еб^альце - п^альца

    (r'\^озин', r'\^озэ'),  # вирту^озен - п^озе

    (r'\^убым', r'\^убам'),  # грубым - ледорубом

    (r'\^ыскра', r'\^ыстра'),  # и́скра - кани́стра

    (r'\^ызар', r'\^ыза'),  # телевизор - антифриза

    (r'\^анай', r'\^аный'),  # манной - странный

    (r'\^очна', r'\^очный'),  # нарочно - молочный

    (r'\^юц[эа]', r'\^удц[аэ]'),  # льются - блюдце

    (r'[:C:]\^ое', r'[:C:]\^оя'),  # простое - покоя

    (r'\^ает', r'\^ают'),  # чает - повенчают

    (r'\^ывый', r'\^ыва'),  # белогривый - некрасиво

    (r'\^энья', r'\^энье'),  # настроенья - упоенье

    (r'\^айна', r'\^айнай'),  # неслучайно - тайной

    (r'\^овым', r'\^овам'),  # еловым - основам

    (r'\^авай', r'\^ава'),  #  славой - права

    (r'\^([:A:][:C:]+)а', r'\^([:A:][:C:]+)э'),  # риска - миске

    (r'\^ыны', r'\^ына'),  # цепеллины - господина

    (r'\^([:A:][:C:][:A:][:C:])а', r'\^([:A:][:C:][:A:][:C:])ам'), # яруса - парусом

    (r'\^ысил', r'\^ысыл'),  # чисел - возвысил

    (r'\^([:A:][:C:]{2,})э', r'\^([:A:][:C:]{2,})ай'),  # миске - пропиской

    (r'\^([:A:][щч])ие', r'\^([:A:][щч])ые'),  # щемящее - парящие

    (r'([:C:])\^ою', r'([:C:])\^ое'),  # щекою - такое

    (r'\^([:A:][:C:])ай', r'\^([:A:][:C:])а'),  # прохладой - надо

    (r'\^([:A:][жш])[эау]', r'\^([:A:][жш])[эау]'),  # коже - тревожа

    (r'\^([:A:]ш)ин', r'\^([:A:]ш)ан'),  #  вишен - услышан

    (r'\^([:A:][:C:]+ь)э', r'\^([:A:][:C:]+ь)а'),  # спасенье - вознесенья

    (r'\^([:A:][:C:]+)[ыэ]', r'\^([:A:][:C:]+)[ыэ]'),  # медведи - велосипеде

    (r'\^([:A:][:C:]+)[оуа]', r'\^([:A:][:C:]+)[оау]м'),  # Андрюшка - хрюшкам

    (r'\^([:A:][:C:]+[ыэ])[й]', r'\^([:A:][:C:]+[ыэ])'),  # первый-нервы

    (r'([:C:]+\^[ыэ])[й]', r'([:C:]+\^[ыэ])'),  # свиней-войне

    (r'\^([:A:][:C:]+)[оуа][мн]', r'\^([:A:][:C:]+)[оау]'),  # сонетом - Света

    (r'([:C:])\^[оуа]н', r'([:C:])\^[оау]'),  # Антон - манто

    (r'(\^[:A:][:C:]+)[оуа]', r'(\^[:A:][:C:]+)[оау]'),  # ложа - кожу
                      ]

def check_ending_rx_matching_2(word1, word2, s1, s2):
    for x, y in [(':C:', 'бвгджзклмнпрстфхцчшщт'), (':A:', 'аоеёиуыюэюя')]:
        s1 = s1.replace(x, y)
        s2 = s2.replace(x, y)

    m1 = re.search(s1 + '$', word1)
    m2 = re.search(s2 + '$', word2)
    if m1 and m2:
        for g1, g2 in zip(m1.groups(), m2.groups()):
            if g1 != g2:
                return False

        return True
    else:
        return False


def render_xword(accentuator, word, stress_pos, ud_tags, unstressed_prefix, unstressed_tail):
    unstressed_prefix_transcript = transcript_unstressed(unstressed_prefix)
    unstressed_tail_transcript = transcript_unstressed(unstressed_tail)

    phonems = []

    VOWELS = 'уеыаоэёяию'

    # Упрощенный алгоритм фонетической транскрипции - не учитываем йотирование, для гласных июяеё не помечаем
    # смягчение предшествующих согласных, etc.
    v_counter = 0
    for i, c in enumerate(word.lower()):
        if c in VOWELS:
            v_counter += 1
            if v_counter == stress_pos:
                # Достигли ударения
                # Вставляем символ "^"
                phonems.append('^')

                ending = word[i:]
                if ud_tags is not None and ('ADJ' in ud_tags or 'DET' in ud_tags) and ending == 'ого':
                    # Меняем "люб-ОГО" на "люб-ОВО"
                    phonems.extend('ова')
                    break
                elif ending[1:] in ('ться', 'тся'):
                    phonems.append(c)
                    phonems.append('ц')
                    phonems.extend('а')
                    break
                else:
                    # Добавляем ударную гласную и продолжаем обрабатывать символы справа от него как безударные
                    if c == 'е':
                        c = 'э'
                    elif c == 'я':
                        c = 'а'
                    elif c == 'ё':
                        c = 'о'
                    elif c == 'ю':
                        c = 'у'
                    elif c == 'и':
                        # 01.02.2022 меняем ударную "и" на "ы": сливы==>сл'Ывы, живы==>жЫвы
                        c = 'ы'

                    phonems.append(c)
            else:
                # Еще не достигли ударения или находимся справа от него.
                if c == 'о':
                    # безударная "о" превращается в "а"
                    c = 'а'
                elif c == 'е':
                    if len(phonems) == 0 or phonems[-1] in VOWELS+'ь':
                        # первую в слове, и после гласной, 'е' оставляем (должно быть что-то типа je)
                        pass
                    else:
                        # металле ==> митал'э
                        if i == len(word)-1:
                            c = 'э'
                        else:
                            c = 'и'
                elif c == 'я':
                    if len(phonems) == 0 or phonems[-1] in VOWELS+'ь':
                        pass
                    else:
                        c = 'а'
                elif c == 'ё':
                    if len(phonems) == 0 or phonems[-1] in VOWELS:
                        pass
                    else:
                        c = 'о'
                elif c == 'ю':
                    if len(phonems) == 0 or phonems[-1] in VOWELS+'ь':
                        pass
                    else:
                        c = 'у'
                elif c == 'и':
                    if len(phonems) == 0 or phonems[-1] in VOWELS+'ь':
                        pass
                    else:
                        # меняем ЦИ -> ЦЫ
                        #if c2 in 'цшщ' and ending[0] == 'и':
                        c = 'ы'

                phonems.append(c)
        else:
            # строго говоря, согласные надо бы смягчать в зависимости от следующей буквы (еёиюяь).
            # но нам для разметки стихов это не нужно.

            if c == 'ж':
                # превращается в "ш", если дальше идет глухая согласная
                # прожка ==> прошка
                if i < len(word)-1 and word[i+1] in 'пфктс':
                    c = 'ш'

            if i == len(word)-1:
                if c == 'д':  # последняя "д" оглушается до "т":  ВЗГЛЯД
                    c = 'т'
                elif c == 'ж':  # оглушаем последнюю "ж": ЁЖ
                    c = 'ш'
                elif c == 'з':  # оглушаем последнюю "з": МОРОЗ
                    c = 'с'
                elif c == 'г':  # оглушаем последнюю "г": БОГ
                    c = 'х'
                elif c == 'б':  # оглушаем последнюю "б": ГРОБ
                    c = 'п'
                elif c == 'в':  # оглушаем последнюю "в": КРОВ
                    c = 'ф'

            phonems.append(c)

    if len(phonems) > 2 and phonems[-1] == 'ь' and phonems[-2] in 'шч':  # убираем финальный мягкий знак: "ВОЗЬМЁШЬ", РОЖЬ, МЫШЬ
        phonems = phonems[:-1]

    xword = unstressed_prefix_transcript + ''.join(phonems) + unstressed_tail_transcript
    #xword = accentuator.pronounce(xword)

    # СОЛНЦЕ -> СОНЦЕ
    xword = xword.replace('лнц', 'нц')

    # СЧАСТЬЕ -> ЩАСТЬЕ
    xword = xword.replace('сч', 'щ')

    # БРАТЬСЯ -> БРАЦА
    xword = xword.replace('ться', 'ца')

    # БОЯТСЯ -> БОЯЦА
    xword = xword.replace('тся', 'ца')

    # БРАТЦЫ -> БРАЦЫ
    xword = xword.replace('тц', 'ц')

    #
    #         # ЖИР -> ЖЫР
    #         s = s.replace('жи', 'жы')
    #
    #         # ШИП -> ШЫП
    #         s = s.replace('ши', 'шы')
    #
    #         # МОЦИОН -> МОЦЫОН
    #         s = s.replace('ци', 'цы')
    #
    #         # ЖЁСТКО -> ЖОСТКО
    #         s = s.replace('жё', 'жо')
    #
    #         # ОКОНЦЕ -> ОКОНЦЭ
    #         s = s.replace('це', 'цэ')
    #

    # двойные согласные:
    # СУББОТА -> СУБОТА
    xword = re.sub(r'([бвгджзклмнпрстфхцчшщ])\1', r'\1', xword)

    # оглушение:
    # СКОБКУ -> СКОПКУ
    new_s = []
    for c1, c2 in zip(xword, xword[1:]):
        if c2 in 'кпстфх':
            new_s.append(accentuator.conson(c1))
        else:
            new_s.append(c1)
    xword = ''.join(new_s) + xword[-1]

    #
    #         # последнюю согласную оглушаем всегда:
    #         # ГОД -> ГОТ
    #         new_s.append(self.conson(s[-1]))
    #
    #         s = ''.join(new_s)


    # огрушаем последнюю согласную с мягким знаком:
    # ВПРЕДЬ -> ВПРЕТЬ
    if len(xword) >= 2 and xword[-1] == 'ь' and xword[-2] in 'бвгдз':
        xword = xword[:-2] + accentuator.conson(xword[-2]) + 'ь'

    if '^' in xword:
        apos = xword.index('^')
        if apos == len(xword) - 2:
            # ударная гласная - последняя, в этом случае включаем предшествующую букву.
            clausula = xword[apos-1:]
        else:
            clausula = xword[apos:]
    else:
        clausula = xword

    return xword, clausula


def rhymed_fuzzy(accentuator, word1, stress1, ud_tags1, word2, stress2, ud_tags2):
    return rhymed_fuzzy2(accentuator, word1, stress1, ud_tags1, '', None, word2, stress2, ud_tags2, '', None)


def rhymed_fuzzy2(accentuator, word1, stress1, ud_tags1, unstressed_prefix1, unstressed_tail1, word2, stress2, ud_tags2, unstressed_prefix2, unstressed_tail2):
    if stress1 is None:
        stress1 = accentuator.get_accent(word1, ud_tags1)

    if stress2 is None:
        stress2 = accentuator.get_accent(word2, ud_tags2)

    xword1, clausula1 = render_xword(accentuator, word1, stress1, ud_tags1, unstressed_prefix1, unstressed_tail1)
    xword2, clausula2 = render_xword(accentuator, word2, stress2, ud_tags2, unstressed_prefix2, unstressed_tail2)

    if len(clausula1) >= 3 and clausula1 == clausula2:
        # клаузуллы достаточно длинные и совпадают:
        # поэтом - ответом
        return True

    for s1, s2 in fuzzy_ending_pairs:
        if check_ending_rx_matching_2(xword1, xword2, s1, s2):
            #print('\nDEBUG@859 word1={} rx={}  <==>  word2={} rx={}\n'.format(xword1, s1, xword2, s2))
            return True

        if check_ending_rx_matching_2(xword1, xword2, s2, s1):
            #print('\nDEBUG@863 word1={} rx={}  <==>  word2={} rx={}\n'.format(xword1, s2, xword2, s1))
            return True

    if accentuator.allow_rifmovnik and len(word1) >= 2 and len(word2) >= 2:
        eword1, keys1 = extract_ekeys(word1, stress1)
        eword2, keys2 = extract_ekeys(word2, stress2)
        for key1 in keys1:
            if key1 in accentuator.rhyming_dict:
                for key2 in keys2:
                    if key2 in accentuator.rhyming_dict[key1]:
                        #print('\nDEBUG@1006 for word word1="{}" word2="{}"\n'.format(word1, word2))
                        return True

    return False



def extract_ekeys(word, stress):
    cx = []
    vcount = 0
    stressed_c = None
    for c in word:
        if c in 'аеёиоуыэюя':
            vcount += 1
            if vcount == stress:
                stressed_c = c.upper()
                cx.append(stressed_c)
            else:
                cx.append(c)
        else:
            cx.append(c)

    word1 = ''.join(cx)
    keys1 = []
    eword1 = None
    for elen in range(2, len(word1)):
        eword1 = word1[-elen:]
        if eword1[0] == stressed_c or eword1[1] == stressed_c:
            keys1.append(eword1)
    return eword1, keys1


if __name__ == '__main__':
    data_folder = '../../data/poetry/dict'
    tmp_dir = '../../tmp'

    # Проверим валидность содержимого файла с неоднозначными ударениями.
    ambiguous_accents2 = yaml.safe_load(io.open(os.path.join(data_folder, 'ambiguous_accents_2.yaml'), 'r', encoding='utf-8').read())
    for key, vars in ambiguous_accents2.items():
        if not all((len(re.findall('[АЕЁИОУЫЭЮЯ]', s)) == 1) for s in vars):
            print('Файл "ambiguous_accents_2.yaml" для статьи "{}" содержит вариант без ударности: {}'.format(key, ' '.join(vars)))
            exit(0)

    # НАЧАЛО ОТЛАДКИ
    if False:
        accents = Accents()
        accents.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
        accents.after_loading(stress_model_dir='../../tmp/stress_model')

        true_accents = set()
        with io.open(os.path.join(data_folder, 'true_accents.txt'), 'r') as rdr:
            for line in rdr:
                true_accents.add(line.strip().lower())

        with io.open(os.path.join(tmp_dir, 'invalid_accents.txt'), 'w') as wrt:
            for word, stress in accents.word_accents_dict.items():
                if 'ё' in word and word not in true_accents:
                    accent_pos = word.index('ё')
                    expected_stress = accents.get_vowel_count(word[:accent_pos], abbrevs=False) + 1

                    if stress != expected_stress:
                        stress_found = False
                        n_vowels = 0
                        s = ''
                        for c in word:
                            if c in 'уеыаоэёяию':
                                n_vowels += 1
                            if n_vowels == stress and not stress_found:
                                s += '^'
                                stress_found = True
                            s += c
                        print('{}\t\t{}'.format(s, word.replace('ё', 'Ё')))
                        wrt.write(word.replace('ё', 'Ё')+'\n')
        exit(0)
    # КОНЕЦ ОТЛАДКИ

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

    # проверяем детектирование ударения для слов, в которых это не требует морфологических тегов.
    xwords = 'чЯщя щЯми худОжэственный вдвоЁм разъЁм пъЯнки птИцэй жОлтый шОпот тишынА сжыгАю соцыАльная магнитозавИсимыми электрощЁточную'.split() +\
             'шестиузловОй двадцатисторОнними двухпротОнными восьмимЕрного пятиствОрчатый тЁрок алЁна самовЫгул нанореснИчками'.split() +\
             'трёхвалЕнтный голубОе дЕтям сочнЕйшего землЕй дождЕй полудождЯми конЕк остаЁтся остаЕтся гОды кИн кинО сЫн'.split() +\
             'крУтиш знАеш смОжеш льЮцца бъЕтся грУсный брАццы прЯчте блЕщщут щщУпать Эхооооо мчИцца рвЕтьса сссУка клАвишь'.split() +\
             'плОтьник щщенкИ скрЫтьса стрАсно сьЕли дмИтрий дмИттрий льЮцца спрЯчте свЕточь здАчу гнУтса'.split() +\
             'швАбрщик сберЕч встрЕтимса трИццать щщАстья вощОный суперрУку бьЮца лжеолЕгом втроЁм вдвоЁм проЁмом плЁс'.split() +\
             'метаисУсу'.split()
    for word in xwords:
        n_vowels = 0
        true_stress = -1
        for c in word:
            if c.lower() in 'уеыаоэёяию':
                n_vowels += 1
                if c.isupper():
                    true_stress = n_vowels
                    break
        pred_stress = accents.get_accent(word.lower())
        if pred_stress != true_stress:
            print('Invalid stress position for word "{}" expected: {}, predicted: {}'.format(word, true_stress, pred_stress))
            exit(0)


    i = accents.get_accent('груди', ['Case=Loc'])
    assert(i == 2)


    # =========================================================
    # Поверка точной рифмовки слов без неоднозначностей
    # =========================================================

    r = rhymed(accents, 'проём', [], 'втроём', [])
    assert(r is True)

    r = rhymed(accents, 'разъём', [], 'вдвоём', [])
    assert(r is True)

    r = rhymed(accents, 'яны', [], 'обезьяны', [])
    assert(r is True)

    r = rhymed(accents, 'говорят', [], 'взгляд', [])
    assert(r is True)

    r = rhymed(accents, 'ёж', [], 'возьмёшь', [])
    assert(r is True)

    r = rhymed(accents, 'рожь', [], 'ножь', [])
    assert(r is True)

    r = rhymed(accents, 'гроб', [], 'поп', [])
    assert(r is True)

    r = rhymed(accents, 'семью', [], 'пью', [])
    assert(r is True)

    r = rhymed(accents, 'друзья', [], 'я', [])
    assert(r is True)

    r = rhymed(accents, 'льна', [], 'война', [])
    assert(r is True)

    r = rhymed(accents, 'холодце', [], 'отце', [])
    assert(r is True)

    r = rhymed(accents, 'клён', [], 'омон', [])
    assert(r is True)

    r = rhymed(accents, 'чётки', [], 'обмотки', [])
    assert(r is True)

    r = rhymed(accents, 'ложка', [], 'плошка', [])
    assert(r is True)

    r = rhymed(accents, 'щётка', [], 'обмотка', [])
    assert(r is True)

    r = rhymed(accents, 'парашют', [], 'шут', [])
    assert(r is True)

    r = rhymed(accents, 'фоссы', [], 'осы', [])
    assert(r is True)

    r = rhymed(accents, 'Серёжа', [], 'ложа', [])
    assert(r is True)

    r = rhymed(accents, 'Люся', [], 'муся', [])
    assert(r is True)


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

    r = rhymed(accents, 'побеседуем', [], 'дуем', [])
    assert(r is False)

    r = rhymed(accents, 'кури', [], 'куры', [])
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

    # ======================================================================================
    # Проверяем процедуру проверки рифмованности двух слов с учетом морфологических тегов
    # ======================================================================================

    r = rhymed(accents, 'я', [], 'ружья', ['Case=Gen'])
    assert(r is True)

    r = rhymed(accents, 'семью', [], 'мою', ['ADJ'])
    assert(r is True)

    r = rhymed(accents, 'ружья', ['Case=Gen'], 'твоя', [])
    assert(r is True)

    r = rhymed(accents, 'любой', 'ADJ|Case=Gen'.split('|'), 'русской', 'ADJ|Case=Gen'.split('|'))
    assert(r is False)

    r = rhymed(accents, 'века', ['Case=Gen'], 'аптека', [])
    assert(r is True)

    r = rhymed(accents, 'века', ['Case=Nom'], 'аптека', [])
    assert(r is False)

    r = rhymed(accents, 'коровы', [], 'совы', ['Case=Nom'])
    assert(r is True)

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

    # =======================================
    # ПРОВЕРКА НЕЧЕТКОЙ РИФМОВКИ
    # TODO - переделать на цикл по списку пар.

    # трёхочковый - волочковой
    r = rhymed_fuzzy(accents, 'трёхочковый', None, [], 'волочковой', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'сиськи', None, [], 'виски', None, 'Case=Nom|Number=Sing'.split('|'))
    assert(r is True)

    r = rhymed_fuzzy(accents, 'одержимым', None, [], 'жимом', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'трёхочковый', None, [], 'волочковой', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'тело', None, [], 'хотела', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'пиздежами', None, [], 'пижаме', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'ебальце', None, [], 'пальца', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'виртуозен', None, [], 'позе', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'нет', None, [], 'людоед', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'телевизор', None, [], 'антифриза', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'манной', None, [], 'странный', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'нарочно', None, [], 'молочный', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'льются', None, [], 'блюдце', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'простое', None, [], 'покоя', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'спасенье', None, [], 'вознесенья', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'чает', None, [], 'повенчают', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'белогривый', None, [], 'некрасиво', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'благо', None, [], 'шагом', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'настроенья', None, [], 'упоенье', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'неслучайно', None, [], 'тайной', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'еловым', None, [], 'основам', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'славой', None, [], 'право', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'спастись', None, [], 'ввысь', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'риска', None, [], 'миске', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'колыбели', None, [], 'еле', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'славой', None, [], 'право', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'цепеллины', None, [], 'господина', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'яруса', None, [], 'парусом', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'чисел', None, [], 'возвысил', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'поэтом', None, [], 'ответом', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'миске', None, [], 'пропиской', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'чумазый', None, [], 'заразы', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'щемящее', None, [], 'парящие', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'щекою', None, [], 'такое', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'главном', None, [], 'плавно', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'прохладой', None, [], 'надо', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'вишен', None, [], 'услышан', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'коже', None, [], 'тревожа', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'сигаретку', None, [], 'редко', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'блюде', None, [], 'люди', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'рубашки', None, [], 'отмашке', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'свиней', None, [], 'войне', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'Андрюшка', None, [], 'хрюшкам', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'Антон', None, [], 'манто', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'первый', None, [], 'нервы', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'сонетом', None, [], 'Света', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'лето', None, [], 'котлету', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'победа', None, [], 'приеду', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'манежу', None, [], 'невежа', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'Антошка', None, [], 'дорожкам', None, [])
    assert(r is True)

    r = rhymed_fuzzy(accents, 'медведи', None, [], 'велосипеде', None, [])
    assert(r is True)

    # =====================================

    # Проверяем вспомогательную процедуру определения ударения для случаев, когда
    # нужно учитывать морфологические признаки слова, чтобы снять неоднозначность.
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
