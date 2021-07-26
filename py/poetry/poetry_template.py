# -*- coding: utf-8 -*-

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import codecs
import logging


class PoetryToken:
    def __init__(self, token, gren):
        tx = token.split('|')
        self.word = tx[0]
        self.tags = []
        self.pos = gren.get_pos(self.word)

        if self.word in u'до как'.split():
            self.tags.append((u'ПАДЕЖ', u'ИМ'))
            self.tags.append((u'<<none>>', u'<<none>>'))  # особый тег, чтобы запретить подстановку для слова
        else:
            if len(tx) > 1:
                for tag in tx[1:]:
                    if tag == u'им':
                        self.tags.append((u'ПАДЕЖ', u'ИМ'))
                    elif tag == u'род':
                        self.tags.append((u'ПАДЕЖ', u'РОД'))
                    elif tag == u'тв':
                        self.tags.append((u'ПАДЕЖ', u'ТВОР'))
                    elif tag == u'дат':
                        self.tags.append((u'ПАДЕЖ', u'ДАТ'))
                    elif tag == u'вин':
                        self.tags.append((u'ПАДЕЖ', u'ВИН'))
                    elif tag == u'предл':
                        self.tags.append((u'ПАДЕЖ', u'ПРЕДЛ'))
                    elif tag == u'ед':
                        self.tags.append((u'ЧИСЛО', u'ЕД'))
                    elif tag == u'мн':
                        self.tags.append((u'ЧИСЛО', u'МН'))
                    elif tag == u'муж':
                        self.tags.append((u'РОД', u'МУЖ'))
                    elif tag == u'жен':
                        self.tags.append((u'РОД', u'ЖЕН'))
                    elif tag == u'ср':
                        self.tags.append((u'РОД', u'СР'))
                    elif tag == u'одуш':
                        self.tags.append((u'ОДУШ', u'ОДУШ'))
                    elif tag == u'неодуш':
                        self.tags.append((u'ОДУШ', u'НЕОДУШ'))
                    elif tag == u'кр':
                        self.tags.append((u'КРАТКИЙ', u'1'))
                    elif tag == u'страд':
                        self.tags.append((u'СТРАД', u'1'))
                    elif tag == u'прич':
                        self.tags.append((u'ПРИЧ', u'1'))
                    elif tag == u'сравн':
                        self.tags.append((u'СТЕПЕНЬ', u'СРАВН'))
                    elif tag == u'инф':
                        self.tags.append((u'НАКЛОНЕНИЕ', u'ИНФИНИТИВ'))
                    elif tag == u'деепр':
                        self.tags.append((u'НАКЛОНЕНИЕ', u'ДЕЕПРИЧАСТИЕ'))
                    elif tag == u'побуд':
                        self.tags.append((u'НАКЛОНЕНИЕ', u'ПОБУД'))
                    elif tag == u'изъяв':
                        self.tags.append((u'НАКЛОНЕНИЕ', u'ИЗЪЯВ'))
                    elif tag == u'1':
                        self.tags.append((u'ЛИЦО', u'1'))
                    elif tag == u'2':
                        self.tags.append((u'ЛИЦО', u'2'))
                    elif tag == u'3':
                        self.tags.append((u'ЛИЦО', u'3'))
                    elif tag == u'прош':
                        self.tags.append((u'ВРЕМЯ', u'ПРОШЕДШЕЕ'))
                    elif tag == u'наст':
                        self.tags.append((u'ВРЕМЯ', u'НАСТОЯЩЕЕ'))
                    elif tag == u'буд':
                        self.tags.append((u'ВРЕМЯ', u'БУДУЩЕЕ'))
                    elif tag == u'прич':
                        self.tags.append((u'ПРИЧАСТИЕ', u'1'))
                    elif tag == u'мод':
                        self.tags.append((u'МОДАЛЬНЫЙ', u'1'))
                    elif tag == u'сов':
                        self.tags.append((u'ВИД', u'СОВЕРШ'))
                    elif tag == u'несов':
                        self.tags.append((u'ВИД', u'НЕСОВЕРШ'))
                    elif tag == u'перех':
                        self.tags.append((u'ПЕРЕХОДНОСТЬ', u'ПЕРЕХОДНЫЙ'))
                    elif tag == u'нареч':
                        self.tags.append((u'ЧАСТЬ_РЕЧИ', u'НАРЕЧИЕ'))
                        self.pos = 'НАРЕЧИЕ'
                    elif tag == u'прил':
                        self.tags.append((u'ЧАСТЬ_РЕЧИ', u'ПРИЛАГАТЕЛЬНОЕ'))
                        self.pos = 'ПРИЛАГАТЕЛЬНОЕ'
                    elif tag == u'гл':
                        self.tags.append((u'ЧАСТЬ_РЕЧИ', u'ГЛАГОЛ'))
                        self.pos = 'ГЛАГОЛ'
                    elif tag == u'сущ':
                        self.tags.append((u'ЧАСТЬ_РЕЧИ', u'СУЩЕСТВИТЕЛЬНОЕ'))
                        self.pos = 'СУЩЕСТВИТЕЛЬНОЕ'
                    elif tag == u'none':
                        self.tags.append((u'<<none>>', u'<<none>>'))  # особый тег, чтобы запретить подстановку для слова
                    else:
                        logging.error('Unknown tag "%s" in template token "%s', tag, token)
                        raise RuntimeError()

        self.ud_tags = []

        if self.pos == 'СУЩЕСТВИТЕЛЬНОЕ':
            self.ud_tags.append('NOUN')
        elif self.pos == 'ПРИЛАГАТЕЛЬНОЕ':
            self.ud_tags.append('ADJ')
        elif self.pos in ('ГЛАГОЛ', 'ИНФИНИТИВ', 'ДЕЕПРИЧАСТИЕ'):
            self.ud_tags.append('VERB')

        for t in self.tags:
            if t[1] == 'ИМ':
                self.ud_tags.append('Case=Nom')
            elif t[1] == 'РОД':
                self.ud_tags.append('Case=Gen')
            elif t[1] == 'ВИН':
                self.ud_tags.append('Case=Acc')
            elif t[1] == 'ТВОР':
                self.ud_tags.append('Case=Ins')
            elif t[1] == 'ДАТ':
                self.ud_tags.append('Case=Dat')
            elif t[1] == 'ПРЕДЛ':
                self.ud_tags.append('Case=Loc')
            elif t[1] == 'ЕД':
                self.ud_tags.append('Number=Sing')
            elif t[1] == 'МН':
                self.ud_tags.append('Number=Plur')
            elif t[1] == 'МУЖ':
                self.ud_tags.append('Gender=Mas')
            elif t[1] == 'ЖЕН':
                self.ud_tags.append('Gender=Fem')
            elif t[1] == 'СР':
                self.ud_tags.append('Gender=Neut')
            elif t[1] == 'БУДУЩЕЕ':
                self.ud_tags.append('Tense=Fut')
            elif t[1] == 'НАСТОЯЩЕЕ':
                self.ud_tags.append('Tense=Pres')
            elif t[1] == 'ПРОШЕДШЕЕ':
                self.ud_tags.append('Tense=Past')

    def is_replaceable(self):
        return self.tags and (u'<<none>>', u'<<none>>') not in self.tags

    def __repr__(self):
        s = self.word
        if self.tags:
            s += '|' + '|'.join('{}={}'.format(t, v) for t, v in self.tags)
        return s


class PoetryLine:
    def __init__(self, orig_text, tokens):
        self.orig_text = orig_text
        self.tokens = tokens

    def __repr__(self):
        return ' '.join(str(t) for t in self.tokens)


class PoetryTemplate:
    def sanitize_line(self, line):
        clean_text = []
        icur = 0
        l = len(line)
        while icur < l:
            c = line[icur]
            icur += 1
            if c != u'|':
                clean_text.append(c)
            else:
                while icur < l:
                    if line[icur] in u' ,:;-!?.—':
                        break
                    icur += 1

        return u''.join(clean_text)

    def __init__(self, template_path, author_id, tokenizer, gren):
        self.path = template_path
        self.author_id = author_id
        self.lines = []
        with codecs.open(template_path, 'r', 'utf-8') as rdr:
            for iline, line in enumerate(rdr):
                try:
                    line = line.rstrip()  # отсекает только пробелы справа, так как левые нужны для стиля Маяковского
                    orig_line = self.sanitize_line(line)
                    tokens = tokenizer.tokenize(line)
                    line_tokens = [PoetryToken(token, gren) for token in tokens]
                    parsed_line = PoetryLine(orig_line, line_tokens)
                    self.lines.append(parsed_line)
                except Exception as ex:
                    print('Error in line #{} in template "{}": {}'.format(iline, template_path, line))
                    print(ex)
                    raise RuntimeError()

    def __repr__(self):
        return str(self.lines[0])

    def get_all_words(self):
        words = []
        for line in self.lines:
            for token in line.tokens:
                words.append(token.word)
        return words
