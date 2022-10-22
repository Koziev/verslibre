"""
Обертка над пайплайном Stanza NLP плюс набор ad hoc костылей, чтобы исправлять
безумные результаты в частотных случаях.
"""

import stanza
import re


class StanzaParserWord:
    def __init__(self, stanza_word):
        self.tags = stanza_word.feats.split('|') if stanza_word.feats is not None else []
        self.id = stanza_word.id
        self.form = stanza_word.text
        self.lemma = stanza_word.lemma
        self.upos = stanza_word.upos
        if self.form.lower() in ('бы', 'б', 'ка'):
            self.upos = 'PART'

    def __repr__(self):
        return self.form

class StanzaParser(object):
    def __init__(self):
        #self.nlp = stanza.Pipeline(lang='ru', processors='tokenize,pos,lemma,depparse', verbose=False)
        self.nlp = stanza.Pipeline(lang='ru', processors='tokenize,pos,lemma', verbose=False)

    def load(self, model_path):
        pass

    def parse_text(self, text):
        parsings = []

        doc = self.nlp(text)
        for sent in doc.sentences:
            parsing = []
            for token in sent.tokens:
                for iword, word in enumerate(token.words):
                    t_word = StanzaParserWord(word)
                    if t_word.form.lower() == 'иль' and t_word.upos != 'CCONJ':
                        t_word.upos = 'CCONJ'
                        t_word.tags = []
                    elif t_word.form.lower() in ('ль', 'ага') and t_word.upos != 'PART':
                        t_word.upos = 'PART'
                        t_word.tags = []
                    elif t_word.form == 'маня' and t_word.upos == 'NOUN':
                        # переделываем на деепричастие.
                        t_word.upos = 'VERB'
                        t_word.lemma = 'манить'
                        t_word.tags = ['VerbForm=Conv']
                    elif t_word.form == 'года' and parsing and parsing[-1].form.lower() == 'пол':
                        # токенизатор станзы разбивает слово "полгода" на 2 токена.
                        # схлопнем обратно.
                        t_word.form = parsing[-1].form + t_word.form
                        parsing = parsing[:-1]
                    elif t_word.upos == 'PUNCT' and re.match(r'^\w+$', t_word.form, re.I):
                        # Иногда Stanza делает невообразимые вещи, например - распознает глагол как PUNCT:
                        # Дай руку и словом согрей
                        #                   ^^^^^^
                        # Это мешает в генераторе стихов при поиске рифмуемого слова, так как финальные пунктуаторы
                        # мы должны отсекать.
                        t_word.upos = 'UNK'

                    parsing.append(t_word)

            for word1, word2 in zip(parsing, parsing[1:]):
                if word1.form.lower() == 'пора' and word1.upos == 'NOUN' and word2.upos == 'VERB' and 'VerbForm=Inf' in word2.tags:
                    # Часто возникает ошибка с определением признаков безличного глагола "пора":
                    # "пора спать"
                    # Меняем существительное на глагол.
                    word1.upos = 'VERB'
                    word1.tags = []
                elif word1.form.lower() == 'для' and word2.form.lower() == 'начала' and word2.upos == 'VERB':
                    # станца считает "начала" глаголом.
                    # это очень частотное словосочетание, поэтому поправим руками.
                    # Чтоб вспомнить акушерку, для начала
                    word2.upos = 'NOUN'
                    word2.tags = ['Case=Gen', 'Number=Sing', 'Gender=Neut']
                elif word1.form.lower() == 'в' and word2.form.lower() == 'дома' and 'Number=Sing' in word2.tags:
                    # Дедок Святой в дома ходил
                    #                ^^^^
                    word2.tags = ['Case=Acc', 'Number=Plur', 'Gender=Masc']
                elif word1.form.lower() == 'на' and word2.form.lower() == 'луга' and 'Number=Sing' in word2.tags:
                    # Водил на Луга Изобилий
                    #       ^^^^^^^
                    word2.tags = ['Case=Acc', 'Number=Plur', 'Gender=Masc']

            if parsing:
                parsings.append(parsing)

        return parsings


if __name__ == '__main__':
    import terminaltables

    parser = StanzaParser()

    while True:
        text = input(':> ').replace('\u0301', '').strip()
        parsings = parser.parse_text(text)
        for parsing in parsings:
            table = [['id', 'word', 'lemma', 'upos', 'feats']]
            for word in parsing:
                table.append((word.id, word.form, word.lemma, word.upos, ' '.join(word.tags)))

            table = terminaltables.AsciiTable(table)
            print(table.table)

