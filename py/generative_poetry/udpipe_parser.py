import os
import pickle

import pyconll
from ufal.udpipe import Model, Pipeline, ProcessingError


class UDPipeToken:
    def __init__(self, ud_token, upos=None, tags=None):
        self.id = ud_token.id
        self.form = ud_token.form
        self.upos = ud_token.upos if upos is None else upos
        self.lemma = ud_token.lemma
        self.tags = [(k + '=' + list(vx)[0]) for k, vx in ud_token.feats.items()] if tags is None else list(tags)
        self.deprel = ud_token.deprel
        self.head = ud_token.head

    def __repr__(self):
        return self.form

    def get_attr(self, attr_name):
        k = attr_name + '='
        for t in self.tags:
            if t.startswith(k):
                return t.split('=')[1]
        return ''

    def feats(self):
        return dict((s.split('=')) for s in self.tags)


def get_attr(token, tag_name):
    if tag_name in token.feats:
        v = list(token.feats[tag_name])[0]
        return v

    return ''


class Parsing(object):
    def __init__(self, tokens, text):
        self.tokens = tokens
        self.text = text

    def get_text(self):
        return self.text

    def __repr__(self):
        return self.text

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return list(self.tokens).__iter__()

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.tokens[i]
        elif isinstance(i, slice):
            return self.tokens[i]
        else:
            return self.tokens[int(i)-1]

    def get_root(self):
        for t in self.tokens:
            if t.deprel == 'root':
                return t
        return None


class UdpipeParser:
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.error = None
        self.word2lemma = None

    def load(self, model_path):
        model_dir = model_path
        if os.path.isfile(model_path):
            udp_model_file = model_path
            model_dir = os.path.dirname(udp_model_file)
        else:
            udp_model_file = os.path.join(model_path, 'udpipe_syntagrus.model')

        self.model = Model.load(udp_model_file)
        self.pipeline = Pipeline(self.model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        self.error = ProcessingError()

        with open(os.path.join(model_dir, 'word2lemma.pkl'), 'rb') as f:
            self.word2lemma = pickle.load(f)
            #self.word2posx = pickle.load(f)

    def parse_text(self, text):
        parsings = []

        processed = self.pipeline.process(text, self.error)
        if self.error.occurred():
            return None
        try:
            for parsing0 in pyconll.load_from_string(processed):
                parsing = []
                for itoken, token in enumerate(parsing0):
                    utoken = token.form.lower()

                    # 24-12-2021 Руками исправляем некоторые очень частотные ошибки.
                    if utoken == 'душе':
                        is_soul_dative = False
                        if token.id == '1':
                            is_soul_dative = True
                        else:
                            for neighb_token in parsing0[itoken-1: itoken+2]:
                                if neighb_token.upos in ('ADJ', 'DET') and get_attr(neighb_token, 'Gender') == 'Fem':
                                    is_soul_dative = True
                                    break

                        if is_soul_dative:
                            parsing.append(UDPipeToken(token, upos='NOUN', tags=['Case=Dat']))
                        else:
                            parsing.append(UDPipeToken(token))

                    elif utoken in ['чтоб']:
                        # Исправляем ошибки разметки некоторых слов в UDPipe.Syntagrus
                        parsing.append(UDPipeToken(token, upos='SCONJ', tags=[]))
                    elif utoken in ['средь']:
                        parsing.append(UDPipeToken(token, upos='ADP', tags=[]))
                    elif token.upos == 'PROPN' and token.form[0] == utoken[0]:
                        # UDPipe с Синтагрусом вдруг начал считать некоторые слова именами собственными, в данном примере "вэ"
                        # Сделаем их обратно существительными, если с прописной буквы.
                        # си́дючи в вэ ка́
                        #          ^^
                        token.upos = 'NOUN'
                        parsing.append(UDPipeToken(token))
                    elif utoken == 'белей' and token.upos == 'NOUN':
                        # Все белей и белей седина
                        parsing.append(UDPipeToken(token, upos='ADJ', tags=['Degree=Cmp']))
                    elif utoken == 'мимо' and token.upos == 'ADJ':
                        # 27.12.2022
                        # А эти парни все мимо и мимо.
                        #                 ^^^^   ^^^^
                        token.upos = 'ADV'
                        parsing.append(UDPipeToken(token))
                    else:
                        parsing.append(UDPipeToken(token))

                for token in parsing:
                    uform = token.form.lower()
                    if uform in self.word2lemma:
                        token.lemma = self.word2lemma[uform]

                # 03.12.2022
                # Новая модель syntagrus почему-то стала разбивать "полгода" на 2 токена.
                # Сольем их обратно вместе?
                if 'полгода' in parsing0.text.lower():
                    for i in range(len(parsing)-1):
                        if parsing[i].upos == 'NUM' and parsing[i].form.lower() == 'пол' and parsing[i+1].upos == 'NOUN' and parsing[i+1].get_attr('Animacy') == 'Inan':
                            parsing[i+1].form = parsing[i].form + parsing[i+1].form
                            parsing = parsing[:i] + parsing[i+1:]
                            break

                parsings.append(Parsing(parsing, parsing0.text))
        except:
            return None

        return parsings


if __name__ == '__main__':
    parser = UdpipeParser()
    parser.load('/home/inkoziev/polygon/text_generator/models')

    parsing = parser.parse_text('Твоей душе испорченной')[0]
    for token in parsing:
        print('{} {} {}'.format(token.form, token.upos, token.tags))


