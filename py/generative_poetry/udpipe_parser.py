import os

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


class UdpipeParser:
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.error = None

    def load(self, model_path):
        if os.path.isfile(model_path):
            udp_model_file = model_path
        else:
            udp_model_file = os.path.join(model_path, 'udpipe_syntagrus.model')

        self.model = Model.load(udp_model_file)
        self.pipeline = Pipeline(self.model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        self.error = ProcessingError()

    def parse_text(self, text):
        parsings = []

        processed = self.pipeline.process(text, self.error)
        if self.error.occurred():
            return None
        try:
            for parsing0 in pyconll.load_from_string(processed):
                parsing = []
                for token in parsing0:
                    utoken = token.form.lower()
                    if utoken in ['чтоб']:
                        # Исправляем ошибки разметки некоторых слов в UDPipe.Syntagrus
                        parsing.append(UDPipeToken(token, upos='SCONJ', tags=[]))
                    elif utoken in ['средь']:
                        parsing.append(UDPipeToken(token, upos='ADP', tags=[]))
                    else:
                        parsing.append(UDPipeToken(token))
                parsings.append(parsing)
        except:
            return None

        return parsings
