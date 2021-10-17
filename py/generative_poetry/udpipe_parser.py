import os

import pyconll
from ufal.udpipe import Model, Pipeline, ProcessingError


class UdpipeParser:
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.error = None

    def load(self, models_dir):
        udp_model_file = os.path.join(models_dir, 'udpipe_syntagrus.model')
        self.model = Model.load(udp_model_file)
        self.pipeline = Pipeline(self.model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        self.error = ProcessingError()

    def parse_text(self, text):
        processed = self.pipeline.process(text, self.error)
        if self.error.occurred():
            return None
        try:
            parsings = pyconll.load_from_string(processed)
            if len(parsings) == 0:
                return None
        except:
            return None

        return parsings
