"""
Вспомогательная утилитка для обработки одного стиха поэтическим транскриптором. Результаты
работы транскриптора выводятся в консоль для визуального контроля.
"""

import os

from poetry.phonetic import Accents
from generative_poetry.stanza_parser import StanzaParser
from generative_poetry.poetry_alignment import PoetryStressAligner
from generative_poetry.udpipe_parser import UdpipeParser


if __name__ == '__main__':
    proj_dir = os.path.expanduser('~/polygon/text_generator')
    tmp_dir = os.path.join(proj_dir, 'tmp')
    data_dir = os.path.join(proj_dir, 'data')
    models_dir = os.path.join(proj_dir, 'models')

    parser = UdpipeParser()
    parser.load(models_dir)
    #parser = StanzaParser()

    accents = Accents()
    accents.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
    accents.after_loading(stress_model_dir=os.path.join(tmp_dir, 'stress_model'))

    aligner = PoetryStressAligner(parser, accents, data_dir=os.path.join(data_dir, 'poetry', 'dict'))

    # Текст для разметки.
    poem = """С тобой бок о бок я страдаю""".split('\n')
    a = aligner.align(poem)

    print('score={} meter={} scheme={}'.format(a.score, a.meter, a.rhyme_scheme))
    print(a.get_stressed_lines())

    #encoded_lines = a.split_to_syllables(do_arabize=True)
    #print('Encoding:\n{}'.format('\n'.join(encoded_lines)))

    r = aligner.detect_repeating(a)
    p = aligner.detect_poor_poetry(a)
    print('Repetition={} Poor={}'.format(r, p))

    print('Signatures:')
    for pline in a.poetry_lines:
        print('{}\t{}'.format(str(pline), pline.stress_signature_str))
