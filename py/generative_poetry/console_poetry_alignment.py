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
    tmp_dir = '../../tmp'
    data_dir = '../../data'
    models_dir = '../../models'

    #parser = UdpipeParser()
    #parser.load(models_dir)
    parser = StanzaParser()

    accents = Accents()
    accents.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
    accents.after_loading(stress_model_dir=os.path.join(tmp_dir, 'stress_model'))

    aligner = PoetryStressAligner(parser, accents, data_dir=os.path.join(data_dir, 'poetry', 'dict'))

    # Текст для разметки.
    poem = """В Круг Малый и Большой
принимают тех, кто Любы
независимо от нации и
отличительности прочей""".split('\n')

    a = aligner.align(poem)

    print('score={} meter={}'.format(a.score, a.meter))
    print(a.get_stressed_lines())
