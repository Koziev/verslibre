# -*- coding: utf-8 -*-


def decode_pos(pos):
    if pos in [u'ДЕЕПРИЧАСТИЕ', u'ГЛАГОЛ', u'ИНФИНИТИВ']:
        return u'ГЛАГОЛ'
    else:
        return pos
