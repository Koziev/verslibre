"""
08.05.2022 Добавлено приведение "ё" к "е", чтобы ловить плохие хайку типа:

                в клетке орёл
                в клетке решка
                в клетке орел
"""

import re


def remove_trailing_punct(s):
    while s and s[-1] in '.?,!:;…-':
        s = s[:-1].strip()
    return s


def is_good_haiku(s):
    # 01.01.2022 отбраковываем генерации, в которых три строки повторяются:
    # Лягушка прыгнула в пруд.
    # Лягушка прыгнула в пруд.
    # Лягушка прыгнула в пруд…
    lines = [remove_trailing_punct(x.strip()).replace('ё', 'е') for x in s.split('\n')]
    if len(lines) != 3:
        # 12.11.2022 ожитдаем всегда 3 строчки
        return False

    if lines[0] == lines[1] or lines[1] == lines[2] or lines[0] == lines[2]:
        return False

    for line in lines:
        # Ловим повторы типа: И солнце и солнце.
        #                       ^^^^^^^^^^^^^^^
        tokens = re.split(r'[.?,!:;…\-\s]', line)
        for t1, t2, t3 in zip(tokens, tokens[1:], tokens[2:]):
            if t1 == t3 and t1 not in ('еще', 'ещё', 'снова', 'вновь', 'сильнее') and t2 == 'и':
                return False

        # Ловим повторы типа XXX XXX
        for t1, t2 in zip(tokens, tokens[1:]):
            if t1 == t2:
                return False

    return True
