import random

from generative_poetry.metre_classifier import get_syllables


# небольшой хак - список слов с опциональным ударением в стихах.
opt_words = ['лишь', 'вроде', 'если', 'чтобы', 'когда', 'просто', 'мимо', 'даже', 'всё', 'хотя', 'едва', 'нет',
             'эти', 'эту', 'это', 'мои', 'твои', 'моих', 'твоих', 'моим', 'твоим', 'моей', 'твоей',
             'мою', 'твою', 'его', 'ее', 'её', 'себе', 'тебя', 'свою', 'свои', 'своим', 'они', 'она',
             'уже', 'есть', 'раз', 'быть']


def break_to_syllables(udpipe_parser, accentuator, sentence_text):
    res_tokens = []

    for c in '\'.‚,?!:;…-–—⸺«»″”“„‘’`ʹ"˝[]‹›·<>*/=()+®©‛¨×№\u05f4≤≥':
        sentence_text = sentence_text.replace(c, ' ' + c + ' ').replace('  ', ' ')

    parsings = udpipe_parser.parse_text(sentence_text)
    if parsings is None:
        raise RuntimeError()

    for parsing in parsings:
        for ud_token in parsing:
            stress_pos = 0
            word = ud_token.form.lower()
            nvowels = sum((c in 'уеыаоэёяию') for c in word)

            # для слов из списка ударение опционально, его мы будем выставлять рандомно, т.е. ставить или нет,
            # с вероятностью 0.5
            is_optional_stress = False

            if ud_token.upos in ('PRON', 'ADV', 'DET') and nvowels == 1:
                # Односложные наречия, местоимения и т.д.
                is_optional_stress = True
            elif ud_token.upos in ('PUNCT', 'ADP', 'PART', 'SCONJ', 'CCONJ', 'INTJ'):
                is_optional_stress = True
            elif word in opt_words:
                is_optional_stress = True

            if word in accentuator.ambiguous_accents2:
                # для слов типа пОнял-понЯл, где есть варианты ударения для одной формы,
                # берем первый вариант в 70% случаев, а в 30% - второй. Подразумевается,
                # что в словаре ambiguous_accents2.json первый вариант ударения - основной.
                stress_pos = accentuator.ambiguous_accents2[word][0] if random.random() <= 0.70 else accentuator.ambiguous_accents2[word][1]
                if stress_pos == -1:
                    raise RuntimeError()
            elif not is_optional_stress:
                stress_pos = accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])
            else:
                if random.random() > 0.5:
                    stress_pos = accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])

            sx = get_syllables(ud_token.form)

            if len(sx) < 2:
                # Односложное слово.

                if nvowels == 0:
                    # гласных нет.
                    # TODO - это может быть аббревиатура, с этим надо что-то делать!
                    token = ud_token.form
                else:
                    if stress_pos == 1:
                        # Единственная гласная в этом слове под ударением
                        cx = []
                        for c in ud_token.form:
                            cx.append(c)
                            if c.lower() in 'уеыаоэёяию':
                               cx.append('\u0301')
                        token = ''.join(cx)
                    else:
                        token = ud_token.form

                res_tokens.append(token)
            else:
                token_syllables = []
                for i, syllable in enumerate(sx, start=1):
                    syllable = syllable.text

                    if i == stress_pos:
                        # В этом слоге надо проставить знак ударения на гласной
                        cx = []
                        for c in syllable:
                            cx.append(c)
                            if c.lower() in 'уеыаоэёяию':
                               cx.append('\u0301')
                        syllable = ''.join(cx)

                    token_syllables.append(syllable)
                res_tokens.append(' '.join(token_syllables))

    return ' | '.join(res_tokens)


def prose_markup(parser, accentuator, text):
    res_lines = []

    for line in text.split('\n'):
        res_tokens = []
        parsings = parser.parse_text(line)
        if parsings is None:
            raise RuntimeError()

        for parsing in parsings:
            for ud_token in parsing:
                stress_pos = 0
                word = ud_token.form.lower()
                nvowels = sum((c in 'уеыаоэёяию') for c in word)

                # для слов из списка ударение опционально, его мы будем выставлять рандомно, т.е. ставить или нет,
                # с вероятностью 0.5
                is_optional_stress = False

                if ud_token.upos in ('PRON', 'ADV', 'DET') and nvowels == 1:
                    # Односложные наречия, местоимения и т.д.
                    is_optional_stress = True
                elif ud_token.upos in ('PUNCT', 'ADP', 'PART', 'SCONJ', 'CCONJ', 'INTJ'):
                    is_optional_stress = True
                elif word in opt_words:
                    is_optional_stress = True

                if not is_optional_stress:
                    stress_pos = accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])
                else:
                    if random.random() > 0.5:
                        stress_pos = accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])

                cx = []
                vowel_counter = 0
                for c in ud_token.form:
                    cx.append(c)
                    if c.lower() in 'уеыаоэёяию':
                        vowel_counter += 1
                        if vowel_counter == stress_pos:
                            cx.append('\u0301')
                token2 = ''.join(cx)

                res_tokens.append(token2)
        res_lines.append(' '.join(res_tokens))

    return '\n'.join(res_lines)
