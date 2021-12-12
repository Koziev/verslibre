from generative_poetry.metre_classifier import get_syllables


def break_to_syllables(udpipe_parser, accentuator, sentence_text):
    res_tokens = []

    parsings = udpipe_parser.parse_text(sentence_text)
    if parsings is None:
        raise RuntimeError()

    for parsing in parsings:
        for ud_token in parsing:
            stress_pos = 0
            if ud_token.upos not in ('PUNCT', 'ADP', 'PART', 'SCONJ', 'CCONJ'):
                word = ud_token.form.lower()
                stress_pos = accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])

            sx = get_syllables(ud_token.form)

            if len(sx) < 2:
                if not any((c in 'уеыаоэёяию') for c in ud_token.form.lower()):
                    token = ud_token.form
                else:
                    if stress_pos == 1:
                        # Единственная гласная в этом слове под ударением
                        cx = []
                        for c in ud_token.form:
                            cx.append(c)
                            if c.lower() in 'уеыаоэёяию':
                               cx.append('\u0301')
                        #token = '[' + ''.join(cx) + ']'
                        token = ''.join(cx)
                    else:
                        #token = '[' + ud_token.form + ']'
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

                    #if i == 1:
                    #    token = '[' + syllable
                    #elif i == len(sx):
                    #    token = syllable + ']'
                    #else:
                    #    token = syllable
                    token_syllables.append(syllable)
                res_tokens.append(' '.join(token_syllables))

    #return ' '.join(res_tokens)
    return ' | '.join(res_tokens)


def break_to_syllables_v1(udpipe_parser, accentuator, sentence_text):
    res_tokens = []

    parsings = udpipe_parser.parse_text(sentence_text)
    if parsings is None:
        raise RuntimeError()

    for parsing in parsings:
        for ud_token in parsing:
            stress_pos = 0
            if ud_token.upos not in ('PUNCT', 'ADP', 'PART', 'SCONJ', 'CCONJ'):
                word = ud_token.form.lower()
                stress_pos = accentuator.get_accent(word, ud_tags=[(k + '=' + list(vx)[0]) for k, vx in ud_token.feats.items()] + [ud_token.upos])

            sx = get_syllables(ud_token.form)

            if len(sx) < 2:
                if not any((c in 'уеыаоэёяию') for c in ud_token.form.lower()):
                    token = ud_token.form
                else:
                    if stress_pos == 1:
                        # Единственная гласная в этом слове под ударением
                        cx = []
                        for c in ud_token.form:
                            cx.append(c)
                            if c.lower() in 'уеыаоэёяию':
                               cx.append('\u0301')
                        token = '[' + ''.join(cx) + ']'
                    else:
                        token = '[' + ud_token.form + ']'

                res_tokens.append(token)
            else:
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

                    if i == 1:
                        token = '[' + syllable
                    elif i == len(sx):
                        token = syllable + ']'
                    else:
                        token = syllable

                    res_tokens.append(token)

    return ' '.join(res_tokens)
