import itertools


class UdpipeMeterRenderer:
    def __init__(self, udpipe_parser, accentuator):
        self.udpipe_parser = udpipe_parser
        self.accentuator = accentuator

    def render_by_tokens(self, sentence_text):
        """return: список кортежей, в которых первых элемент - токен UDPipe, второй - список ударности гласных"""
        signs = []

        parsings = self.udpipe_parser.parse_text(sentence_text)
        for parsing in parsings:
            for ud_token in parsing:
                word_signs = []
                if ud_token.upos != 'PUNCT':
                    word = ud_token.form.lower()
                    a = self.accentuator.get_accent(word, ud_tags=[(k + '=' + list(vx)[0]) for k, vx in ud_token.feats.items()])

                    n_vowels = 0
                    for c in word:
                        if c in 'уеыаоэёяию':
                            n_vowels += 1
                            if n_vowels == a:
                                word_signs.append(1)
                            else:
                                word_signs.append(0)

                signs.append((ud_token, word_signs))

        return signs

    def render_meter(self, text):
        signs = self.render_by_tokens(text)
        final_signs = list(itertools.chain(*[accent for token, accent in signs]))
        return final_signs

    def render_last_word(self, text):
        # Возвращаем метрику для последнего слова, причем с учетом его контекста (для неоднозначно ударяемых!).

        signs = self.render_by_tokens(text)

        # Отсекаем финальные пунктуаторы
        while signs[-1][0].upos == 'PUNCT':
            signs = signs[:-1]

        return signs[-1][1]

    def render_word(self, word0):
        word = word0.lower()
        word_signs = []
        a = self.accentuator.get_accent(word)
        n_vowels = 0
        for c in word:
            if c in 'уеыаоэёяию':
                n_vowels += 1
                if n_vowels == a:
                    word_signs.append(1)
                else:
                    word_signs.append(0)

        return word_signs
