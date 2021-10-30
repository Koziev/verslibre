"""
Модель для генерации коротких стихов без рифмовки по заданной затравке (теме).
"""

import os
import argparse
import collections

import rutokenizer

from rugpt_generator import RugptGenerator


def get_last_word(tokens):
    if tokens:
        last_token = tokens[-1]
        if last_token in [',', '-', '.', '!', '?', ';']:
            return get_last_word(tokens[:-1])
        else:
            return last_token.lower()
    else:
        return ''


def is_good_poem(poem_text):
    last_words = collections.Counter()
    for poem_line in [l.strip() for l in poem_text.split('\n')]:
        tokens = tokenizer.tokenize(poem_line)
        w = get_last_word(tokens)
        last_words[w] += 1

    # если одно из слов встречается минимум дважды в конце строки (не считая пунктуации),
    # то вернем False.
    return last_words.most_common(1)[0][1] == 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verslibre generator')
    parser.add_argument('--topic', type=str)

    args = parser.parse_args()
    topic = args.topic

    tokenizer = rutokenizer.Tokenizer()
    tokenizer.load()

    models_dir = '../../models'

    poem_generator = RugptGenerator()
    poem_generator.load(os.path.join(models_dir, 'rugpt_poem_generator'))

    caption_generator = RugptGenerator()
    caption_generator.load(os.path.join(models_dir, 'rugpt_caption_generator'))

    while True:
        if topic:
            q = topic
        else:
            q = input(':> ').strip()

        if q:
            context = q

            px = poem_generator.generate_output(q)
            for ipoem, p in enumerate(px, start=1):
                if '\n' in p:
                    if is_good_poem(p):
                        captions = caption_generator.generate_output(p)
                        caption = captions[0]
                        print('POEM #{} for seed={}:'.format(ipoem, q))
                        print('--- {} ---\n'.format(caption))
                        print(p)
                        print('\n\n')

            print('')

        if topic:
            break

