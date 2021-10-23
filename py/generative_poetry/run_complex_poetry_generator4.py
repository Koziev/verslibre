"""
Третья версия генератора рифмованных четверостиший из одних GPT.
Первые две строки берем из эталонных стихов.
"""

import io
import pickle
import os
import collections
import math
import random
import argparse
import logging

import numpy as np
import jellyfish

import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove, Update

import rutokenizer
import ruword2tags

from poetry.phonetic import Accents
from select_rhymed_words import RhymeSelector
from rugpt_generator import RugptGenerator
from antiplagiat import Antiplagiat
from udpipe_parser import UdpipeParser
from meter_renderer import UdpipeMeterRenderer
from poetry_seeds import generate_seeds
from init_logging import init_logging


def signs2str(signs):
    return ' '.join(map(str, signs))


def ngrams(s, n):
    return set(u''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    #if len(shingles1) == 0 or len(shingles2) == 0:
    #    print('ERROR@135 s1="{}" s2="{}"'.format(s1, s2))
    #    exit(0)

    return float(len(shingles1 & shingles2))/float(len(shingles1 | shingles2) + 1e-6)


def get_attr(token, tag_name):
    if tag_name in token.feats:
        v = list(token.feats[tag_name])[0]
        return v

    return ''


def v_cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom > 0:
        return np.dot(a, b) / denom
    else:
        return 0


def get_last_word(tokens):
    if tokens:
        last_token = tokens[-1]
        if last_token in [',', '-', '.', '!', '?', ';', '…', '–', '—']:
            return get_last_word(tokens[:-1])
        else:
            return last_token.lower()
    else:
        return ''


def is_good_poem(poem_text):
    last_words = collections.Counter()
    for poem_line in [l.strip() for l in poem_text.split('|')]:
        tokens = tokenizer.tokenize(poem_line)
        last_words[get_last_word(tokens)] += 1

    # если одно из слов встречается минимум дважды в конце строки (не считая пунктуации),
    # то вернем False.
    return last_words.most_common(1)[0][1] == 1


def estimate_line_quality(line):
    """
    Возвращает оценку того, насколько хорош лексически текст в строке line.
    В частности, штрафуем за употребление местоимений, кроме личных Я и ТЫ.
    """
    bad_line_words = "он она оно они ее его им ей нее неё него них ними ею нею мы вы нас вас нами вами ими их ней нём нему ним".split()
    n_bad = sum((word.lower() in bad_line_words) for word in tokenizer.tokenize(line))
    return 1.0 / (1.0 + n_bad)


vowels = "уеыаоэёяию"
def count_vowels(text):
    return sum((c.lower() in vowels) for c in text)


def meter_dist(meter1, meter2):
    return 1.0 - jellyfish.damerau_levenshtein_distance(''.join(map(str, meter1)), ''.join(map(str, meter2))) / float(len(meter1))
    #diff = sum((c1!=c2) for c1, c2 in zip(meter1, meter2))
    #return 1.0 - diff / float(len(meter1))


def get_user_id(update: Update) -> str:
    user_id = str(update.message.from_user.id)
    return user_id


LIKE = 'Нравится!'
DISLIKE = 'Плохо :('
NEW = 'Новая тема'
MORE = 'Еще...'

last_user_poems = dict()
last_user_poem = dict()


def start(update, context) -> None:
    user_id = get_user_id(update)
    logging.debug('Entering START callback with user_id=%s', user_id)

    seeds = generate_seeds(user_id)

    keyboard = [seeds]
    reply_markup = ReplyKeyboardMarkup(keyboard,
                                       one_time_keyboard=True,
                                       resize_keyboard=True,
                                       per_user=True)

    context.bot.send_message(chat_id=update.message.chat_id,
                             text="Привет, {}!\n".format(update.message.from_user.full_name)+\
                                  "Я бот для генерации стихов. Мои исходники можно найти в https://github.com/Koziev/verslibre.\n"+\
                                  "Задавайте тему в виде словосочетания из прилагательного и существительного.\n"+\
                                  "Либо выберите готовую тему из предложенных",
                             reply_markup=reply_markup)
    logging.debug('Leaving START callback with user_id=%s', user_id)


def echo(update, context):
    # update.chat.first_name
    # update.chat.last_name
    try:
        user_id = get_user_id(update)

        if update.message.text == NEW:
            keyboard = [generate_seeds(user_id)]
            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)
            context.bot.send_message(chat_id=update.message.chat_id,
                                     text="Выберите тему из трёх предложенных или введите свою",
                                     reply_markup=reply_markup)
            return

        if update.message.text == LIKE:
            # Какой текст полайкали:
            caption = last_user_poem[user_id][0]
            poem = last_user_poem[user_id][1].replace('\n', ' | ')
            logging.info('LIKE: caption="%s" poem="%s" user="%s"', caption, poem, user_id)

            if len(last_user_poems[user_id]):
                keyboard = [[NEW, MORE]]
            else:
                keyboard = [[NEW]]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id, text="Спасибо :)", reply_markup=reply_markup)
            return

        if update.message.text == DISLIKE:
            # Какой текст не понравился:
            caption = last_user_poem[user_id][0]
            poem = last_user_poem[user_id][1].replace('\n', ' | ')
            logging.info('DISLIKE: caption="%s" poem="%s" user="%s"', caption, poem, user_id)

            if len(last_user_poems[user_id]):
                keyboard = [[NEW, MORE]]
            else:
                keyboard = [[NEW]]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id, text="Понятно. Жаль :(", reply_markup=reply_markup)
            return

        if update.message.text == MORE:
            # Выведем следующее из уже сгенерированных
            caption, poem, msg = last_user_poems[user_id][-1]
            if caption == '***':
                encoded_poem = poem.replace('\n', ' | ')
                logging.debug('Generating caption for poem="%s" user="%s"', encoded_poem, user_id)
                captions = caption_generator.generate_output(encoded_poem, num_return_sequences=1)
                caption = captions[0]
                logging.debug('Caption generation: poem="%s" caption="%s" user="%s"', encoded_poem, caption, user_id)
                msg = '--- {} ---\n\n{}'.format(caption, poem)

            last_user_poem[user_id] = (caption, poem, msg)
            last_user_poems[user_id] = last_user_poems[user_id][:-1]

            if len(last_user_poems[user_id]):
                keyboard = [[LIKE, DISLIKE, MORE]]
            else:
                keyboard = [[LIKE, DISLIKE], generate_seeds(user_id)]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id, text=last_user_poem[user_id][2],
                                     reply_markup=reply_markup)

            return

        msg = random.choice(['Минуточку, или лучше две...', 'Ок, сажусь писать...', 'Хорошо, буду сочинять...',
                             'Понял, приступаю...', 'Отлично, сейчас что-нибудь придумаю...'])
        context.bot.send_message(chat_id=update.message.chat_id, text=msg)

        seed = update.message.text
        logging.info('Will generate a poem using seed="%s" for user="%s" id=%s in chat=%s', seed, update.message.from_user.name, user_id, str(update.message.chat_id))

        poems2 = generate_poems(seed)

        last_user_poems[user_id] = []
        last_user_poem[user_id] = None

        for ipoem, (caption, poem, score) in enumerate(poems2, start=0):
            if ipoem == 1:
                # Для первой поэмы сразу сгенерируем заголовок
                if caption == '***':
                    encoded_poem = poem.replace('\n', ' | ')
                    logging.debug('Generating caption for poem="%s" seed="%s" user="%s"', encoded_poem, seed, user_id)
                    captions = caption_generator.generate_output(encoded_poem, num_return_sequences=1)
                    caption = captions[0]
                    logging.debug('Caption generation: seed="%s" poem="%s" caption="%s" user_id="%s"', seed, encoded_poem, caption, user_id)

                msg = '--- {} ---\n\n{}'.format(caption, poem)

                last_user_poem[user_id] = (caption, poem, msg)
            else:
                msg = '--- {} ---\n\n{}'.format(caption, poem)
                last_user_poems[user_id].append((caption, poem, msg))

        if last_user_poem[user_id]:
            if len(last_user_poems[user_id]):
                keyboard = [[LIKE, DISLIKE, MORE]]
            else:
                keyboard = [[LIKE, DISLIKE], generate_seeds(user_id)]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id,
                                     text=last_user_poem[user_id][2],
                                     reply_markup=reply_markup)
        else:
            keyboard = [generate_seeds(user_id)]
            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id,
                                     text='Что-то не получается сочинить :(\nЗадайте другую тему, пожалуйста',
                                     reply_markup=reply_markup)

    except Exception as ex:
        logging.error('Error in "echo"')
        logging.error(ex)  # sys.exc_info()[0]


def generate_poems(seed):
    logging.info('Start generating for seed="%s"', seed)
    generated_poems = []
    all_texts = set()
    prototypes = headlines_selector.find_nearest(seed, 10)
    while len(generated_poems) < 20:
        try:
            poem_lines = random.choice(prototypes).split('\n')

            # Берем первые 2 строки
            head_lines = poem_lines[:2]

            meter1 = renderer.render_meter(head_lines[0])
            meter2 = renderer.render_meter(head_lines[1])

            # Подбираем рифмы к первым двум строкам
            rhyme_pairs = []
            if False:
                rhymes1 = rselector.get_rhymes(get_last_word(tokenizer.tokenize(head_lines[0])))[:5]
                rhymes2 = rselector.get_rhymes(get_last_word(tokenizer.tokenize(head_lines[1])))[:5]

                for rhyme1, score1 in rhymes1:
                    for rhyme2, score2 in rhymes2:
                        rhyme_pairs.append((rhyme1, rhyme2, score1 + score2))

                rhyme_pairs = sorted(rhyme_pairs, key=lambda z: random.random())[:10]
            else:
                w3 = get_last_word(tokenizer.tokenize(poem_lines[2]))
                w4 = get_last_word(tokenizer.tokenize(poem_lines[3]))
                rhyme_pairs = [(w3, w4, 1.0)]

            if len(rhyme_pairs) > 0:
                rhyme1, rhyme2, rhyme_score = random.choice(rhyme_pairs)

                input_context = []

                input_context.append(poem_lines[0])  # первая строка
                input_context.append('[ ' + signs2str(renderer.render_meter(poem_lines[0])) + ' ]')  # метрика первой строки
                input_context.append(';')

                input_context.append(poem_lines[1])  # вторая строка
                input_context.append('[ ' + signs2str(renderer.render_meter(poem_lines[1])) + ' ]')  # метрика второй строки
                input_context.append(';')

                input_context.append('[ ' + signs2str(renderer.render_word(rhyme1)) + ' ]')
                input_context.append(rhyme1)
                input_context.append(';')

                input_context.append('[ ' + signs2str(renderer.render_word(rhyme2)) + ' ]')
                input_context.append(rhyme2)

                filler_input = ' '.join(input_context)

                #logging.debug('Running slot_filler with "{}"...'.format(filler_input))
                lines34x = slot_filler.generate_output(filler_input, num_return_sequences=4)
                for lines34 in lines34x:
                    lines34 = [x.strip() for x in lines34.split('\n')]
                    if len(lines34) == 2:
                        poem_score = 1.0

                        # Делаем оценку качества метрики получившегося стиха
                        meter3 = renderer.render_meter(lines34[0])
                        meter4 = renderer.render_meter(lines34[1])

                        p_13 = meter_dist(meter1, meter3)
                        p_24 = meter_dist(meter2, meter4)

                        vowels1 = count_vowels(poem_lines[0])
                        vowels2 = count_vowels(poem_lines[1])
                        vowels3 = count_vowels(lines34[0])
                        vowels4 = count_vowels(lines34[1])
                        p_13 *= math.exp(-(vowels1 - vowels3) * (vowels1 - vowels3))
                        p_24 *= math.exp(-(vowels2 - vowels4) * (vowels2 - vowels4))

                        p_3 = estimate_line_quality(lines34[0])
                        p_4 = estimate_line_quality(lines34[1])

                        poem_score = p_13 * p_24 * p_3 * p_4  # * rhyme_score

                        final_text = '\n'.join([poem_lines[0], poem_lines[1], lines34[0], lines34[1]])
                        if final_text not in all_texts:
                            caption = '***'
                            generated_poems.append((caption, final_text, poem_score))
                            all_texts.add(final_text)
        except KeyboardInterrupt:
            logging.error('Keyboard interrupt.')
            break

    # Отсортировать собранные стихи по убыванию скора, оставить несколько лучших
    best_poems = []
    logging.debug('=== BEST POEMS ===')
    for caption, poem_text, score in sorted(generated_poems, key=lambda z: -z[2])[:4]:
        logging.debug('# score=%5.3f', score)
        logging.debug('--- %s ---', caption)
        logging.debug('%s', poem_text.replace('\n', ' | '))
        best_poems.append((caption, poem_text, score))

    return best_poems


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verslibre generator v.4')
    parser.add_argument('--token', type=str, default='', help='Telegram token')
    parser.add_argument('--mode', type=str, default='console', choices='console telegram'.split())
    parser.add_argument('--tmp_dir', default='../../tmp', type=str)
    parser.add_argument('--models_dir', default='../../models', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--log', type=str, default='../../tmp/haiku_generator.log')

    args = parser.parse_args()
    mode = args.mode
    tmp_dir = os.path.expanduser(args.tmp_dir)
    models_dir = os.path.expanduser(args.models_dir)
    dataset_path = os.path.expanduser(args.dataset_path) if args.dataset_path else None

    init_logging(args.log, True)

    udpipe = UdpipeParser()
    udpipe.load(models_dir)

    #t2f = transcriber.Text2Phonems()
    #t2f.load()

    # Гибридный ударятор - словарь ударений + нейросетевая модель для oov слов
    accents = Accents()
    accents.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
    accents.after_loading(stress_model_dir=os.path.join(tmp_dir, 'stress_model'))

    renderer = UdpipeMeterRenderer(udpipe, accents)

    tokenizer = rutokenizer.Tokenizer()
    tokenizer.load()

    gren = ruword2tags.RuWord2Tags()
    gren.load()

    logging.info('Loading the models from "%s"...', models_dir)

    fasttext_model = None  #fasttext.load_model("/home/inkoziev/polygon/w2v/fasttext.CBOW=1_WIN=5_DIM=64")

    # Модель подбора рифмующихся слов
    rselector = RhymeSelector(accents, fasttext_model, gren, tokenizer)
    rselector.load_udpipe(os.path.join(models_dir, 'udpipe_syntagrus.model'))
    rselector.load_pickle(tmp_dir)

    # Генератор заголовка стиха по тексту стиха
    logging.info('Loading GPT caption generator...')
    caption_generator = RugptGenerator()
    caption_generator.load(os.path.join(models_dir, 'rugpt_caption_generator'))

    headlines_selector_path = os.path.join(tmp_dir, 'headlines_selector.pkl')
    if os.path.exists(headlines_selector_path):
        with open(headlines_selector_path, 'rb') as f:
            headlines_selector = pickle.load(f)
    else:
        headlines_selector = Antiplagiat()
        with io.open(os.path.join(tmp_dir, 'poetry_corpus.readable.txt'), 'r') as rdr:
            lines = []
            for line in rdr:
                s = line.strip()
                if s:
                    lines.append(s)
                else:
                    if len(lines) == 4:
                        last_word3 = get_last_word(tokenizer.tokenize(lines[2]))
                        last_word4 = get_last_word(tokenizer.tokenize(lines[3]))
                        doc = '\n'.join([lines[0], lines[1], last_word3, last_word4])
                        headlines_selector.add_document(doc)

                    lines = []

        with open(headlines_selector_path, 'wb') as f:
            pickle.dump(headlines_selector, f)

    # Генератор двух строк, заканчивающихся заданными рифмами
    logging.info('Loading GPT slot filler...')
    slot_filler = RugptGenerator()
    slot_filler.load(os.path.join(models_dir, 'rugpt_slot_filler'))

    # Для работы с сервером телеграмма нужен зарегистрированный бот.
    # Результатом регистрации является токен - уникальная строка символов.
    # Вот эту строку надо сейчас ввести с консоли.
    # Возможно, следует предусмотреть передачу токена через опцию ком. строки.
    if args.mode == 'telegram':
        telegram_token = args.token
        if len(telegram_token) == 0:
            telegram_token = input('Enter Telegram token:> ').strip()

    if args.mode == 'telegram':
        logging.info('Starting telegram bot')

        # Телеграм-версия генератора
        tg_bot = telegram.Bot(token=telegram_token).getMe()
        bot_id = tg_bot.name
        logging.info('Telegram bot "%s" id=%s', tg_bot.name, tg_bot.id)

        updater = Updater(token=telegram_token)
        dispatcher = updater.dispatcher

        start_handler = CommandHandler('start', start)
        dispatcher.add_handler(start_handler)

        echo_handler = MessageHandler(Filters.text, echo)
        dispatcher.add_handler(echo_handler)

        logging.getLogger('telegram.bot').setLevel(logging.INFO)
        logging.getLogger('telegram.vendor.ptb_urllib3.urllib3.connectionpool').setLevel(logging.INFO)

        logging.info('Start polling messages for bot %s', tg_bot.name)
        updater.start_polling()
        updater.idle()
    else:
        # Тестирование в консоли
        while True:
            seed = input(':> ').strip()
            generated_poems = generate_poems(seed)

            print('\n====== BEST POEMS ======\n')
            for caption, text, score in sorted(generated_poems, key=lambda z: -z[2])[:4]:
                print('# score={}'.format(score))

                if caption == '***':
                    captions = caption_generator.generate_output(text.replace('\n', ' | '), num_return_sequences=1)
                    caption = captions[0]

                print('--- {} ---'.format(caption))
                print(text)
                print('\n\n')
