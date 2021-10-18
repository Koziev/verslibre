"""
Модель для генерации хайку с помощью ruGPT-модели.
"""

import io
import os
import argparse
import logging.handlers

import numpy as np

import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove, Update

from rugpt_generator import RugptGenerator
from antiplagiat import Antiplagiat
from poetry_seeds import generate_seeds
from init_logging import init_logging


def v_cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom > 0:
        return np.dot(a, b) / denom
    else:
        return 0


def get_user_id(update: Update) -> str:
    user_id = str(update.message.from_user.id)
    return user_id


LIKE = 'Нравится!'
DISLIKE = 'Плохо :('
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
                             text='Привет, {}!\nЗадавайте тему в виде словосочетания из прилагательного и существительного.\nЛибо выберите готовую тему из предложенных'.format(update.message.from_user.full_name),
                             reply_markup=reply_markup)
    logging.debug('Leaving START callback with user_id=%s', user_id)


def echo(update, context):
    # update.chat.first_name
    # update.chat.last_name
    try:
        user_id = get_user_id(update)

        if update.message.text == MORE:
            # Выведем следующее из уже сгенерированных
            m = last_user_poems[user_id][0]
            last_user_poem[user_id] = m
            last_user_poems[user_id] = last_user_poems[user_id][1:]

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

        if update.message.text == LIKE:
            # Какой текст полайкали:
            caption = last_user_poem[user_id][0]
            haiku = last_user_poem[user_id][1].replace('\n', ' | ')
            logging.info('LIKE: caption="%s" haiku="%s"', caption, haiku)
            context.bot.send_message(chat_id=update.message.chat_id, text="Спасибо :)")

            last_user_poem[user_id] = None
            last_user_poems[user_id] = []

            return

        if update.message.text == DISLIKE:
            # Какой текст не понравился:
            caption = last_user_poem[user_id][0]
            haiku = last_user_poem[user_id][1].replace('\n', ' | ')
            logging.info('DISLIKE: caption="%s" haiku="%s"', caption, haiku)
            context.bot.send_message(chat_id=update.message.chat_id, text="Понятно :(")

            last_user_poem[user_id] = None
            last_user_poems[user_id] = []

            return

        context.bot.send_message(chat_id=update.message.chat_id, text='Минуточку...')

        q = update.message.text
        logging.info('Will generate a haiku using seed="%s" for user="%s" id=%s in chat=%s', q, update.message.from_user.name, user_id, str(update.message.chat_id))

        haikux = haiku_generator.generate_output(q, num_return_sequences=5)
        # px = ranker.rerank(q, px)

        haikux2 = []

        for ipoem, haiku in enumerate(haikux, start=1):
            if '|' in haiku:
                haiku = haiku.replace(' | ', '\n')

            if haiku.count('\n') == 2:
                #captions = caption_generator.generate_output(haiku)
                #caption = captions[0]
                caption = '***'
                p_plagiat = antiplagiat.score(haiku)
                logging.info('HAIKU #%d for seed="%s" user_id=%s p_plagiat=%5.3f: %s', ipoem, q, user_id, p_plagiat, haiku.replace('\n', ' | '))
                if p_plagiat < 0.90:
                    haikux2.append((caption, haiku))

        last_user_poems[user_id] = []
        last_user_poem[user_id] = None

        for ipoem, (caption, haiku) in enumerate(haikux2, start=0):
            msg = '--- {} ---\n\n{}'.format(caption, haiku)
            if ipoem == 1:
                last_user_poem[user_id] = (caption, haiku, msg)
            else:
                last_user_poems[user_id].append((caption, haiku, msg))

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Telegram chatbot')
    parser.add_argument('--token', type=str, default='', help='Telegram token')
    parser.add_argument('--mode', type=str, default='console', choices='console telegram'.split())
    parser.add_argument('--models_dir', type=str, default='../../models')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--log', type=str, default='../../tmp/haiku_generator.log')

    args = parser.parse_args()

    mode = args.mode

    models_dir = os.path.expanduser(args.models_dir)
    dataset_path = os.path.expanduser(args.dataset_path) if args.dataset_path else None

    #logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #logging.getLogger().setLevel(logging.DEBUG)
    #logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    init_logging(args.log, True)

    # Для работы с сервером телеграмма нужен зарегистрированный бот.
    # Результатом регистрации является токен - уникальная строка символов.
    # Вот эту строку надо сейчас ввести с консоли.
    # Возможно, следует предусмотреть передачу токена через опцию ком. строки.
    if args.mode == 'telegram':
        telegram_token = args.token
        if len(telegram_token) == 0:
            telegram_token = input('Enter Telegram token:> ').strip()

    #tokenizer = rutokenizer.Tokenizer()
    #tokenizer.load()

    #ranker = TopicRanker(tokenizer)
    #ranker.load("/home/inkoziev/polygon/w2v/fasttext.CBOW=1_WIN=5_DIM=64")

    antiplagiat = Antiplagiat()
    if dataset_path:
        logging.info('Initializing antiplagiat module with dataset "%s"', dataset_path)
        with io.open(dataset_path, 'r', encoding='utf-8') as rdr:
            lines = []
            for line in rdr:
                s = line.strip()
                if s:
                    if s.startswith('<s>'):
                        s = s[s.index('#')+1:]

                    lines.append(s.replace('<s>', '').replace('</s>', '').strip())
                    if s.endswith('</s>'):
                        text = '\n'.join(lines)
                        antiplagiat.add_document(text)
                        lines = []
    else:
        logging.error('Antiplagiat dataset is not available')

    logging.info('Loading the models from "%s"...', models_dir)
    haiku_generator = RugptGenerator()
    haiku_generator.load(os.path.join(models_dir, 'rugpt_haiku_generator'))

    #caption_generator = RugptGenerator()
    #caption_generator.load(os.path.join(models_dir, 'rugpt_caption_generator'))

    if mode == 'telegram':
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
            q = input(':> ').strip()
            if q:
                px = haiku_generator.generate_output(q)
                #px = ranker.rerank(q, px)
                for ipoem, haiku in enumerate(px, start=1):
                    if '|' in haiku:
                        haiku = haiku.replace(' | ', '\n')

                    if haiku.count('\n') == 2:
                        #captions = caption_generator.generate_output(haiku)
                        #caption = captions[0]
                        caption = '***'
                        print('HAIKU #{} for seed={}:'.format(ipoem, q))
                        p_plagiat = antiplagiat.score(haiku)
                        print('p_plagiat={}'.format(p_plagiat))
                        print('--- {} ---\n'.format(caption))
                        print(haiku)
                        print('\n\n')

                print('')
