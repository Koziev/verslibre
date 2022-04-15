"""
Модель для генерации хайку с помощью ruGPT-модели.

01.02.2022 Отбраковываем генерации, в которых три строки повторяются (без учета финальных пунктуаторов).
03.02.2022 Отрисовка текста хайку в HTML с fixed-font, чтобы визуально они лучше выделялись в чате
04.02.2022 Изменился API генерации затравок: теперь это класс SeedGenerator
15.04.2022 Эксперимент с переключением доменов генерации: хайку или бусидо
"""

import io
import os
import argparse
import logging.handlers
import re

import numpy as np

import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler
from telegram import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove, Update

from rugpt_generator import RugptGenerator
from antiplagiat import Antiplagiat
from poetry_seeds import SeedGenerator
from init_logging import init_logging
from is_good_haiku import is_good_haiku


def get_user_id(update: Update) -> str:
    user_id = str(update.message.from_user.id)
    return user_id


def render_haiku_html(haiku_txt):
    # 03.02.2022 для отрисовки текста сгенерированного хайку в HTML формате в телеграмме
    s = '<pre>' + haiku_txt + '</pre>'
    return s


LIKE = 'Нравится!'
DISLIKE = 'Плохо :('
MORE = 'Еще...'

user_format = dict()
last_user_poems = dict()
last_user_poem = dict()


def start(update, context) -> None:
    user_id = get_user_id(update)
    logging.debug('Entering START callback with user_id=%s', user_id)

    keyboard = [[InlineKeyboardButton('хайку', callback_data='format=' + 'хайку')],
                [InlineKeyboardButton('бусидо', callback_data='format=' + 'бусидо')],
                ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.send_message(chat_id=update.message.chat_id,
                             text="Привет, {}!\n\nЯ - бот для генерации <b>хайку</b> и <b>бусидо</b> (версия от 15.04.2022).\n\n".format(update.message.from_user.full_name) +\
                             "Для генерации стихов с рифмой используйте бот @verslibre_bot.\n"
                             "Если у вас есть вопросы - напишите мне kelijah@yandex.ru\n\n"
                             "Выберите, что будем сочинять:\n",
                             reply_markup=reply_markup, parse_mode='HTML')

    logging.debug('Leaving START callback with user_id=%s', user_id)


def format_menu(context, callback_data):
    """Пользователь в телеграм-боте выбрал целевой домен генерируемых текстов."""
    user_id = str(context.effective_user.id)
    format = callback_data.match.string.split('=')[1]

    # Запомним этот выбор для этого пользователя.
    user_format[user_id] = format
    logging.info('Target format set to "%s" for user_id="%s"', user_format[user_id], user_id)

    seeds = seed_generator.generate_seeds(user_id)
    keyboard = [seeds]
    reply_markup = ReplyKeyboardMarkup(keyboard,
                                       one_time_keyboard=True,
                                       resize_keyboard=True,
                                       per_user=True)

    if user_format[user_id] == 'хайку':
        help_text = 'Включен режим <b>хайку</b>. Если захотите выбрать другой формат стихов, введите команду <code>/start</code>.\n\n' \
                    'Хайку это короткое нерифмованное трехстишье, выражающие отстраненное восприятие пейзажа.\n\n' \
                    'Теперь вводите какое-нибудь существительное или сочетание прилагательного и существительного, например <i>весна</i>, ' \
                    'и я сочиню хайку с этими словами. '
    elif user_format[user_id] == 'бусидо':
        help_text = 'Включен режим <b>бусидо</b>. Если захотите выбрать другой формат стихов, введите команду <code>/start</code>.\n\n' \
                    'Бусидо - это короткий афористичный текст с привкусом восточной мудрости.\n\n' \
                    'Теперь вводите какое-нибудь существительное или сочетание прилагательного и существительного, например <i>долг</i>, ' \
                    'и я сочиню бусидо с этими словами. '
    else:
        raise NotImplementedError()

    help_text += 'Либо выберите готовую затравку из предложенных - см. кнопки внизу.\n\n' \
                 'Кнопка [<b>Ещё</b>] выведет новый вариант текста на заданную тему.'

    context.callback_query.message.reply_text(text=help_text, reply_markup=reply_markup, parse_mode='HTML')
    return


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
                keyboard = [[LIKE, DISLIKE], seed_generator.generate_seeds(user_id)]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id,
                                     text=render_haiku_html(last_user_poem[user_id]),
                                     reply_markup=reply_markup, parse_mode='HTML')

            return

        if update.message.text == LIKE:
            # Какой текст полайкали:
            haiku = last_user_poem[user_id].replace('\n', ' | ')
            logging.info('LIKE: text="%s"', haiku)
            context.bot.send_message(chat_id=update.message.chat_id, text="Спасибо :)")

            last_user_poem[user_id] = None
            last_user_poems[user_id] = []

            return

        if update.message.text == DISLIKE:
            # Какой текст не понравился:
            haiku = last_user_poem[user_id].replace('\n', ' | ')
            logging.info('DISLIKE: text="%s"', haiku)
            context.bot.send_message(chat_id=update.message.chat_id, text="Понятно :(")

            last_user_poem[user_id] = None
            last_user_poems[user_id] = []

            return

        context.bot.send_message(chat_id=update.message.chat_id, text='Минуточку...')

        format = user_format.get(user_id, 'хайку')
        seed = update.message.text
        logging.info('Will generate a %s using seed="%s" for user="%s" id=%s in chat=%s', format, seed, update.message.from_user.name, user_id, str(update.message.chat_id))

        haikux = haiku_generator.generate_output('['+format+'] '+seed, num_return_sequences=5)

        haikux2 = []

        for ipoem, haiku in enumerate(haikux, start=1):
            if '|' in haiku:
                haiku = haiku.replace(' | ', '\n')

            if format == 'хайку':
                if haiku.count('\n') == 2:
                    if is_good_haiku(haiku):
                        p_plagiat = antiplagiat.score(haiku)
                        logging.info('%s #%d for seed="%s" user_id=%s p_plagiat=%5.3f: %s', format, ipoem, seed, user_id, p_plagiat, haiku.replace('\n', ' | '))
                        if p_plagiat < 0.90:
                            haikux2.append(haiku)
            else:
                p_plagiat = antiplagiat.score(haiku)
                logging.info('%s #%d for seed="%s" user_id=%s p_plagiat=%5.3f: %s', format, ipoem, seed, user_id, p_plagiat, haiku.replace('\n', ' | '))
                if p_plagiat < 0.90:
                    haikux2.append(haiku)

        last_user_poems[user_id] = []
        last_user_poem[user_id] = None

        for ipoem, haiku in enumerate(haikux2, start=0):
            msg = haiku
            if ipoem == 1:
                last_user_poem[user_id] = haiku
            else:
                last_user_poems[user_id].append(haiku)

        if last_user_poem[user_id]:
            if len(last_user_poems[user_id]):
                keyboard = [[LIKE, DISLIKE, MORE]]
            else:
                keyboard = [[LIKE, DISLIKE], seed_generator.generate_seeds(user_id)]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id,
                                     text=render_haiku_html(last_user_poem[user_id]),
                                     reply_markup=reply_markup, parse_mode='HTML')
        else:
            keyboard = [seed_generator.generate_seeds(user_id)]
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

    seed_generator = SeedGenerator(models_dir)

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

        updater.dispatcher.add_handler(CallbackQueryHandler(format_menu, pattern='format=(.*)'))

        echo_handler = MessageHandler(Filters.text, echo)
        dispatcher.add_handler(echo_handler)

        logging.getLogger('telegram.bot').setLevel(logging.INFO)
        logging.getLogger('telegram.vendor.ptb_urllib3.urllib3.connectionpool').setLevel(logging.INFO)

        logging.info('Start polling messages for bot %s', tg_bot.name)
        updater.start_polling()
        updater.idle()
    else:
        # Тестирование в консоли

        # Запрашиваем целевой домен
        format = None
        while format is None:
            print('Целевой домен:')
            print('[0] - хайку')
            print('[1] - бусидо')
            print('[2] - приметы')
            print('[3] - афоризмы')
            print('[4] - миниатюры')
            i = input(':> ').strip()
            if i in '01234':
                format = 'хайку бусидо примета афоризм миниатюра'.split()[int(i)]
            else:
                print('Неверный выбор!')

        while True:
            q = input(':> ').strip()
            if q:
                px = haiku_generator.generate_output('[' + format + '] ' + q)
                for ipoem, haiku in enumerate(px, start=1):
                    if '|' in haiku:
                        haiku = haiku.replace(' | ', '\n')

                    is_good = False
                    if format == 'хайку':
                        if haiku.count('\n') == 2:
                            if is_good_haiku(haiku):
                                is_good = True
                    else:
                        is_good = True

                    if is_good:
                        #captions = caption_generator.generate_output(haiku)
                        #caption = captions[0]
                        #caption = '***'
                        print('{} #{} для затравки "{}":'.format(format, ipoem, q))
                        p_plagiat = antiplagiat.score(haiku)
                        print('p_plagiat={}'.format(p_plagiat))
                        #print('--- {} ---\n'.format(caption))
                        print(haiku)
                        print('\n\n')

                print('')
