"""
End-2-end генерация рифмованного четверостишья с помощью отфайнтюненной GPT с маркировкой ударений.
Используется вводимая затравка в виде словосочетания (именная группа).

09-12-2021 Подключен StressedGptTokenizer и используется tokenizer_config.json
09-12-2021 Доработка для телеграм-бота
11-12-2021 Переписываем код top_t+top_p сэмплинга, чтобы банить цепочки с повтором рифмуемого слова.
14-12-2021 Добавлен код для автоматической пакетной оценки качества генерации.
18-12-2021 Добавлена коррекция пробелов после декодера, модуль whitespace_normalization
27-01-2022 Эксперимент с управлением форматом генерации с помощью тегов [стих | одностишье | двустишье | порошок]
08.02.2022 В ранжирование результатов генерации добавлена проверка бедной рифмовки (включая повторение рифмуемого слова)
21.02.2022 Добавлена модель генерации следующих 4х строчек по первому четверостишью.
26.02.2022 Рефакторинг - генерация стиха, ранжировка и прочее вынесены в отдельный модуль, а тут остается только фронт
03.05.2022 Добавлена кнопка "Новая тема" для формирования новых саджестов
08.05.2022 Эксперимент с генерацией рубаи
08.05.2022 Из телеграм-версии исключен режим моностихов
24.05.2022 Добавлен второй уровень рубрикации четверостиший - выделены рубаи, частушки, мистика и т.д.
27.05.2022 В режиме генерации рубаи не делается продолжение (след. 4 строки по первым 4м)
22.10.2022 Реализован повтор попыток генерации с повышенной температурой, если все сгенерированные варианты отсеяны фильтрами качества
23.10.2022 Эксперимент с отдельной моделью для генерации длинных стихов: LongPoetryGenerator
27.10.2022 Если генерация в модели long poems ничего не дала, запускаем генерацию в старой модели 4-строчников.
11.11.2022 Большая чистка: убираем все жанры генерации, кроме лирики, меняем стартовый экран. Новое ядро генератора.
"""

import os
import io
import logging
import argparse
import traceback
import getpass

import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler
from telegram import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove, Update

import tensorflow as tf

from generative_poetry.init_logging import init_logging
from generative_poetry.poetry_seeds import SeedGenerator

from generative_poetry.long_poem_generator2 import LongPoemGeneratorCore2


def get_user_id(update: Update) -> str:
    user_id = str(update.message.from_user.id)
    return user_id


def render_poem_html(poem_txt):
    # 03.02.2022 для отрисовки текста сгенерированного стиха в HTML формате в телеграмме
    s = '<pre>' + poem_txt + '</pre>'
    return s


top_p = 0.85
top_k = 50
typical_p = 0.7


LIKE = 'Нравится!'
DISLIKE = 'Плохо :('
NEW = 'Новая тема'
MORE = 'Еще...'

last_user_poems = dict()
last_user_poem = dict()
user_format = dict()


FORMAT__COMMON = 'Лирика'
FORMAT__1LINER = 'Однострочники'
FORMAT__2LINER = 'Двухстрочники'
FORMAT__POROSHKI = 'Пирожки и порошки'
FORMAT__RUBAI = 'Рубаи'
FORMAT__KID = "Для детей"
FORMAT__PHIL = "Философские"
FORMAT__HUM = "Юмор и сатира"
#FORMAT__MIST = "Мистика"   23.10.2022 мистика объединилась с лирикой в новой модели генерации длинных стихов
FORMAT__FOLK = "Частушки"


def start(update, context) -> None:
    user_id = get_user_id(update)
    logging.debug('Entering START callback with user_id=%s', user_id)

    # 08.05.2022 сбросим историю использованных затравок
    seed_generator.restart_user_session(user_id)

    intro_text = "Привет, {}!\n\n".format(update.message.from_user.full_name) + \
    "Я - бот для генерации стихов (версия от 11.11.2022).\n" + \
    "Для обратной связи используйте kelijah@yandex.ru или https://github.com/Koziev/verslibre.\n\n" + \
    "Теперь вводите тему - какое-нибудь существительное или сочетание прилагательного и существительного, например <i>счастливая любовь</i>, " + \
    "и я сочиню стишок с этими словами. " + \
    "Либо выберите готовую тему из предложенных - см. кнопки внизу.\n\n" + \
    "Кнопка [<b>Ещё</b>] выведет новый вариант стиха на заданную тему. Кнопка [<b>Новая тема</b>] выведет новые затравки."

    # Получаем порцию саджестов (обычно 3 штуки) под выбранный жанр, чтобы пользователю не пришлось напрягаться
    # с придумыванием затравок.
    seeds = seed_generator.generate_seeds(user_id, domain=user_format.get(user_id))
    keyboard = [seeds]
    reply_markup = ReplyKeyboardMarkup(keyboard,
                                       one_time_keyboard=True,
                                       resize_keyboard=True,
                                       per_user=True)

    context.bot.send_message(chat_id=update.message.chat_id, text=intro_text, reply_markup=reply_markup, parse_mode='HTML')
    logging.debug('Leaving START callback with user_id=%s', user_id)


def dbg_actions(update, context):
    pass # TODO ...
    return


def echo(update, context):
    # update.chat.first_name
    # update.chat.last_name
    try:
        user_id = get_user_id(update)
        format = user_format.get(user_id, 'лирика')

        if update.message.text == NEW:
            # Пользователь хочет, чтобы ему предложили новые саджесты для генерации.
            last_user_poem[user_id] = None
            last_user_poems[user_id] = []

            keyboard = [seed_generator.generate_seeds(user_id, domain=format)]
            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)
            context.bot.send_message(chat_id=update.message.chat_id,
                                     text="Выберите тему из предложенных или введите свою",
                                     reply_markup=reply_markup)
            return

        if update.message.text == LIKE:
            # Какой текст полайкали:
            poem = last_user_poem[user_id].replace('\n', ' | ')
            logging.info('LIKE: format="%s" poem="%s" user="%s"', format, poem, user_id)

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
            poem = last_user_poem[user_id].replace('\n', ' | ')
            logging.info('DISLIKE: format="%s" poem="%s" user="%s"', format, poem, user_id)

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
            poem = last_user_poems[user_id][-1]

            # 23.10.2022 исключил лирику и мистику из списка, так как теперь она будет генерироваться новой модель.
            # TODO: исключить остальные стихи тоже
            #if format in 'детский стишок|философия|юмор'.split('|'):
            #    poem = '\n'.join(poetry_generator.continue8(poem.split('\n')))

            last_user_poem[user_id] = poem
            last_user_poems[user_id] = last_user_poems[user_id][:-1]

            if len(last_user_poems[user_id]):
                keyboard = [[LIKE, DISLIKE, MORE, NEW]]
            else:
                keyboard = [[LIKE, DISLIKE], seed_generator.generate_seeds(user_id, domain=format)]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id,
                                     text=render_poem_html(last_user_poem[user_id]),
                                     reply_markup=reply_markup, parse_mode='HTML')

            return

        seed = update.message.text
        logging.info('Will generate a poem using format="%s" seed="%s" for user="%s" id=%s in chat=%s', format, seed, update.message.from_user.name, user_id, str(update.message.chat_id))

        genre = format
        if format == 'детский стишок':
            genre = 'стихи для детей'

        # 22.10.2022 Если генерация ничего не дала (например, все сгенерированные варианты не прошли фильтры),
        # то увеличиваем температуру и повторяем.
        temperature = 1.0
        max_temperature = 1.6
        while temperature <= max_temperature:
            ranked_poems = long_poetry_generator.generate_poems(genre=genre, topic=seed,
                                                                temperature=temperature, top_p=top_p, top_k=top_k, typical_p=typical_p,
                                                                num_return_sequences=5)
            poems2 = [('\n'.join(lines), score) for lines, score in ranked_poems]

            if len(poems2) > 0:
                break

            temperature *= 1.1
            logging.info('Rising temperature to %f and trying again with seed="%s" for user="%s" id=%s in chat=%s', temperature, seed, update.message.from_user.name, user_id, str(update.message.chat_id))

        if len(poems2) == 0:
            logging.info('Could not generate a poem for seed="%s" for user="%s" id=%s in chat=%s', seed, update.message.from_user.name, user_id, str(update.message.chat_id))

        last_user_poems[user_id] = []
        last_user_poem[user_id] = None

        for ipoem, (poem, score) in enumerate(poems2, start=1):
            if ipoem == 1:
                last_user_poem[user_id] = poem
            else:
                last_user_poems[user_id].append(poem)

        if last_user_poem[user_id]:
            if len(last_user_poems[user_id]):
                keyboard = [[LIKE, DISLIKE, MORE, NEW]]
            else:
                keyboard = [[LIKE, DISLIKE], seed_generator.generate_seeds(user_id, domain=format)]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id,
                                     text=render_poem_html(last_user_poem[user_id]),
                                     reply_markup=reply_markup, parse_mode='HTML')
        else:
            keyboard = [seed_generator.generate_seeds(user_id, domain=format)]
            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id,
                                     text='Что-то не получается сочинить :(\nЗадайте другую тему, пожалуйста',
                                     reply_markup=reply_markup)

    except Exception as ex:
        logging.error('Error in "echo"')
        logging.error(ex)
        logging.error(traceback.format_exc())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verslibre & haiku generator v.15')
    parser.add_argument('--token', type=str, default='', help='Telegram token')
    parser.add_argument('--mode', type=str, default='console', choices='console telegram'.split(), help='Frontend selector')
    parser.add_argument('--tmp_dir', default='../../tmp', type=str)
    parser.add_argument('--data_dir', default='../../data', type=str)
    parser.add_argument('--models_dir', default='../../models', type=str)
    parser.add_argument('--log', type=str, default='../../tmp/stressed_gpt_poetry_generation.{HOSTNAME}.{DATETIME}.log')

    args = parser.parse_args()
    mode = args.mode
    tmp_dir = os.path.expanduser(args.tmp_dir)
    models_dir = os.path.expanduser(args.models_dir)
    data_dir = os.path.expanduser(args.data_dir)

    init_logging(args.log, True)

    # 19-03-2022 запрещаем тензорфлоу резервировать всю память в гпу по дефолту, так как
    # это мешает потом нормально работать моделям на торче.
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Генератор саджестов
    seed_generator = SeedGenerator(models_dir)

    # Генератор рифмованных стихов
    logging.info('Loading the long poetry generation models from "%s"...', models_dir)
    long_poetry_generator = LongPoemGeneratorCore2()
    long_poetry_generator.load(models_dir, data_dir, tmp_dir)

    # Генератор файку - добавлен 12.11.2022
    #logging.info('Loading the haiku generation models from "%s"...', models_dir)
    #haiku_generator = RugptGenerator()
    #haiku_generator.load(os.path.join(models_dir, 'rugpt_haiku_generator'))

    if args.mode == 'telegram':
        # Телеграм-версия генератора
        telegram_token = args.token
        if len(telegram_token) == 0:
            telegram_token = getpass.getpass('Enter Telegram token:> ').strip()

        logging.info('Starting telegram bot')
        tg_bot = telegram.Bot(token=telegram_token).getMe()
        bot_id = tg_bot.name
        logging.info('Telegram bot "%s" id=%s', tg_bot.name, tg_bot.id)

        updater = Updater(token=telegram_token)
        dispatcher = updater.dispatcher

        start_handler = CommandHandler('start', start)
        dispatcher.add_handler(start_handler)

        echo_handler = MessageHandler(Filters.text & ~Filters.command, echo)
        dispatcher.add_handler(echo_handler)

        command_handler = MessageHandler(Filters.all, dbg_actions)
        dispatcher.add_handler(command_handler)
        #dispatcher.add_handler(CallbackQueryHandler(dbg_actions))

        #updater.dispatcher.add_handler(CallbackQueryHandler(format_menu, pattern='format=(.*)'))

        logging.getLogger('telegram.bot').setLevel(logging.INFO)
        logging.getLogger('telegram.vendor.ptb_urllib3.urllib3.connectionpool').setLevel(logging.INFO)

        logging.info('Start polling messages for bot %s', tg_bot.name)
        updater.start_polling()
        updater.idle()
    else:
        # Тестирование в консоли
        menu = """Выберите жанр:
1 - моностихи
2 - двустрочники
3 - порошки и пирожки
4 - лирика
5 - детский стишок
6 - юмор
7 - рубаи
8 - частушка
9 - Филатов
10 - Пушкин
"""
        print(menu)
        format = None
        while not format:
            s = input('1 ... 11 :> ').strip()
            if s == '1':
                format = 'одностишье'
            elif s == '2':
                format = 'двустишье'
            elif s == '3':
                format = 'порошок'
            elif s == '4':
                format = 'лирика'
            elif s == '5':
                format = 'детский стишок'
            elif s == '6':
                format = 'юмор'
            elif s == '7':
                format = 'рубаи'
            elif s == '8':
                format = 'частушка'
            elif s == '9':
                format = 'Филатов'
            elif s == '10':
                format = 'Пушкин'
            else:
                print('Некорректный вариант!')

        print('\nформат={}\n'.format(format))
        print('Вводите затравку для генерации\n')

        while True:
            topic = input(':> ').strip()

            genre = format

            if format == 'детский стишок':
                genre = 'стихи для детей'

            ranked_poems = long_poetry_generator.generate_poems(genre=genre, topic=topic,
                                                                temperature=1.0, top_p=top_p, top_k=top_k, typical_p=typical_p,
                                                                num_return_sequences=5)

            for poem, score in ranked_poems:
                print('\nscore={}'.format(score))
                for line in poem:
                    print(line)
                print('='*50)
