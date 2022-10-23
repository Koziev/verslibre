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
"""

import os
import io
import logging
import argparse
import traceback

import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler
from telegram import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove, Update

from generative_poetry.init_logging import init_logging
from generative_poetry.poetry_seeds import SeedGenerator
from generative_poetry.poetry_generator_core import PoetryGeneratorCore
from generative_poetry.long_poems_generator import LongPoemGeneratorCore


def get_user_id(update: Update) -> str:
    user_id = str(update.message.from_user.id)
    return user_id


def render_poem_html(poem_txt):
    # 03.02.2022 для отрисовки текста сгенерированного стиха в HTML формате в телеграмме
    s = '<pre>' + poem_txt + '</pre>'
    return s


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

    keyboard = [[InlineKeyboardButton(FORMAT__COMMON, callback_data='format='+FORMAT__COMMON)],
                [InlineKeyboardButton(FORMAT__KID, callback_data='format=' + FORMAT__KID)],
                [InlineKeyboardButton(FORMAT__PHIL, callback_data='format=' + FORMAT__PHIL)],
                [InlineKeyboardButton(FORMAT__HUM, callback_data='format=' + FORMAT__HUM)],
                [InlineKeyboardButton(FORMAT__RUBAI, callback_data='format=' + FORMAT__RUBAI)],
                #[InlineKeyboardButton(FORMAT__MIST, callback_data='format=' + FORMAT__MIST)],
                [InlineKeyboardButton(FORMAT__FOLK, callback_data='format=' + FORMAT__FOLK)],
                [InlineKeyboardButton(FORMAT__POROSHKI, callback_data='format='+FORMAT__POROSHKI)],
                [InlineKeyboardButton(FORMAT__2LINER, callback_data='format='+FORMAT__2LINER)],
                ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    context.bot.send_message(chat_id=update.message.chat_id,
                             text="Привет, {}!\n\n".format(update.message.from_user.full_name) +
                                  "Я - бот для генерации стихов разных жанров (версия от 23.10.2022).\n" +
                                  "Для генерации хайку и бусидо попробуйте @haiku_guru_bot.\n" +
                                  "Если у вас есть вопросы - напишите мне kelijah@yandex.ru\n" +
                                  "Репозиторий проекта: https://github.com/Koziev/verslibre\n\n"
                                  "Выберите формат сочиняемых стихов:\n",
                             reply_markup=reply_markup)
    logging.debug('Leaving START callback with user_id=%s', user_id)


def dbg_actions(update, context):
    pass # TODO ...
    return


def format_menu(context, callback_data):
    user_id = str(context.effective_user.id)
    format = callback_data.match.string.split('=')[1]

    if format == FORMAT__COMMON:
        user_format[user_id] = 'лирика'
    elif format == FORMAT__RUBAI:
        user_format[user_id] = 'рубаи'
    elif format == FORMAT__KID:
        user_format[user_id] = 'детский стишок'
    elif format == FORMAT__PHIL:
        user_format[user_id] = 'философия'
    elif format == FORMAT__HUM:
        user_format[user_id] = 'юмор'
    elif format == FORMAT__MIST:
        user_format[user_id] = 'мистика'
    elif format == FORMAT__FOLK:
        user_format[user_id] = 'частушка'
    elif format == FORMAT__1LINER:
        user_format[user_id] = 'одностишье'
    elif format == FORMAT__2LINER:
        user_format[user_id] = 'двустишье'
    elif format == FORMAT__POROSHKI:
        user_format[user_id] = 'порошок'

    logging.info('Target format set to "%s" for user_id="%s"', user_format[user_id], user_id)

    seeds = seed_generator.generate_seeds(user_id)
    keyboard = [seeds]
    reply_markup = ReplyKeyboardMarkup(keyboard,
                                       one_time_keyboard=True,
                                       resize_keyboard=True,
                                       per_user=True)

    if user_format[user_id] == 'порошок':
        help_text = 'Включен режим <b>пирожков</b>. Если захотите выбрать другой формат стихов, введите команду <code>/start</code>.\n\n' \
                    'Порошки, пирожки и депрессяшки это особые жанры. В этих стихах 1) всегда четыре строки 2) часто нет рифмы, 3) нет знаков препинания, ' \
                    '4) все слова пишутся с маленькой буквы, 5) бывает лексика и темы 18+ 6) есть юмор и стёб.\n\n' \
                    'Теперь вводите какое-нибудь существительное или сочетание прилагательного и существительного, например <i>весна</i>, ' \
                    'и я сочиню пирожок с этими словами. '
    elif user_format[user_id] == 'двустишье':
        help_text = 'Включен режим <b>полупирожков-двустрочников</b>. Если захотите выбрать другой формат стихов, введите команду <code>/start</code>.\n\n' \
                    'В этих стихах часто нет рифмы, встречается лексика 18+.\n\n' \
                    'Теперь вводите какое-нибудь существительное или сочетание прилагательного и существительного, например <i>зимняя любовь</i>, ' \
                    'и я сочиню двустрочник с этими словами. '
    elif user_format[user_id] in ('лирика', 'детский стишок', 'философия', 'юмор', 'рубаи', 'мистика', 'частушка'):
        s = ''
        if user_format[user_id] == 'лирика':
            s = 'лирических стихов'
        elif user_format[user_id] == 'детский стишок':
            s = 'стихов для детей'
        elif user_format[user_id] == 'философия':
            s = 'стихов на философские темы'
        elif user_format[user_id] == 'юмор':
            s = 'юмористических стихов'
        elif user_format[user_id] == 'рубаи':
            s = 'рубаи'
        elif user_format[user_id] == 'мистика':
            s = 'стихов на тему мистики'
        elif user_format[user_id] == 'частушка':
            s = 'частушек'

        help_text = 'Включен режим <b>' + s + '</b>. Если захотите выбрать другой формат стихов, введите команду <code>/start</code>.\n\n' \
                    'Теперь вводите какое-нибудь существительное или сочетание прилагательного и существительного, например <i>счастливая любовь</i>, ' \
                    'и я сочиню стишок с этими словами. '
    else:
        help_text = 'Включен режим <b>моностихов</b>. Если захотите выбрать другой формат стихов, введите команду <code>/start</code>.\n\n' \
                    'Теперь вводите какое-нибудь существительное или сочетание прилагательного и существительного, например <i>синица</i>, ' \
                    'и я сочиню стишок-однострочник с этими словами. '

    help_text += 'Либо выберите готовую тему из предложенных - см. кнопки внизу.\n\n' \
                 'Кнопка [<b>Ещё</b>] выведет новый вариант стиха на заданную тему. Кнопка [<b>Новая тема</b>] выведет новые затравки.'

    context.callback_query.message.reply_text(text=help_text, reply_markup=reply_markup, parse_mode='HTML')
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

            keyboard = [seed_generator.generate_seeds(user_id)]
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
            if format in 'детский стишок|философия|юмор'.split('|'):
                poem = '\n'.join(poetry_generator.continue8(poem.split('\n')))

            last_user_poem[user_id] = poem
            last_user_poems[user_id] = last_user_poems[user_id][:-1]

            if len(last_user_poems[user_id]):
                keyboard = [[LIKE, DISLIKE, MORE, NEW]]
            else:
                keyboard = [[LIKE, DISLIKE], seed_generator.generate_seeds(user_id)]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id,
                                     text=render_poem_html(last_user_poem[user_id]),
                                     reply_markup=reply_markup, parse_mode='HTML')

            return

        #msg = random.choice(['Минуточку, или лучше две...', 'Ок, сажусь писать...', 'Хорошо, буду сочинять...',
        #                     'Понял, приступаю...', 'Отлично, сейчас что-нибудь придумаю...',
        #                     'Ни слова больше! Я поймал вдохновение...', 'Стихи сочинять иду я', 'Ловлю волну вдохновения',
        #                     'Уже стучу по кнопкам!', 'Всегда мечтал об этом написать', 'Тема непростая, но я попробую',
        #                     'Сделаю всё, что в моих силах...'])
        #context.bot.send_message(chat_id=update.message.chat_id, text=msg)

        seed = update.message.text
        logging.info('Will generate a poem using format="%s" seed="%s" for user="%s" id=%s in chat=%s', format, seed, update.message.from_user.name, user_id, str(update.message.chat_id))

        # 22.10.2022 Если генерация ничего не дала (например, все сгенерированные варианты не прошли фильтры),
        # то увеличиваем температуру и повторяем.
        temperature = 1.0
        max_temperature = 1.8
        while temperature <= max_temperature:
            if format in 'лирика'.split('|'):
                # 23.10.2022 отдельная модель для длинных стихов
                poems2 = [('\n'.join(lines), score) for lines, score in long_poetry_generator.generate_poems(format, seed, temperature=temperature)]
            else:
                poems2 = [('\n'.join(lines), score) for lines, score in poetry_generator.generate_poems(format, seed, temperature=temperature)]

            if len(poems2) > 0:
                break
            temperature *= 1.2
            logging.info('Rising temperature to %f and trying again with seed="%s" for user="%s" id=%s in chat=%s', temperature, seed, update.message.from_user.name, user_id, str(update.message.chat_id))

        if len(poems2) == 0:
            logging.info('Could not generate a poem for seed="%s" for user="%s" id=%s in chat=%s', seed, update.message.from_user.name, user_id, str(update.message.chat_id))

        last_user_poems[user_id] = []
        last_user_poem[user_id] = None

        for ipoem, (poem, score) in enumerate(poems2, start=1):
            if ipoem == 1:
                if format in 'лирика|детский стишок|философия|юмор|мистика|частушка'.split('|'):
                    poem = '\n'.join(poetry_generator.continue8(poem.split('\n')))

                last_user_poem[user_id] = poem
            else:
                last_user_poems[user_id].append(poem)

        if last_user_poem[user_id]:
            if len(last_user_poems[user_id]):
                keyboard = [[LIKE, DISLIKE, MORE, NEW]]
            else:
                keyboard = [[LIKE, DISLIKE], seed_generator.generate_seeds(user_id)]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id,
                                     text=render_poem_html(last_user_poem[user_id]),
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
        logging.error(ex)
        logging.error(traceback.format_exc())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verslibre generator v.13')
    parser.add_argument('--token', type=str, default='', help='Telegram token')
    parser.add_argument('--mode', type=str, default='console', choices='console telegram'.split(), help='Frontend selector')
    parser.add_argument('--tmp_dir', default='../../tmp', type=str)
    parser.add_argument('--data_dir', default='../../data', type=str)
    parser.add_argument('--models_dir', default='../../models', type=str)
    parser.add_argument('--log', type=str, default='../../tmp/stressed_gpt_poetry_generation.{DATETIME}.log')

    args = parser.parse_args()
    mode = args.mode
    tmp_dir = os.path.expanduser(args.tmp_dir)
    models_dir = os.path.expanduser(args.models_dir)
    data_dir = os.path.expanduser(args.data_dir)

    init_logging(args.log, True)

    seed_generator = SeedGenerator(models_dir)
    poetry_generator = PoetryGeneratorCore()
    poetry_generator.load(models_dir, data_dir, tmp_dir)

    # 23.10.2022
    long_poetry_generator = LongPoemGeneratorCore()
    long_poetry_generator.load(models_dir, data_dir, tmp_dir)

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

        echo_handler = MessageHandler(Filters.text & ~Filters.command, echo)
        dispatcher.add_handler(echo_handler)

        command_handler = MessageHandler(Filters.all, dbg_actions)
        dispatcher.add_handler(command_handler)
        #dispatcher.add_handler(CallbackQueryHandler(dbg_actions))

        updater.dispatcher.add_handler(CallbackQueryHandler(format_menu, pattern='format=(.*)'))

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
6 - философия
7 - юмор
8 - рубаи
9 - частушка
10 - Филатов
11 - Пушкин
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
                format = 'философия'
            elif s == '7':
                format = 'юмор'
            elif s == '8':
                format = 'рубаи'
            elif s == '9':
                format = 'частушка'
            elif s == '10':
                format = 'Филатов'
            elif s == '11':
                format = 'Пушкин'
            else:
                print('Некорректный вариант!')

        print('\nформат={}\n'.format(format))
        print('Вводите затравку для генерации\n')

        while True:
            topic = input(':> ').strip()

            if format in 'лирика'.split('|'):
                # 23.10.2022 лирика генерится отдельной моделью
                ranked_poems = long_poetry_generator.generate_poems(format, topic, num_return_sequences=5)
            else:
                ranked_poems = poetry_generator.generate_poems(format, topic, num_return_sequences=5)

            for poem, score in ranked_poems:
                print('\nscore={}'.format(score))

                if format in 'детский стишок|философия|юмор|частушка'.split('|'):
                    poem = poetry_generator.continue8(poem)

                for line in poem:
                    print(line)

                print('='*50)
