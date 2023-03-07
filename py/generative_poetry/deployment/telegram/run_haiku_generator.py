"""
Модель для генерации хайку с помощью ruGPT-модели.

01.02.2022 Отбраковываем генерации, в которых три строки повторяются (без учета финальных пунктуаторов).
03.02.2022 Отрисовка текста хайку в HTML с fixed-font, чтобы визуально они лучше выделялись в чате
04.02.2022 Изменился API генерации затравок: теперь это класс SeedGenerator
15.04.2022 Эксперимент с переключением доменов генерации: хайку или бусидо
03.05.2022 Для бусидо сделан отдельный список саджестов
10.07.2022 Добавлены новые разделы: приметы, афоризмы, шутки о жизни, о кошках, о детях, про Чака Норриса, про британских ученых
15.10.2022 Добавлям пайплан обработки картинок вместо текстовых затравок для генерации
22.10.2022 Расширение фильтра повторов для бусидо; перегенерация с повышением температуры в случае, если после фильтров не осталось ни одного варианта
"""

import io
import os
import argparse
import logging.handlers
import random
import re
import collections
import pickle
import getpass

import numpy as np
import torch
#from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
#from transformers import AutoModelForSeq2SeqLM
from PIL import Image

import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler
from telegram import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove, Update

from generative_poetry.udpipe_parser import UdpipeParser
from generative_poetry.rugpt_generator import RugptGenerator
from antiplagiat import Antiplagiat
from generative_poetry.poetry_seeds import SeedGenerator
from generative_poetry.init_logging import init_logging
from generative_poetry.is_good_haiku import is_good_haiku
from generative_poetry.thesaurus import Thesaurus


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
NEW = 'Новая тема'

user_format = dict()
last_user_poems = dict()
last_user_poem = dict()


def start(update, context) -> None:
    user_id = get_user_id(update)
    logging.debug('Entering START callback with user_id=%s', user_id)

    keyboard = [[InlineKeyboardButton('хайку', callback_data='format=' + 'хайку')],
                [InlineKeyboardButton('бусидо', callback_data='format=' + 'бусидо')],
                [InlineKeyboardButton('приметы', callback_data='format=' + 'примета')],
                #[InlineKeyboardButton('про Чака Норриса', callback_data='format=' + 'Чак Норрис')],
                #[InlineKeyboardButton('про британских ученых', callback_data='format=' + 'британские ученые')],
                #[InlineKeyboardButton('о детях', callback_data='format=' + 'дети')],
                #[InlineKeyboardButton('о кошках', callback_data='format=' + 'кошки')],
                #[InlineKeyboardButton('о жизни', callback_data='format=' + 'жизнь')],
                #[InlineKeyboardButton('афоризмы', callback_data='format=' + 'афоризм')],
                ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.send_message(chat_id=update.message.chat_id,
                             text="Привет, {}!\n\nЯ - бот для генерации <b>хайку</b>, <b>бусидо</b> и прочих коротких текстовых миниатюр (версия от 11.11.2022).\n\n".format(update.message.from_user.full_name) +\
                             "Для генерации стихов с рифмой используйте бот @verslibre_bot.\n"
                             "Если у вас есть вопросы - напишите мне kelijah@yandex.ru\n"
                             "Репозиторий проекта: https://github.com/Koziev/verslibre\n\n"
                             "Выберите, что будем сочинять:\n",
                             reply_markup=reply_markup, parse_mode='HTML')

    last_user_poems[user_id] = []
    last_user_poem[user_id] = None
    seed_generator.restart_user_session(user_id)
    logging.debug('Leaving START callback with user_id=%s', user_id)


def format_menu(context, callback_data):
    """Пользователь в телеграм-боте выбрал целевой домен генерируемых текстов."""
    user_id = str(context.effective_user.id)
    format = callback_data.match.string.split('=')[1]

    # Запомним этот выбор для этого пользователя.
    user_format[user_id] = format
    logging.info('Target format set to "%s" for user_id="%s"', user_format[user_id], user_id)

    seeds = seed_generator.generate_seeds(user_id, domain=format)
    keyboard = [seeds]
    reply_markup = ReplyKeyboardMarkup(keyboard,
                                       one_time_keyboard=True,
                                       resize_keyboard=True,
                                       per_user=True)

    if user_format[user_id] == 'хайку':
        help_text = 'Включен режим <b>хайку</b>.\n\n' \
                    'Хайку это короткое нерифмованное трехстишье, выражающие отстраненное восприятие пейзажа.\n' \
                    'Если захотите выбрать другой формат генерируемых текстов, введите команду <code>/start</code>.\n\n' \
                    'Теперь кидайте в бот какое-нибудь существительное или сочетание прилагательного и существительного, например <i>весна</i>, ' \
                    'и я сочиню хайку с этими словами. '
    elif user_format[user_id] == 'бусидо':
        help_text = 'Включен режим <b>бусидо</b>.\n\n' \
                    'Бусидо - это короткий афористичный текст с привкусом восточной мудрости.\n\n' \
                    'Если захотите выбрать другой формат генерируемых текстов, введите команду <code>/start</code>.\n\n' \
                    'Теперь кидайте в бот какое-нибудь существительное или сочетание прилагательного и существительного, например <i>долг</i>, ' \
                    'и я сочиню бусидо с этими словами. '
    elif user_format[user_id] == 'примета':
        help_text = 'Включена генерация <b>примет</b>.\n\n' \
                    'Если захотите выбрать другой формат генерируемых текстов, введите команду <code>/start</code>.\n\n' \
                    'Теперь кидайте в бот какое-нибудь существительное или сочетание прилагательного и существительного, например <i>укол совести</i>, ' \
                    'и я сочиню примету с этими словами. '
    elif user_format[user_id] == 'Чак Норрис':
        help_text = 'Включена генерация <b>шуток про Чака Норриса</b>.\n\n' \
                    'Если захотите выбрать другой формат генерируемых текстов, введите команду <code>/start</code>.\n\n' \
                    'Теперь кидайте в бот какое-нибудь существительное или сочетание прилагательного и существительного, например <i>скукоженный шкворень</i>, ' \
                    'и я придумаю какую-нибудь шутку с этими словами и с упоминанием Чака Норриса :). '
    elif user_format[user_id] == 'британские ученые':
        help_text = 'Включена генерация <b>шуток про британских ученых</b>.\n\n' \
                    'Если захотите выбрать другой формат генерируемых текстов, введите команду <code>/start</code>.\n\n' \
                    'Теперь кидайте в бот какое-нибудь существительное или сочетание прилагательного и существительного, например <i>вирус кайфа</i>, ' \
                    'и я придумаю какую-нибудь шутку с этими словами и с упоминанием британских ученых :). '
    elif user_format[user_id] == 'дети':
        help_text = 'Включена генерация <b>шуток про детей</b>.\n\n' \
                    'Если захотите выбрать другой формат генерируемых текстов, введите команду <code>/start</code>.\n\n' \
                    'Теперь кидайте в бот какое-нибудь существительное или сочетание прилагательного и существительного, например <i>школа</i>, ' \
                    'и я придумаю какую-нибудь шутку про детей с этими словами. '
    elif user_format[user_id] == 'кошки':
        help_text = 'Включена генерация <b>шуток про кошек</b>.\n\n' \
                    'Если захотите выбрать другой формат генерируемых текстов, введите команду <code>/start</code>.\n\n' \
                    'Теперь кидайте в бот какое-нибудь существительное или сочетание прилагательного и существительного, например <i>диван</i>, ' \
                    'и я придумаю какую-нибудь шутку про кошек с этими словами. '
    elif user_format[user_id] == 'жизнь':
        help_text = 'Включена генерация <b>шуток про жизнь</b>.\n\n' \
                    'Если захотите выбрать другой формат генерируемых текстов, введите команду <code>/start</code>.\n\n' \
                    'Теперь кидайте в бот какое-нибудь существительное или сочетание прилагательного и существительного, например <i>зарплата</i>, ' \
                    'и я придумаю какую-нибудь шутку про жизнь с этими словами. '
    elif user_format[user_id] == 'афоризм':
        help_text = 'Включена генерация <b>афоризмов</b>.\n\n' \
                    'Если захотите выбрать другой формат генерируемых текстов, введите команду <code>/start</code>.\n\n' \
                    'Теперь кидайте в бот какое-нибудь существительное или сочетание прилагательного и существительного, например <i>стальная решительность</i>, ' \
                    'и я придумаю какой-нибудь афоризм с этими словами. '
    else:
        raise NotImplementedError()

    help_text += 'Либо выберите готовую затравку из предложенных - см. кнопки внизу.\n\n' \
                 'Кнопка [<b>Ещё</b>] выведет новый вариант текста на ранее заданную тему, а [<b>Новая тема</b>] выведет новые затравки.'

    context.callback_query.message.reply_text(text=help_text, reply_markup=reply_markup, parse_mode='HTML')
    return


def echo(update, context):
    # update.chat.first_name
    # update.chat.last_name
    try:
        user_id = get_user_id(update)
        format = user_format.get(user_id, 'хайку')

        if update.message.text == MORE:
            # Выведем следующее из уже сгенерированных
            m = last_user_poems[user_id][0]
            last_user_poem[user_id] = m
            last_user_poems[user_id] = last_user_poems[user_id][1:]

            if len(last_user_poems[user_id]):
                keyboard = [[LIKE, DISLIKE, MORE, NEW]]
            else:
                keyboard = [[LIKE, DISLIKE], seed_generator.generate_seeds(user_id, domain=format)]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id,
                                     text=render_haiku_html(last_user_poem[user_id]),
                                     reply_markup=reply_markup, parse_mode='HTML')

            return

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
                                     text='Выбирайте одну из затравок на кнопках внизу',
                                     reply_markup=reply_markup)

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

        seed = update.message.text
        logging.info('Will generate a %s using seed="%s" for user="%s" id=%s in chat=%s', format, seed, update.message.from_user.name, user_id, str(update.message.chat_id))

        haikux2 = []

        # 22.10.2022 повторяем попытки генерации с повышением температуры до тех пор, пока через фильтры не пройдет хотя бы 1 вариант.
        temperature = 1.0
        max_temperature = 1.6
        while temperature <= max_temperature:
            haikux = haiku_generator.generate_output('['+format+'] '+seed, num_return_sequences=5, temperature=temperature)

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
                    if format == 'бусидо':
                        if is_good_busido(haiku, udpipe):
                            p_plagiat = antiplagiat.score(haiku)
                            logging.info('%s #%d for seed="%s" user_id=%s p_plagiat=%5.3f: %s', format, ipoem, seed, user_id, p_plagiat, haiku.replace('\n', ' | '))
                            if p_plagiat < 0.90:
                                haikux2.append(haiku)
                        else:
                            logging.error('Bad busido generated: %s', haiku)
                    elif format in ('примета', 'афоризм', 'дети', 'жизнь', 'Чак Норрис', 'британские ученые', 'кошки'):
                        p_plagiat = antiplagiat.score(haiku)
                        logging.info('%s #%d for seed="%s" user_id=%s p_plagiat=%5.3f: %s', format, ipoem, seed, user_id, p_plagiat, haiku)
                        if p_plagiat < 0.90:
                            haikux2.append(haiku)

            if haikux2:
                break

            temperature *= 1.15
            logging.info('Rising temperature to %f and trying again with format="%s" seed="%s" for user="%s"', temperature, format, seed, user_id)

        if len(haikux2) == 0:
            logging.info('Could not generate a poem for format="%s" seed="%s" user="%s"', format, seed, user_id)

        last_user_poems[user_id] = []
        last_user_poem[user_id] = None

        for ipoem, haiku in enumerate(haikux2, start=0):
            if ipoem == 1:
                last_user_poem[user_id] = haiku
            else:
                last_user_poems[user_id].append(haiku)

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
                                     text=render_haiku_html(last_user_poem[user_id]),
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
        logging.error(ex)  # sys.exc_info()[0]


def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  max_length = 16
  num_beams = 4
  gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

  output_ids = vit_model.generate(pixel_values, **gen_kwargs)

  preds = vit_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


def extract_constituent(head, parsing):
    tx = [head]
    for t in parsing:
        if t.head == head.id:
            tx.extend(extract_constituent(t, parsing))
    return tx


def extract_keywords(parsing):
    keywords = set()
    for t in parsing:
        if t.upos in ('NOUN', 'PROPN', 'ADJ') and t.deprel == 'root':
            # именное сказуемое
            #keywords.add(t.lemma)
            np = [t]
            for t2 in parsing:
                if t2.head == t.id:
                    if t2.upos in ('ADJ', 'NOUN'):
                        np.append(t2)
                        for t3 in parsing:
                            if t3.head == t2.id and t3.upos == 'ADP':
                                np.append(t3)
                    elif t2.upos == 'VERB':
                        tx2 = extract_constituent(t2, parsing)
                        if len(tx2) == 1:
                            np.append(t2)

            np = sorted(np, key=lambda z: int(z.id))
            key = ' '.join(k.form for k in np)
            return [key]
        elif t.upos in ('NOUN', 'PROPN', 'ADJ'):
            if t.head != 0:
                if parsing[t.head].upos == 'VERB':
                    if len(t.lemma) > 1:
                        keywords.add(t.lemma)
                elif t.deprel == 'appos' and parsing[t.head].upos in ('NOUN', 'PROPN'):
                    key = parsing[t.head].lemma + ' ' + t.lemma
                    keywords.add(key)
        elif t.upos == 'VERB':
            if len(t.lemma) > 1:
                noun_hit = False
                lx = thesaurus.get_linked(t.lemma, 'ГЛАГОЛ')
                if lx:
                    nouns = list(set(word2 for word2, pos2, rel in lx if pos2 == 'СУЩЕСТВИТЕЛЬНОЕ'))
                    if nouns:
                        keywords.add(random.choice(nouns))
                        noun_hit = True

                if not noun_hit:
                    keywords.add(t.lemma)

    return keywords


def on_process_image(update, context):
    try:
        logging.info('Enter "on_process_image"')
        user_id = get_user_id(update)
        format = user_format.get(user_id, 'хайку')

        context.bot.send_message(chat_id=update.message.chat_id, text='Рассматриваю картинку...')
        best_photo = None
        min_dist = 10000000
        for photo in update.message.photo:
            d = abs(512-photo.width) + abs(512-photo.height)
            if d < min_dist:
                best_photo = photo
                min_dist = d

        if best_photo is not None:
            file_id = best_photo.file_id
            file = context.bot.getFile(file_id)
            #print("file_id: " + str(file_id))
            download_path = os.path.join(tmp_dir, 'haiku_tg_image.jpg')
            file.download(download_path)
            en_caption = predict_step([download_path])[0]
            logging.debug('Image caption for user_id="%s" decoded: "%s"', user_id, en_caption)

            inputs = nllb_tokenizer(en_caption, return_tensors="pt").to(device)
            translated_tokens = nllb_model.generate(**inputs, forced_bos_token_id=nllb_tokenizer.lang_code_to_id['rus_Cyrl'],  max_length=60)
            ru_caption = nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            logging.debug('Russian translation of image caption for user_id="%s": "%s"', user_id, ru_caption)

            # Извлечем ключевое слово, например - именную группу
            keys = set()
            for parsing in udpipe.parse_text(ru_caption):
                keys.update(extract_keywords(parsing))

            keys = sorted(keys, key=lambda z: (-z.count(' ') - len(z)//4))
            wx = [1.0/i for i in range(1, len(keys)+1)]
            seed = random.choices(population=keys, weights=wx, k=1)[0]

            context.bot.send_message(chat_id=update.message.chat_id, text='Сочиняю текст...')

            logging.info('Will generate a %s using seed="%s" for user="%s" id=%s in chat=%s', format, seed,
                         update.message.from_user.name, user_id, str(update.message.chat_id))

            haikux = haiku_generator.generate_output('[' + format + '] ' + seed, num_return_sequences=5)

            haikux2 = []

            for ipoem, haiku in enumerate(haikux, start=1):
                if '|' in haiku:
                    haiku = haiku.replace(' | ', '\n')

                if format == 'хайку':
                    if haiku.count('\n') == 2:
                        if is_good_haiku(haiku):
                            p_plagiat = antiplagiat.score(haiku)
                            logging.info('%s #%d for seed="%s" user_id=%s p_plagiat=%5.3f: %s', format, ipoem, seed,
                                         user_id, p_plagiat, haiku.replace('\n', ' | '))
                            if p_plagiat < 0.90:
                                haikux2.append(haiku)
                else:
                    if format == 'бусидо':
                        if is_good_busido(haiku, udpipe):
                            p_plagiat = antiplagiat.score(haiku)
                            logging.info('%s #%d for seed="%s" user_id=%s p_plagiat=%5.3f: %s', format, ipoem, seed,
                                         user_id, p_plagiat, haiku.replace('\n', ' | '))
                            if p_plagiat < 0.90:
                                haikux2.append(haiku)
                        else:
                            logging.error('Bad busido generated: %s', haiku)
                    elif format in ('примета', 'афоризм', 'дети', 'жизнь', 'Чак Норрис', 'британские ученые', 'кошки'):
                        p_plagiat = antiplagiat.score(haiku)
                        logging.info('%s #%d for seed="%s" user_id=%s p_plagiat=%5.3f: %s', format, ipoem, seed,
                                     user_id, p_plagiat, haiku)
                        if p_plagiat < 0.90:
                            haikux2.append(haiku)

            last_user_poems[user_id] = []
            last_user_poem[user_id] = None

            for ipoem, haiku in enumerate(haikux2, start=0):
                if ipoem == 1:
                    last_user_poem[user_id] = haiku
                else:
                    last_user_poems[user_id].append(haiku)

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
                                         text=render_haiku_html(last_user_poem[user_id]),
                                         reply_markup=reply_markup, parse_mode='HTML')
            else:
                keyboard = [seed_generator.generate_seeds(user_id, domain=format)]
                reply_markup = ReplyKeyboardMarkup(keyboard,
                                                   one_time_keyboard=True,
                                                   resize_keyboard=True,
                                                   per_user=True)

                context.bot.send_message(chat_id=update.message.chat_id,
                                         text='Что-то не получается сочинить :(\nЗадайте другую тему или пришлите другую картинку, пожалуйста',
                                         reply_markup=reply_markup)
    except Exception as ex:
        logging.error('Error in "on_process_image"')
        logging.error(ex)


def is_good_busido(busido_text, parser):
    parsings = parser.parse_text(busido_text)
    for parsing in parsings:
        noun_freqs = collections.Counter()
        for t in parsing:
            if t.upos in ('PROPN', 'NOUN', 'VERB', 'ADJ', 'INTJ', 'SYM'):
                noun_freqs[t.lemma.lower().replace('ё', 'е')] += 1
        if noun_freqs.most_common(1)[0][1] > 2:
            return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Telegram bot for haiku and busido generation')
    parser.add_argument('--token', type=str, default='', help='Telegram token')
    parser.add_argument('--mode', type=str, default='console', choices='console telegram'.split())
    parser.add_argument('--models_dir', type=str, default='../../models')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--log', type=str, default='~/polygon/text_generator/tmp/haiku_generator.log')
    parser.add_argument('--tmp', type=str, default='~/polygon/text_generator/tmp')

    args = parser.parse_args()

    mode = args.mode

    models_dir = os.path.expanduser(args.models_dir)
    dataset_path = os.path.expanduser(args.dataset_path) if args.dataset_path else None

    tmp_dir = os.path.expanduser(args.tmp)

    init_logging(os.path.expanduser(args.log), True)

    # Для работы с сервером телеграмма нужен зарегистрированный бот.
    # Результатом регистрации является токен - уникальная строка символов.
    # Вот эту строку надо сейчас ввести с консоли.
    # Возможно, следует предусмотреть передачу токена через опцию ком. строки.
    if args.mode == 'telegram':
        telegram_token = args.token
        if len(telegram_token) == 0:
            telegram_token = getpass.getpass('Enter Telegram token:> ').strip()

    seed_generator = SeedGenerator(models_dir)

    #tokenizer = rutokenizer.Tokenizer()
    #tokenizer.load()

    #ranker = TopicRanker(tokenizer)
    #ranker.load("/home/inkoziev/polygon/w2v/fasttext.CBOW=1_WIN=5_DIM=64")

    antiplagiat = Antiplagiat()
    if dataset_path:
        logging.info('Initializing antiplagiat module with dataset "%s"', dataset_path)
        with io.open(dataset_path, 'r', encoding='utf-8') as rdr:
            for sample in rdr.read().split('</s>'):
                text = sample.replace('<s>', '')
                if '#' in text:
                    text = text[text.index('#')+1:].strip()
                    antiplagiat.add_document(text)
    else:
        logging.error('Antiplagiat dataset is not available')

    logging.info('Loading image processing models...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Модели для обработки картинок
    if False:
        logging.info('Start loading image captioning model')
        vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        vit_model.to(device)

        logging.info('Start loading NLLB model')
        nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        nllb_model.to(device)

    logging.info('Loading the haiku generation models from "%s"...', models_dir)
    haiku_generator = RugptGenerator()
    haiku_generator.load(os.path.join(models_dir, 'rugpt_haiku_generator'))

    # Парсер будет нужен для отсева текстов бусидо с повторами.
    udpipe = UdpipeParser()
    udpipe.load(models_dir)

    thesaurus_path = os.path.join(tmp_dir, 'thesaurus.pkl')
    with open(thesaurus_path, 'rb') as f:
        thesaurus = pickle.load(f)

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

        #img_handler = MessageHandler(Filters.photo, on_process_image)
        #dispatcher.add_handler(img_handler)

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
            print('[4] - шутки о жизни')
            print('[5] - шутки о кошках')
            print('[6] - шутки о детях')
            print('[7] - шутки про Чака Норриса')
            print('[8] - шутки про британских ученых')
            print('[9] - поговорки')
            print('[10] - календарь')
            print('[11] - миниатюры')
            i = input(':> ').strip()
            if 0 <= int(i) <= 10:
                format = 'хайку|бусидо|примета|афоризм|о жизни|кошки|дети|Чак Норрис|британские ученые|поговорка|календарь|миниатюра'.split('|')[int(i)]
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
