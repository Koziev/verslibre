"""
End-2-end генерация рифмованного четверостишья с помощью отфайнтюненной GPT с маркировкой ударений.
Используется вводимая затравка в виде словосочетания (именная группа).

09-12-2021 Подключен StressedGptTokenizer и используется tokenizer_config.json
09-12-2021 Доработка для телеграм-бота
11-12-2021 Переписываем код top_t+top_p сэмплинга, чтобы банить цепочки с повтором рифмуемого слова.
"""

import os
import json
import logging
import argparse
import random
import traceback
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn
import transformers
import transformers.generation_utils

from transformers.generation_logits_process import (
    #EncoderNoRepeatNGramLogitsProcessor,
    #ForcedBOSTokenLogitsProcessor,
    #ForcedEOSTokenLogitsProcessor,
    #HammingDiversityLogitsProcessor,
    #InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxNewTokensCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)



import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove, Update

from init_logging import init_logging
from poetry.phonetic import Accents
from generative_poetry.udpipe_parser import UdpipeParser
from generative_poetry.poetry_alignment import PoetryStressAligner
from generative_poetry.experiments.rugpt_with_stress.break_to_syllables import break_to_syllables
from generative_poetry.experiments.rugpt_with_stress.arabize import arabize
from generative_poetry.experiments.rugpt_with_stress.stressed_gpt_tokenizer import StressedGptTokenizer
from poetry_seeds import generate_seeds


def sample_v2(
        self,
        input_ids: torch.LongTensor,
        logits_processor = None, #: Optional[LogitsProcessorList]
        stopping_criteria = None, #: Optional[StoppingCriteriaList]
        logits_warper = None, #: Optional[LogitsProcessorList]
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
): # -> Union[SampleOutput, torch.LongTensor]:
    r"""
    Generates sequences for models with a language modeling head using multinomial sampling.

    Parameters:

        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (:obj:`LogitsProcessorList`, `optional`):
            An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
            :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
            head applied at each generation step.
        stopping_criteria (:obj:`StoppingCriteriaList`, `optional`):
            An instance of :class:`~transformers.StoppingCriteriaList`. List of instances of class derived from
            :class:`~transformers.StoppingCriteria` used to tell if the generation loop should stop.
        logits_warper (:obj:`LogitsProcessorList`, `optional`):
            An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
            :class:`~transformers.LogitsWarper` used to warp the prediction score distribution of the language
            modeling head applied before multinomial sampling at each generation step.
        max_length (:obj:`int`, `optional`, defaults to 20):
            **DEPRECATED**. Use :obj:`logits_processor` or :obj:`stopping_criteria` directly to cap the number of
            generated tokens. The maximum length of the sequence to be generated.
        pad_token_id (:obj:`int`, `optional`):
            The id of the `padding` token.
        eos_token_id (:obj:`int`, `optional`):
            The id of the `end-of-sequence` token.
        output_attentions (:obj:`bool`, `optional`, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
            returned tensors for more details.
        output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
            for more details.
        output_scores (:obj:`bool`, `optional`, defaults to `False`):
            Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
        return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        synced_gpus (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If
            model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

    Return:
        :class:`~transformers.generation_utils.SampleDecoderOnlyOutput`,
        :class:`~transformers.generation_utils.SampleEncoderDecoderOutput` or obj:`torch.LongTensor`: A
        :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
        :class:`~transformers.generation_utils.SampleDecoderOnlyOutput` if
        ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
        :class:`~transformers.generation_utils.SampleEncoderDecoderOutput` if
        ``model.config.is_encoder_decoder=True``.

    Examples::

        >>> from transformers import (
        ...    AutoTokenizer,
        ...    AutoModelForCausalLM,
        ...    LogitsProcessorList,
        ...    MinLengthLogitsProcessor,
        ...    TopKLogitsWarper,
        ...    TemperatureLogitsWarper,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList([
        ...     MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
        ... ])
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList([
        ...     TopKLogitsWarper(50),
        ...     TemperatureLogitsWarper(0.7),
        ... ])

        >>> outputs = model.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper)

        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
    """

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    output_scores = output_scores if output_scores is not None else self.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    cur_len = input_ids.shape[-1]

    this_peer_finished = False  # used by synced_gpus only
    # auto-regressive generation
    while True:
        #if synced_gpus:
        #    # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
        #    # The following logic allows an early break if all peers finished generating their sequence
        #    this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
        #    # send 0.0 if we finished, 1.0 otherwise
        #    dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
        #    # did all peers finish? the reduced sum will be 0.0 then
        #    if this_peer_finished_flag.item() == 0.0:
        #        break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # sample
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        cur_len = cur_len + 1

        # =============================================
        # inkoziev start
        nl_token_id = self.tokenizer.vocab['<nl>']
        break_token_id = self.tokenizer.vocab['|']

        input_ids_2 = input_ids.cpu().numpy()
        nb_bad_rows = 0
        row_validity = [True] * input_ids_2.shape[0]
        for irow, row_ids in enumerate(input_ids_2):
            rhymes = []
            state = 'nl_hit'
            last_nl_pos = -1
            bad_row = False
            for j, x in enumerate(row_ids[1:]):  # первый токен в каждой цепочке это <s>, его пропускаем
                if x == nl_token_id:
                    state = 'nl_hit'
                    last_nl_pos = j
                elif x == break_token_id:
                    if state == 'nl_hit':  # нас интересует первое слово после <nl> (так как у нас цепочки right2left, то это фактически последнее слово в строке)
                        rhyme = ' '.join(map(str, row_ids[last_nl_pos+1: j]))
                        if rhyme in rhymes:
                            bad_row = True
                            nb_bad_rows += 1
                            break
                        else:
                            rhymes.append(rhyme)
                            state = 'break_hit'

            if bad_row:
                row_validity[irow] = False
            #    # Плохая цепочка генерации, т.к. есть повтор в рифме. Исключим эту цепочку.
            #    input_ids_2 = np.delete(input_ids_2, axis=0, obj=irow)
            #    unfinished_sequences_2 = np.delete(unfinished_sequences_2, axis=0, obj=irow)
            #else:
            #    irow += 1


        if nb_bad_rows > 0:
            # из текущего тензора с цепочками генерации исключена 1 или больше цепочек.
            input_ids = input_ids[row_validity]
            unfinished_sequences = unfinished_sequences[row_validity]
            model_kwargs['attention_mask'] = model_kwargs['attention_mask'][row_validity]
            next_tokens = next_tokens[row_validity]
            if model_kwargs['past'] is not None:
                new_pkv = []
                for tensor1, tensor2 in model_kwargs['past']:
                    new_pkv.append((tensor1[row_validity], tensor2[row_validity]))
                model_kwargs['past'] = tuple(new_pkv)

        # inkoziev end
        # =============================================

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    return input_ids


class RugptGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load(self, model_dir):
        with open(os.path.join(model_dir, 'tokenizer_config.json'), 'r') as f:
            config = json.load(f)
            tokenizer_class = config['tokenizer_class']
            if tokenizer_class == 'BertTokenizer':
                self.tokenizer = transformers.BertTokenizer.from_pretrained(model_dir)
            elif tokenizer_class == 'StressedGptTokenizer':
                self.tokenizer = StressedGptTokenizer.from_pretrained(model_dir)
            else:
                raise NotImplementedError()

        self.model = transformers.GPT2LMHeadModel.from_pretrained(model_dir)

        self.model.sample = sample_v2.__get__(self.model)  # меням на свой сэмплер

        self.model.tokenizer = self.tokenizer  # он нам понадобится внутри нашей версии sample()

        self.model.to(self.device)

    def generate_output(self, context, num_return_sequences=10, temperature=1.0):
        top_k = 30
        top_p = 0.85
        repetition_penalty = 1.0
        prompt_text = "<s> " + context + ' $'
        stop_token = "</s>"
        length = 150

        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=length + len(encoded_prompt[0]),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            #num_beams=5,
            #num_beam_groups=5,
            num_return_sequences=num_return_sequences,
            pad_token_id=0
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = set()
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()[encoded_prompt.shape[1]:]

            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            if stop_token in text:
                text = text[: text.find(stop_token)]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            #total_sequence = text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
            #total_sequence = total_sequence.strip().replace('<pad>', '')
            #generated_sequences.add(total_sequence)
            generated_sequences.add(text.strip().replace('<pad>', ''))

        return list(generated_sequences)


def decode_line2(line0, remove_stress_marks=True):
    out_words = []

    tokens = [z.strip() for z in line0.split('|')]
    for token in tokens:
        if remove_stress_marks:
            syllabs = token.replace('\u0301', '').split(' ')
        else:
            syllabs = token.split(' ')

        out_word = ''.join(syllabs[::-1])
        out_words.append(out_word)

    s = ' '.join(out_words[::-1])
    s = s.replace(' ,', ',').replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' …', '…').replace(' :', ':')
    return s


def generate_poems(topic):
    seed = arabize(break_to_syllables(udpipe, accents, topic.lower()))

    poems = poem_generator.generate_output(seed, num_return_sequences=10)

    # Отранжируем результат генерации по консистентности ритма...
    ranked_poems = []
    for poem in poems:
        lines = [decode_line2(line) for line in poem.split('<nl>') if len(line) > 0]
        if len(lines) == 4:
            score = 0.0
            try:
                a = aligner.align(lines)
                if a is not None:
                    score = a.score
            except:
                pass
            if score > 0.05:
                ranked_poems.append((lines, score))
            elif score == 0.0:
                # НАЧАЛО ОТЛАДКИ
                logging.info('@451 === BAD GENERATION ===')
                logging.info('Raw lines:')
                for line in poem.split('<nl>'):
                    if len(line) > 0:
                        logging.info('%s', line)

                logging.info('Decoded lines:')
                for line in poem.split('<nl>'):
                    if len(line) > 0:
                        logging.info('%s', decode_line2(line, remove_stress_marks=False))
                # КОНЕЦ ОТЛАДКИ

    ranked_poems = sorted(ranked_poems, key=lambda z: -z[1])
    return ranked_poems


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
                                     text="Выберите тему из предложенных или введите свою",
                                     reply_markup=reply_markup)
            return

        if update.message.text == LIKE:
            # Какой текст полайкали:
            poem = last_user_poem[user_id].replace('\n', ' | ')
            logging.info('LIKE: poem="%s" user="%s"', poem, user_id)

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
            logging.info('DISLIKE: poem="%s" user="%s"', poem, user_id)

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

            last_user_poem[user_id] = poem
            last_user_poems[user_id] = last_user_poems[user_id][:-1]

            if len(last_user_poems[user_id]):
                keyboard = [[LIKE, DISLIKE, MORE]]
            else:
                keyboard = [[LIKE, DISLIKE], generate_seeds(user_id)]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id,
                                     text=last_user_poem[user_id],
                                     reply_markup=reply_markup)

            return

        msg = random.choice(['Минуточку, или лучше две...', 'Ок, сажусь писать...', 'Хорошо, буду сочинять...',
                             'Понял, приступаю...', 'Отлично, сейчас что-нибудь придумаю...',
                             'Ни слова больше! Пошло вдохновение...', 'Стихи сочинять иду я', 'Ловлю волну вдохновения',
                             'Уже стучу по кнопкам!', 'Всегда мечтал об этом написать', 'Тема непростая, но я попробую',
                             'Сделаю всё, что в моих силах...'])
        context.bot.send_message(chat_id=update.message.chat_id, text=msg)

        seed = update.message.text
        logging.info('Will generate a poem using seed="%s" for user="%s" id=%s in chat=%s', seed, update.message.from_user.name, user_id, str(update.message.chat_id))

        poems2 = [('\n'.join(lines), score) for lines, score in generate_poems(seed)]

        last_user_poems[user_id] = []
        last_user_poem[user_id] = None

        for ipoem, (poem, score) in enumerate(poems2, start=1):
            if ipoem == 1:
                last_user_poem[user_id] = poem
            else:
                last_user_poems[user_id].append(poem)

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
                                     text=last_user_poem[user_id],
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
        logging.error(ex)
        logging.error(traceback.format_exc())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verslibre generator v.6')
    parser.add_argument('--token', type=str, default='', help='Telegram token')
    parser.add_argument('--mode', type=str, default='console', choices='console telegram'.split())
    parser.add_argument('--tmp_dir', default='../../tmp', type=str)
    parser.add_argument('--models_dir', default='../../models', type=str)
    parser.add_argument('--log', type=str, default='../../tmp/stressed_gpt_poetry_generation.log')

    args = parser.parse_args()
    mode = args.mode
    tmp_dir = os.path.expanduser(args.tmp_dir)
    models_dir = os.path.expanduser(args.models_dir)

    init_logging(args.log, True)

    poem_generator = RugptGenerator()
    poem_generator.load(os.path.join(models_dir, 'stressed_poetry_generator'))

    # НАЧАЛО ОТЛАДКИ
    #generated_sequence = [1003, 20681, 20772, 19860, 20772, 19875, 1017, 20772, 10142, 20772, 671, 20772, 338, 20772, 10533, 20772, 13658, 5, 24335, 19999, 18486, 17874, 20772, 4095, 11931, 20772, 25279, 12683, 14439, 584, 4149, 755, 12232, 5, 1003, 15430, 20772, 14447, 14591, 12265, 20772, 6632, 8749, 20772, 1375, 19864, 5, 24335, 15571, 7756, 20772, 13641, 20772, 10142, 20772, 23242, 20772, 10205, 19073, 7939, 20772, 13658, 5, 2]
    #s = poem_generator.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    #exit(0)
    # КОНЕЦ ОТЛАДКИ


    udpipe = UdpipeParser()
    udpipe.load(models_dir)

    accents = Accents()
    accents.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
    accents.after_loading(stress_model_dir=os.path.join(tmp_dir, 'stress_model'))

    aligner = PoetryStressAligner(udpipe, accents)

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
            topic = input(':> ').strip()

            ranked_poems = generate_poems(topic)

            for poem, score in ranked_poems:
                print('\nscore={}'.format(score))
                for line in poem:
                    print(line)

                print('='*50)
