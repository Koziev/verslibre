"""
Основной код инференса для модели генеративной поэзии, без привязки к фронту.

15-03-2022 Перенос сюда кода для управления генерацией через коррекцию финальных логитов по спискам
           позитивной и негативной лексики.

08.05.2022 Эксперимент с отдельным тегом для рубаи
25.05.2022 Для частушек (или возможно других жанров) в конце строк не отсекаются знаки препинания, поэтому не считаем
           повтором рифмовки ситуацию повтора пунктуатора перед <nl>
27.05.2022 В режиме генерации рубаи наказываются схемы рифмовки не-AABA
22.10.2022 Отбрасываем генерации, в которых появляются подстроки типа <s>
"""

import os
import json
import logging
import traceback
import warnings
import collections
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


from poetry.phonetic import Accents
from generative_poetry.udpipe_parser import UdpipeParser
from generative_poetry.poetry_alignment import PoetryStressAligner
from generative_poetry.experiments.rugpt_with_stress.break_to_syllables import break_to_syllables
from generative_poetry.experiments.rugpt_with_stress.arabize import arabize
from generative_poetry.experiments.rugpt_with_stress.stressed_gpt_tokenizer import StressedGptTokenizer
from generative_poetry.whitespace_normalization import normalize_whitespaces
from generative_poetry.metre_classifier import get_syllables


upper_cyr = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'


logits_booster = None


class ForceTopicWordsLogitsProcessor(transformers.LogitsProcessor):
    def __init__(self, positive_syllabic_ngrams, negative_syllabic_ngrams, gpt_tokenizer):
        self.positive_syllabic_ngrams = positive_syllabic_ngrams
        self.negative_syllabic_ngrams = negative_syllabic_ngrams
        self.gpt_tokenizer = gpt_tokenizer

        # для перевода слогов из целевой позитивной и негативной лексики в gpt-токены
        # нам нужно знать найти их без знаков ударения.
        self.syllab2tokens = collections.defaultdict(list)
        for token, token_id in gpt_tokenizer.vocab.items():
            unstressed_token = token.replace('\u0301', '').lower()
            self.syllab2tokens[unstressed_token].append((token, token_id))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        #boost_matrix = np.ones((scores.shape[0], scores.shape[1]), dtype=np.float32)
        boost_matrix = np.zeros((scores.shape[0], scores.shape[1]), dtype=np.float32)

        for iseq, seq_ids in enumerate(input_ids.tolist()):
            tx = [self.gpt_tokenizer.id2str[token_id] for token_id in seq_ids]

            if '$' in tx:
                tx = ['|'] + tx[tx.index('$')+1:]

            # Ищем, какие токены после правого края цепочки стоит усилить или придавить
            last1 = tx[-1]

            key = None
            if last1 in ('|', '<nl>'):
                # 1-граммы
                key = ('|',)
            else:
                if len(tx) > 1 and tx[-2] in ('|', '<nl>'):
                    # 2-граммы
                    key = ('|', tx[-1])

            if key is not None:
                # Позитивная лексика
                for next_token, score in self.positive_syllabic_ngrams.get(key, []):
                    for next_token, next_token_id in self.syllab2tokens.get(next_token, []):
                        boost_matrix[iseq, next_token_id] = 0.01*(len(key[1]) + len(next_token))  #pow(1.05, len(key[1]) + len(next_token))

                # Негативная лексика
                for next_token, score in self.negative_syllabic_ngrams.get(key, []):
                    for next_token, next_token_id in self.syllab2tokens.get(next_token, []):
                        boost_matrix[iseq, next_token_id] = -1.00*(len(key[1]) + len(next_token))  #pow(0.95, len(key[1]) + len(next_token))


        boost_matrix = torch.from_numpy(boost_matrix).to(scores.device)
        scores.add_(boost_matrix)
        #scores.mul_(boost_matrix)

        #... TODO ...

        #if self.static_bad_words_mask is None and len(self.bad_words_id_length_1) > 0:
        #    self.static_bad_words_mask = self._calc_static_bad_word_mask(scores)
        #dynamic_banned_tokens = self._calc_banned_bad_words_ids(input_ids.tolist())
        #scores = self._set_scores_to_inf_for_banned_tokens(scores, dynamic_banned_tokens)

        return scores


# https://arxiv.org/pdf/2202.00666.pdf
# https://github.com/cimeister/typical-sampling/blob/3e676cfd88fa2e6a24f2bdc6f9f07fddb87827c2/src/transformers/generation_logits_process.py#L242-L272
class TypicalLogitsWarper(transformers.LogitsWarper):
    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):

        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # shift and sort
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind[last_ind < 0] = 0
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores



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
):  # -> Union[SampleOutput, torch.LongTensor]:
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

    # ========================================================
    # 05-01-2022 эксперимент с бустером
    if logits_booster is not None:
        logits_processor.append(logits_booster)
    # конец эксперимента
    # ========================================================

    # НАЧАЛО ЭКСПЕРИМЕНТА С TYPICAL DECODING
    #logits_warper.clear()
    #logits_warper.append(TypicalLogitsWarper(mass=0.95))
    # КОНЕЦ ЭКСПЕРИМЕНТА С TYPICAL DECODING



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
        prompt_token_id = self.tokenizer.vocab['$']  # символ для отделения затравки и тела стиха

        punkt_ids = [self.tokenizer.vocab[c] for c in '.,!?;:-=()•…—–➖' if c in self.tokenizer.vocab]

        input_ids_2 = input_ids.cpu().numpy()
        # НАЧАЛО ОТЛАДКИ
        #input_ids_2 = np.asarray([[2938, 25608, 12894, 20772, 13641, 20772, 8969, 22282, 24705, 20772, 13641, 20772, 14627, 20772, 13641, 20772, 15751, 20772, 17874, 20772, 3638, 20772, 22030, 20772, 24341, 11959, 5, 25604, 20772, 1017, 19467, 20772, 3413, 10931, 9189, 20772, 18333, 20772, 12038, 19142, 20772, 24341, 20772, 20317, 5, 2938, 25608, 12894, 20772, 22030, 20772, 9382, 4235, 671, 20772, 17993, 20772, 20523, 14097, 12138, 20772, 6127, 20772, 13641, 20772, 6710, 20772, 9382, 11225, 20772, 20317, 5, 9783, 9245, 20772, 6920, 6345, 20772, 24975, 20772, 13641, 20772, 7355, 11225, 20772, 13641, 20772, 1003, 21359, 20772, 3372, 21333, 20772, 23719, 5, 2]], dtype=np.int)
        #sss = self.tokenizer.decode(input_ids_2[0, :], clean_up_tokenization_spaces=True)
        # КОНЕЦ ОТЛАДКИ
        nb_bad_rows = 0
        row_validity = [True] * input_ids_2.shape[0]
        for irow, row_ids in enumerate(input_ids_2):
            rhymes = []
            state = 'unknown'
            last_nl_pos = -1
            bad_row = False
            for j, x in enumerate(row_ids):
                if x in (prompt_token_id, nl_token_id):
                    state = 'nl_hit'
                    last_nl_pos = j
                elif x == break_token_id:
                    if state == 'nl_hit':  # нас интересует первое слово после "<nl>" или "$" (так как у нас цепочки right2left, то это фактически последнее слово в строке)
                        rhyme = ' '.join(map(str, row_ids[last_nl_pos+1: j])).strip()
                        # 25.05.2022 для частушек (или возможно других жанров) в конце строк не отсекаются знаки препинания,
                        # поэтому мы не должны реагировать на пунктуаторы в этом коде. В идеале надо бы пропустить пунктуатор и выделить
                        # следующее слово, и уже его использовать в проверке повтора рифмовки.
                        if rhyme and rhyme.count(' ') == 0 and int(rhyme) in punkt_ids:
                            state = 'break_hit'
                        else:
                            if rhyme in rhymes:
                                bad_row = True
                                nb_bad_rows += 1
                                break
                            else:
                                rhymes.append(rhyme)
                                state = 'break_hit'

            if bad_row:
                row_validity[irow] = False

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

        # НАЧАЛО ОТЛАДКИ
        if False:
            print('DEBUG@325')
            sep_id = self.tokenizer.vocab['$']
            X = input_ids.cpu().numpy()
            for i, row in enumerate(X):
                row = row.tolist()
                sep_pos = row.index(sep_id)
                row2 = row[sep_pos+1:]
                print('[{}] {}'.format(i, ', '.join(str(x) for x in row2)))
                print('{}'.format(self.tokenizer.decode(row2, clean_up_tokenization_spaces=True)))
            print('END OF DEBUG@332')
        # КОНЕЦ ОТЛАДКИ

        # inkoziev end
        # =============================================

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        if unfinished_sequences.shape[0] == 0 or input_ids.shape[0] == 0:
            #print('DEBUG@350')
            break

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    return input_ids




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
    #s = s.replace(' ,', ',').replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' …', '…').replace(' :', ':')
    s = normalize_whitespaces(s)
    return s



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
        self.model.eval()

    def generate_output(self, context, num_return_sequences=10,
                        method='sampling',
                        temperature=1.0,
                        top_k=30,
                        top_p=0.40,
                        typical_p=1.0,
                        num_beams=None,
                        positive_words=None, negative_words=None):
        global logits_booster

        repetition_penalty = 1.0
        prompt_text = "<s> " + context + ' $'
        stop_token = "</s>"
        length = 150

        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)
        max_len = length + len(encoded_prompt[0])

        # 15-05-2022 НАЧАЛО ЭКСПЕРИМЕНТА с управлением генерацией через логиты
        if positive_words is not None or negative_words is not None:
            positive_syllabic_ngrams = collections.defaultdict(list)
            if positive_words is not None:
                for word, score in positive_words.items():
                    sx = ['|'] + [x.text for x in get_syllables(word)][::-1]
                    if len(sx) > 1:
                        positive_syllabic_ngrams[sx[0]].append((sx[1], score))
                        if len(sx) > 2:
                            positive_syllabic_ngrams[(sx[0], sx[1])].append((sx[2], score))

            negative_syllabic_ngrams = collections.defaultdict(list)
            for word, score in negative_words.items():
                sx = ['|'] + [x.text for x in get_syllables(word)][::-1]
                if len(sx) > 1:
                    negative_syllabic_ngrams[sx[0]].append((sx[1], score))
                    if len(sx) > 2:
                        negative_syllabic_ngrams[(sx[0], sx[1])].append((sx[2], score))

            # может получится, что некоторые n-граммы входят и в позитивные, и в негативные.
            # такие нграммы мы просто исключим из списков, и не будем на них влиять.
            nx1 = set(positive_syllabic_ngrams.keys())
            nx2 = set(negative_syllabic_ngrams.keys())
            for k in nx1 & nx2:
                del positive_syllabic_ngrams[k]
                del negative_syllabic_ngrams[k]

            logits_booster = ForceTopicWordsLogitsProcessor(positive_syllabic_ngrams,
                                                            negative_syllabic_ngrams,
                                                            self.tokenizer)
        else:
            logits_booster = None
        # 15-05-2022 КОНЕЦ ЭКСПЕРИМЕНТА

        do_sample = method in ['sampling', 'typical decoding']

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=max_len,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_k=top_k if do_sample else None,
            top_p=top_p if do_sample else None,
            typical_p=typical_p if method == 'typical decoding' else 1.0,
            num_beams=num_beams if method == 'beam search' else None,
            #num_beam_groups=5,
            num_return_sequences=num_return_sequences,
            pad_token_id=0,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        # НАЧАЛО ОТЛАДКИ
        if False:
            x = output_sequences.cpu().numpy()
            for i, row in enumerate(x):
                print('[{}] {}'.format(i, ', '.join(str(x) for x in row[encoded_prompt.shape[1]:])))
                print('{}'.format(self.tokenizer.decode(row[encoded_prompt.shape[1]:], clean_up_tokenization_spaces=True)))
        # КОНЕЦ ОТЛАДКИ

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
            generated_sequences.add(text.strip().replace('<pad>', '').strip())

        return list(generated_sequences)


class RugptGenerator8(RugptGenerator):
    def generate_output(self, context, num_return_sequences=10, temperature=1.0):
        top_k = 30
        top_p = 0.85
        repetition_penalty = 1.0
        prompt_text = "<s> " + context + ' <nl>\n'
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


class PoetryGeneratorCore(object):
    def __init__(self, using_stresses=True):
        self.poem_generator = None
        self.poem8_generator = None
        self.udpipe = None
        self.accents = None
        self.aligner = None
        self.using_stresses = using_stresses

    def load(self, models_dir, data_dir, tmp_dir, poetry_generator_model_path=None):
        self.poem_generator = RugptGenerator()
        if poetry_generator_model_path is None:
            self.poem_generator.load(os.path.join(models_dir, 'stressed_poetry_generator'))
        else:
            self.poem_generator.load(poetry_generator_model_path)

        self.poem8_generator = RugptGenerator8()
        self.poem8_generator.load(os.path.join(models_dir, 'stressed_8liner_generator'))

        self.udpipe = UdpipeParser()
        self.udpipe.load(models_dir)

        self.accents = Accents()
        self.accents.load_pickle(os.path.join(tmp_dir, 'accents.pkl'))
        self.accents.after_loading(stress_model_dir=os.path.join(tmp_dir, 'stress_model'))

        self.aligner = PoetryStressAligner(self.udpipe, self.accents, os.path.join(data_dir, 'poetry', 'dict'))

    def detect_bad_substring(self, text):
        if any((s in text) for s in ['<s>']):
            return True
        return False

    def generate_poems(self, format, topic, verbosity=1, num_return_sequences=30,
                       positive_words=None, negative_words=None,
                       method='sampling', temperature=1.0, top_p=0.5, top_k=30, typical_p=1.0, num_beams=None):
        # НАЧАЛО ОТЛАДКИ
        #text = 'скажите ктулху так назвали ктулху'
        #a = self.aligner.align([text], check_rhymes=False)
        #r = self.aligner.detect_repeating(a, strict=True)
        # КОНЕЦ ОТЛАДКИ

        try:
            if self.using_stresses:
                seed = arabize(break_to_syllables(self.udpipe, self.accents, format + ' , ' + topic))
            else:
                seed = format + ' , ' + topic
                seed_tokens = []
                for word in seed.split(' '):
                    if seed_tokens:
                        seed_tokens.append('|')
                    for s in get_syllables(word):
                        seed_tokens.append(s.text)

                seed_tokens = seed_tokens[::-1]
                seed = ' '.join(seed_tokens)

            poems = self.poem_generator.generate_output(seed,
                                                        num_return_sequences=num_return_sequences,
                                                        positive_words=positive_words,
                                                        negative_words=negative_words,
                                                        method=method,
                                                        num_beams=num_beams,
                                                        temperature=temperature,
                                                        top_p=top_p,
                                                        top_k=top_k,
                                                        typical_p=typical_p)
        except Exception as ex:
            logging.error(ex)
            return []

        # ===============================================================
        # Отранжируем результат генерации по консистентности ритма...
        # ===============================================================
        ranked_poems = []

        prohibit_repetition = False  # полностью пресекать повтор леммы глагола и существительного (кроме некоторых служебных)

        if format in 'лирика|детский стишок|философия|юмор|мистика'.split('|'):
            do_check_rhymes = True
            score_threshold = 0.10
            lines_count = 4
        elif format in 'рубаи|частушка|Филатов|Пушкин|Крылов'.split('|'):
            do_check_rhymes = False
            score_threshold = 0.00
            lines_count = 4
        else:
            # для моностихов, двустрочников и порошков рифмовку не проверяем!
            do_check_rhymes = False
            score_threshold = 0.10
            if format == 'одностишье':
                lines_count = 1
                prohibit_repetition = True
            elif format == 'двустишье':
                lines_count = 2
                prohibit_repetition = True
            elif format == 'порошок':
                lines_count = 4
            else:
                logging.error('Unknown target poetry format "%s"', format)
                raise ValueError(format)

        for ipoem, poem in enumerate(poems):
            lines = [decode_line2(line) for line in poem.split('<nl>') if len(line) > 0]

            if len(lines) == lines_count:
                score = 0.0
                try:
                    if any(self.detect_bad_substring(line) for line in lines):
                        continue

                    a = self.aligner.align(lines, check_rhymes=do_check_rhymes)
                    if a is not None:
                        if a.rhyme_scheme == '----' and do_check_rhymes:
                            continue

                        score = a.score

                        if format == 'рубаи' and a.rhyme_scheme != 'AABA':
                            score *= 0.1

                        if self.aligner.detect_repeating(a):
                            # 13-01-2022 штрафуем за повторы
                            score *= 0.1
                            logging.warning('Repetition detected: %s', a.get_unstressed_lines().replace('\n', ' | '))

                        if self.aligner.detect_poor_poetry(a):
                            # скучные рифмы и повторы рифмуемого слова:
                            # В садах следы цветущих елей
                            #                        ^^^^
                            # Следы невиданных дубов и елей.
                            #                          ^^^^
                            score *= 0.1
                            logging.warning('Poor poetry detected: %s', a.get_unstressed_lines().replace('\n', ' | '))

                except Exception as e:
                    logging.error('Exception: %s', str(e) + '\n' + traceback.format_exc() + '\n')

                if score >= score_threshold:
                    if prohibit_repetition and self.aligner.detect_repeating(a, strict=True):
                        logging.warning('Prohibited repetition detected: %s', a.get_unstressed_lines().replace('\n', ' | '))
                    else:
                        ranked_poems.append((lines, score))
                elif score == 0.0:
                    if verbosity > 0:
                        logging.info('@451 === BAD GENERATION ===')
                        logging.info('Raw lines:')
                        for line in poem.split('<nl>'):
                            if len(line) > 0:
                                logging.info('%s', line)

                        logging.info('Decoded lines:')
                        for line in poem.split('<nl>'):
                            if len(line) > 0:
                                logging.info('%s', decode_line2(line, remove_stress_marks=False))

        ranked_poems = sorted(ranked_poems, key=lambda z: -z[1])
        return ranked_poems

    def continue8(self, poem_lines):
        """Для формата генерации обычных стихов добавляем 4 новые строки с помощью
         отдельной модели.
         Возвращается 9 строк, включая пустую разделительную строку между первыми и вторыми 4мя строками"""
        try:
            encoded_headlines = [arabize(break_to_syllables(self.udpipe, self.accents, line)) for line in poem_lines]
            encoded_seed = ' <nl> '.join(encoded_headlines)
            next_lines = self.poem8_generator.generate_output(encoded_seed, num_return_sequences=1)
            if next_lines:
                next_lines = [decode_line2(line) for line in next_lines[0].split('<nl>') if len(line) > 0]
                poem8 = poem_lines + [''] + next_lines
                return poem8
        except Exception as ex:
            logging.error(ex)
            logging.error(traceback.format_exc())

        return poem_lines
