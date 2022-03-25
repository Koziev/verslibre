"""
Эксперименты с PPLM поверх ruGPT.

20.03.2022 Первая реализация - форк от verslibre_pplm.py
23.03.2022 Вычисление лосса для негативной лексики через log(1-y)
23.03.2022 Расчет кол-ва употреблений лексики относительно числа слов в тексте, чтобы нивелировать
           вариативность числа слов при оценке эффективности
25.03.2022 Добавлена оценка качества работы в режиме использования классификатора (т.е. с  --discrim)
"""

#! /usr/bin/env python3
# coding=utf-8

# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import io
from operator import add
from typing import List, Optional, Tuple, Union
import os
import re
import math
import collections

import colorama
import terminaltables
import numpy as np
import torch
from torch import nn
from tqdm import trange

from pplm_classification_head import ClassificationHead
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.file_utils import cached_path
import tensorflow as tf

from rutokenizer import Tokenizer
from run_pplm_discrim_train__rugpt import Discriminator


PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10

BAG_OF_WORDS_ARCHIVE_MAP = {
    "legal": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    "military": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    "politics": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    "religion": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    "science": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    "space": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    "technology": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)


def perturb_past(
    past_12,
    model,
    last,
    unpert_past=None,
    unpert_logits=None,
    accumulated_hidden=None,
    grad_norms=None,
    stepsize=0.01,
    positive_one_hot_bows_vectors=None,  # 21-03-2022 inkoziev - отдельно учитываем "позитивную" и "негативную" лексику
    negative_one_hot_bows_vectors=None,
    classifier=None,
    class_label=None,
    loss_type=0,
    num_iterations=3,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    kl_scale=0.01,
    device="cuda",
):
    # в transformers==4.11.3 формат past_key_values сменился с тензоров на пары тензоров
    # склеим обратно половинки
    past = []
    for p0, p1 in past_12:
        p12 = torch.cat((torch.unsqueeze(p0, dim=0), torch.unsqueeze(p1, dim=0)), dim=0)
        past.append(p12)

    # Generate inital perturbed past
    grad_accumulator = [(np.zeros(p.shape).astype("float32")) for p in past]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(0.0, 1.0 + SMALL_CONST, 1.0 / (window_length))[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = tuple(past[0].shape[:-2]) + tuple([window_length]) + tuple(past[0].shape[-1:])

        zeros_key_val_shape = (
            tuple(past[0].shape[:-2]) + tuple([curr_length - window_length]) + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        #print("Iteration ", i + 1)
        #curr_perturbation = [torch.from_numpy(p_).requires_grad_(True).to(device=device) for p_ in grad_accumulator]
        curr_perturbation = [torch.from_numpy(p_).requires_grad_(True).to(device=device) for p_ in grad_accumulator]
        # make sure p_.grad is not None
        for p_ in curr_perturbation:
            p_.retain_grad()

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))

        _, _, _, curr_length, _ = curr_perturbation[0].shape
        lm_output = model(last, past_key_values=perturbed_past)
        all_logits, all_hidden = lm_output["logits"], lm_output["hidden_states"]
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = nn.functional.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []

        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            for one_hot_bow in positive_one_hot_bows_vectors:
                bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                bow_loss = -torch.log(torch.sum(bow_logits))  # отрицательный лосс - для стимулирования положительной лексики
                loss += bow_loss
                loss_list.append(bow_loss)

            t_ones = torch.ones_like(probs)
            for one_hot_bow in negative_one_hot_bows_vectors:
                bow_logits = torch.mm(torch.sub(t_ones, probs), torch.t(one_hot_bow))
                bow_loss = torch.log(torch.sum(bow_logits)) # положительный лосс - для избегания негативной лексики
                loss += bow_loss
                loss_list.append(bow_loss)

        if loss_type == 2 or loss_type == 3:
            ce_loss = nn.CrossEntropyLoss()
            # TODO why we need to do this assignment and not just using unpert_past? (Sumanth)
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            wte = model.resize_token_embeddings()
            for _ in range(horizon_length):
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                lm_output = model(past_key_values=curr_unpert_past, inputs_embeds=inputs_embeds)
                curr_all_logits, curr_unpert_past, curr_all_hidden = (
                    lm_output["logits"],
                    lm_output["past_key_values"],
                    lm_output["hidden_states"],
                )
                curr_logits = curr_all_logits[:, -1, :]
                curr_probs = nn.functional.softmax(curr_logits, dim=-1)
                curr_probs = torch.unsqueeze(curr_probs, dim=1)
                curr_hidden = curr_all_hidden[-1]
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(curr_hidden, dim=1)

            prediction = classifier(new_accumulated_hidden / (curr_length + 1 + horizon_length))

            label = torch.tensor(prediction.shape[0] * [class_label], device=device, dtype=torch.long)
            discrim_loss = ce_loss(prediction, label)
            #print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
            loss += discrim_loss
            loss_list.append(discrim_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = nn.functional.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = unpert_probs + SMALL_CONST * (unpert_probs <= SMALL_CONST).float().to(device).detach()
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * ((corrected_probs * (corrected_probs / unpert_probs).log()).sum())
            #print(" kl_loss", kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy().item())
        #print(" pplm_loss", (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        loss.backward()

        # calculate gradient norms
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST) for index, p_ in enumerate(curr_perturbation)
            ]

        # normalize gradients
        grad = [
            -stepsize * (p_.grad * window_mask / grad_norms[index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [torch.from_numpy(p_).requires_grad_(True).to(device=device) for p_ in grad_accumulator]
    pert_past = list(map(add, past, grad_accumulator))

    # теперь отформатируем обратно pert_past в список пар тензоров для transformers>=4.11.3
    pert_past_2 = [(p[0, :, :, :, :], p[1, :, :, :, :]) for p in pert_past]

    return pert_past_2, new_accumulated_hidden, grad_norms, loss_per_iter


def get_classifier(
    name: Optional[str], class_label: Union[str, int], device: str
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(class_size=params["class_size"], embed_size=params["embed_size"]).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified in the discriminator model parameters")
    classifier.load_state_dict(torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str], tokenizer) -> List[List[List[int]]]:
    bow_indices = list()
    for id_or_path in bag_of_words_ids_or_paths:
        if id_or_path in BAG_OF_WORDS_ARCHIVE_MAP:
            filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[id_or_path])
        else:
            filepath = id_or_path
        with open(filepath, "r") as f:
            words = f.read().strip().split("\n")
        #bow_indices.append([tokenizer.encode(word.strip(), add_prefix_space=True) for word in words])
        lx = set()
        for word in words:
            encoded_word = tokenizer.encode(word, add_special_tokens=False)

            # НАЧАЛО ЭКСПЕРИМЕНТА 23-03-2022 берем только первый BPE-токен слова, и если его символьное представление длиннее 2 букв.
            token1 = encoded_word[:1]
            stoken1 = tokenizer.decode(token1)
            if len(stoken1) >= 3:
                lx.add(tuple(token1))

            continue
            # КОНЕЦ ЭКСПЕРИМЕНТА

            lx.add(tuple(encoded_word))

        if lx:
            bow_indices.append(lx)

    return list(bow_indices)


def build_bows_one_hot_vectors(bow_indices, tokenizer, device="cuda"):
    if bow_indices is None or len(bow_indices) == 0:
        return []

    one_hot_bows_vectors = []

    if True:
        # Все 1-hot векторы упаковываем в один
        one_hot_bow = np.zeros(shape=(1, tokenizer.vocab_size), dtype=np.float32)
        for single_bow in bow_indices:
            for xx in single_bow:
                for x in xx:
                    one_hot_bow[0, x] = 1.0

        one_hot_bows_vectors.append(torch.tensor(one_hot_bow).to(device))
    else:
        # Исходная процедура порождает много 1-hot векторов, с модификацией - еще и для каждого слога в ключевых словах
        for single_bow in bow_indices:
            #single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
            single_bow2 = []
            for xx in single_bow:
                for x in xx:
                    single_bow2.append([x])

            #single_bow = torch.tensor(single_bow).to(device)
            single_bow = torch.tensor(single_bow2).to(device)

            num_words = single_bow.shape[0]
            one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
            one_hot_bow.scatter_(1, single_bow, 1)
            one_hot_bows_vectors.append(one_hot_bow)

    return one_hot_bows_vectors


def full_text_generation(
    model,
    tokenizer,
    context=None,
    num_samples=1,
    device="cuda",
    positive_bag_of_words=None,
    negative_bag_of_words=None,
    discrim=None,
    class_label=None,
    length=100,
    stepsize=0.02,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=3,
    grad_length=10000,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
    repetition_penalty=1.0,
    **kwargs
):
    classifier, class_id = get_classifier(discrim, class_label, device)

    positive_bow_indices = []
    if positive_bag_of_words:
        positive_bow_indices = get_bag_of_words_indices(positive_bag_of_words.split(";"), tokenizer)

    negative_bow_indices = []
    if negative_bag_of_words:
        negative_bow_indices = get_bag_of_words_indices(negative_bag_of_words.split(";"), tokenizer)

    if (positive_bag_of_words or negative_bag_of_words) and classifier:
        #print("Both PPLM-BoW and PPLM-Discrim are on. This is not optimized.")
        loss_type = PPLM_BOW_DISCRIM

    elif positive_bag_of_words or negative_bag_of_words:
        loss_type = PPLM_BOW
        #print("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        #print("Using PPLM-Discrim")

    else:
        raise Exception("Specify either a bag of words or a discriminator")

    unpert_gen_tok_text, _, _ = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        length=length,
        sample=sample,
        perturb=False,
        repetition_penalty=repetition_penalty,
    )
    if device == "cuda":
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    for _ in trange(num_samples, ascii=True):
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=device,
            perturb=True,
            positive_bow_indices=positive_bow_indices,
            negative_bow_indices=negative_bow_indices,
            classifier=classifier,
            class_label=class_id,
            loss_type=loss_type,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            repetition_penalty=repetition_penalty,
        )

        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if device == "cuda":
        torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time


def generate_text_pplm(
    model,
    tokenizer,
    context=None,
    past=None,
    device="cuda",
    perturb=True,
    positive_bow_indices=None,
    negative_bow_indices=None,
    classifier=None,
    class_label=None,
    loss_type=0,
    length=100,
    stepsize=0.02,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=3,
    grad_length=10000,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
    repetition_penalty=1.0,
):
    output_so_far = None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t

    # collect one hot vectors for bags of words
    positive_one_hot_bows_vectors = build_bows_one_hot_vectors(positive_bow_indices, tokenizer, device)
    negative_one_hot_bows_vectors = build_bows_one_hot_vectors(negative_bow_indices, tokenizer, device)

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []
    #for i in trange(length, ascii=True):
    for i in range(length):

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                past = model(output_so_far[:, :-1])["past_key_values"]

        lm_output = model(output_so_far)
        unpert_logits, unpert_past, unpert_all_hidden = (
            lm_output["logits"],
            lm_output["past_key_values"],
            lm_output["hidden_states"],
        )
        unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    positive_one_hot_bows_vectors=positive_one_hot_bows_vectors,
                    negative_one_hot_bows_vectors=negative_one_hot_bows_vectors,
                    classifier=classifier,
                    class_label=class_label,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        lm_output = model(last, past_key_values=pert_past)
        pert_logits, past = (
            lm_output["logits"],
            lm_output["past_key_values"],
        )
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST

        for token_idx in set(output_so_far[0].tolist()):
            if pert_logits[0, token_idx] < 0:
                pert_logits[0, token_idx] *= repetition_penalty
            else:
                pert_logits[0, token_idx] /= repetition_penalty

        pert_probs = nn.functional.softmax(pert_logits, dim=-1)

        if classifier is not None:
            ce_loss = nn.CrossEntropyLoss()
            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor([class_label], device=device, dtype=torch.long)
            unpert_discrim_loss = ce_loss(prediction, label)
            #print("unperturbed discrim loss", unpert_discrim_loss.data.cpu().numpy())
        else:
            unpert_discrim_loss = 0

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = nn.functional.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = (pert_probs ** gm_scale) * (unpert_probs ** (1 - gm_scale))  # + SMALL_CONST
            pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = nn.functional.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)

        #print(tokenizer.decode(output_so_far.tolist()[0], clean_up_tokenization_spaces=False))

    return output_so_far, unpert_discrim_loss, loss_in_time


def set_generic_model_params(discrim_weights, discrim_meta):
    if discrim_weights is None:
        raise ValueError("When using a generic discriminator, discrim_weights need to be specified")
    if discrim_meta is None:
        raise ValueError("When using a generic discriminator, discrim_meta need to be specified")

    with open(discrim_meta, "r") as discrim_meta_file:
        meta = json.load(discrim_meta_file)
    meta["path"] = discrim_weights
    DISCRIMINATOR_MODELS_PARAMS["generic"] = meta


def run_pplm_example(
    pretrained_model,
    cond_text="",
    uncond=False,
    num_samples=1,
    positive_bag_of_words=None,
    negative_bag_of_words=None,
    discrim=None,
    discrim_weights=None,
    discrim_meta=None,
    class_label=-1,
    length=100,
    stepsize=0.02,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=3,
    grad_length=10000,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
    seed=0,
    no_cuda=False,
    repetition_penalty=1.0,
):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    if discrim == "generic":
        set_generic_model_params(discrim_weights, discrim_meta)

    if discrim is not None:
        pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim]["pretrained_model"]
        print("discrim = {}, pretrained_model set to discriminator's = {}".format(discrim, pretrained_model))

    # load pretrained model
    #model = GPT2LMHeadModel.from_pretrained(pretrained_model, output_hidden_states=True)

    model = GPT2LMHeadModel.from_pretrained(pretrained_model, output_hidden_states=True)

    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '</s>'
    tokenizer.pad_token = '<pad>'
    #tokenizer = StressedGptTokenizer.from_pretrained(pretrained_model)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # figure out conditioning text
    if uncond:
        tokenized_cond_text = tokenizer.encode([tokenizer.bos_token])
    else:
        raw_text = cond_text
        while not raw_text:
            print("Did you forget to add `--cond_text`? ")
            raw_text = input("Model prompt >>> ")
        tokenized_cond_text = tokenizer.encode(tokenizer.bos_token + raw_text)

    #print("= Prefix of sentence =")
    #print(tokenizer.decode(tokenized_cond_text, clean_up_tokenization_spaces=False))
    #print()

    # generate unperturbed and perturbed texts

    # full_text_generation returns:
    # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        context=tokenized_cond_text,
        device=device,
        num_samples=num_samples,
        positive_bag_of_words=positive_bag_of_words,
        negative_bag_of_words=negative_bag_of_words,
        discrim=discrim,
        class_label=class_label,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        repetition_penalty=repetition_penalty,
    )

    # untokenize unperturbed text
    unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0][len(tokenized_cond_text):], clean_up_tokenization_spaces=False)
    if '</s>' in unpert_gen_text:
        unpert_gen_text = unpert_gen_text[:unpert_gen_text.index('</s>')]

    unpert_gen_text = unpert_gen_text.replace('<pad>', '').strip()

    #print("=" * 80)
    #print("= Unperturbed generated text =")
    #lines = [decode_line2(line) for line in unpert_gen_text.split('<nl>') if len(line) > 0]
    #for line in lines:
    #    print(line)
    #print()

    #generated_texts = []

    # iterate through the perturbed texts
    texts = []
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            # untokenize perturbed text
            # ... colorama ...
            pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0][len(tokenized_cond_text):])  # , clean_up_tokenization_spaces=True
            if '</s>' in pert_gen_text:
                pert_gen_text = pert_gen_text[:pert_gen_text.index('</s>')].strip()

            texts.append(pert_gen_text)
        except Exception as exc:
            print("Ignoring error while generating perturbed text:", exc)

        # keep the prefix, perturbed seq, original seq for each index
        #generated_texts.append((tokenized_cond_text, pert_gen_tok_text, unpert_gen_tok_text))

    return texts


def render_hits(text, posit_hits, negat_hits):
    res_text = text

    for word in posit_hits:
        res_text = re.sub(r'\b'+word+r'\b', colorama.Back.GREEN + colorama.Fore.WHITE + word + colorama.Style.RESET_ALL, res_text)

    for word in negat_hits:
        res_text = re.sub(r'\b'+word+r'\b', colorama.Back.RED + colorama.Fore.WHITE + word + colorama.Style.RESET_ALL, res_text)

    return res_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument("--cond_text", type=str, help="Prefix texts to condition on")
    parser.add_argument("--uncond", action="store_true", help="Generate from end-of-text as prefix")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )

    parser.add_argument("--positive_bag_of_words", "-PB", type=str, default=None)
    parser.add_argument("--negative_bag_of_words", "-NB", type=str, default=None)

    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        choices=("clickbait", "sentiment", "toxicity", "generic"),
        help="Discriminator to use",
    )
    parser.add_argument(
        "--discrim_weights",
        type=str,
        default=None,
        help="Weights for the generic discriminator",
    )
    parser.add_argument(
        "--discrim_meta",
        type=str,
        default=None,
        help="Meta information for the generic discriminator",
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sample", action="store_true", help="Generate from end-of-text as prefix")
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; 0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true", help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Penalize repetition. More than 1.0 -> less repetition",
    )

    # 19-03-2022 запрещаем тензорфлоу резервировать всю память в гпу по дефолту, так как
    # это мешает потом нормально работать моделям на торче.
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    args = parser.parse_args()

    # Загружаем другие компоненты уже из проекта verslibre
    #proj_dir = '/home/jovyan/polygon/text_generator'
    proj_dir = '/home/inkoziev/polygon/text_generator'
    tmp_dir = os.path.join(proj_dir, 'tmp')
    models_dir = os.path.join(proj_dir, 'models')
    data_dir = os.path.join(proj_dir, 'data')

    rtokenizer = Tokenizer()
    rtokenizer.load()

    # Списки с позитивной и негативной лексикой
    positive_bow_words = set()
    if args.positive_bag_of_words:
        with io.open(args.positive_bag_of_words, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                positive_bow_words.add(line.strip())

    negative_bow_words = set()
    if args.negative_bag_of_words:
        with io.open(args.negative_bag_of_words, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                negative_bow_words.add(line.strip())

    # Запускаем генерацию
    texts = run_pplm_example(**vars(args))

    # ======================================
    # Анализируем результаты генерации
    # ======================================

    # Если заданы списки позитивной или негативной лексики
    if args.negative_bag_of_words or args.positive_bag_of_words:
        positive_rate_list = []
        negative_rate_list = []

        # Для определения текста, в котором будет макс. число употреблений позитивной лексики
        best_text_by_posit_hits = None
        max_posit_hits = 0.0

        # Для определения текста, в котором будет макс. число употреблений негативной лексики
        worst_text_by_negat_hits = None
        max_negat_hits = 0.0

        for text in texts:
            positive_bow_hits = 0
            negative_bow_hits = 0
            posit_hits = set()
            negat_hits = set()
            words = rtokenizer.tokenize(text)
            for word in words:
                uword = word.lower()
                if uword in positive_bow_words:
                    positive_bow_hits += 1
                    posit_hits.add(word)
                if uword in negative_bow_words:
                    negative_bow_hits += 1
                    negat_hits.add(word)

            if positive_bow_hits > max_posit_hits:
                max_posit_hits = positive_bow_hits
                best_text_by_posit_hits = (text, posit_hits, negat_hits)

            if negative_bow_hits > max_negat_hits:
                max_negat_hits = negative_bow_hits
                worst_text_by_negat_hits = (text, posit_hits, negat_hits)

            len_words = float(len(words))
            posit_rate = positive_bow_hits / len_words
            negat_rate = negative_bow_hits / len_words
            positive_rate_list.append(posit_rate)
            negative_rate_list.append(negat_rate)

        if best_text_by_posit_hits is not None:
            print('Best text with {} positive rate:\n{}\n\n'.format(max_posit_hits, render_hits(*best_text_by_posit_hits)))

        if worst_text_by_negat_hits is not None:
            print('Worst text with {} negative rate:\n{}\n\n'.format(max_negat_hits, render_hits(*worst_text_by_negat_hits)))

        if best_text_by_posit_hits is None and worst_text_by_negat_hits is None:
            print('No hits in generated texts:\n')
            for i, text in enumerate(texts):
                print('[{}] {}\n'.format(i, text))

        print('Mean positive rate={}'.format(np.mean(positive_rate_list)))
        print('Mean negative rate={}'.format(np.mean(negative_rate_list)))

    if args.discrim_weights:
        # Задан классификатор.
        # Оцениваем, насколько хорошо результаты генерации соответствуют заданному целевому классу.

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        classifier, class_id = get_classifier('generic', 'joy', device)

        discriminator = Discriminator(class_size=6, pretrained_model=args.pretrained_model, cached_mode=False, device=device).to(device)
        discriminator.classifier_head = classifier
        tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model)

        idx2class = {0: "joy", 1: "sadness", 2: "surprise", 3: "fear", 4: "anger", 5: "none"}

        class_hits = collections.Counter()
        best_text = None
        best_score = 0.0
        for text0 in texts:
            text = args.cond_text + ' ' + text0
            tx = tokenizer.encode(text)
            ttx = torch.LongTensor(tx).unsqueeze(0).to(device)
            lprobas = discriminator(ttx)[0]
            class_px = [math.exp(y.item()) for y in lprobas]
            iclass = np.argmax(class_px)
            class_hits[iclass] += 1

            classif_score = class_px[args.class_label]
            if classif_score > best_score:
                best_score = classif_score
                best_text = text

        total = float(sum(class_hits.values()))
        table = [['class', 'hits', 'proba']]
        for iclass, hits in sorted(class_hits.items(), key=lambda z: z[0]):
            table.append((idx2class[iclass], hits, hits / total))

        table = terminaltables.AsciiTable(table)
        print(table.table)

        if best_text is not None:
            print('Best text: {}'.format(best_text))
