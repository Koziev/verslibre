"""
Код модели предсказания положения основного ударения в русских словах.
Файлы модели лежат в репозитории https://huggingface.co/inkoziev/accentuator
"""

import os
import re
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import huggingface_hub


class Vectorizer(object):
    def __init__(self):
        # спецсимволы для начала цепочки, конца и pad-символ
        self.bos_token = '['
        self.eos_token = ']'
        self.pad_token = ' '

        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    @staticmethod
    def from_pretrained(config):
        v = Vectorizer()
        v.max_len = config['max_len']
        v.char2index = config['char2index']
        v.num_outputs = config['num_outputs']
        return v

    def fit(self, samples):
        self.max_len = 0
        vocab = set()
        max_output = 0
        for sample in samples:
            max_output = max(max_output, sample['stress_pos'])
            vocab.update(sample['word'])
            self.max_len = max(self.max_len, 2+len(sample['word']))  # 2 символа для левой и правой границ

        self.char2index = {self.bos_token: self.bos_token_id,
                           self.eos_token: self.eos_token_id,
                           self.pad_token: self.pad_token_id}
        for c in vocab:
            if c not in self.char2index:
                self.char2index[c] = len(self.char2index)

        self.num_outputs = max_output + 1

    def encode(self, word):
        input_ids = [self.char2index.get(c, self.pad_token_id) for c in '[' + word.lower() + ']'][:self.max_len]
        l = len(input_ids)
        if l < self.max_len:
            input_ids += [self.pad_token_id] * (self.max_len - l)
        return torch.LongTensor([input_ids])


class AccentuatorModel(torch.nn.Module):
    def __init__(self, config):
        super(AccentuatorModel, self).__init__()

        embed_dim = config['embed_dim']

        self.embedding = torch.nn.Embedding(config['vocab_size'], embed_dim)
        self.arch = config['arch']
        if self.arch == 1:
            s0 = embed_dim * config['max_len']
            s1 = embed_dim * config['max_len']
            s2 = config['num_outputs']*2
            self.fc1 = torch.nn.Linear(in_features=s0, out_features=s1)
            self.fc2 = torch.nn.Linear(in_features=s1, out_features=s1)
            self.fc3 = torch.nn.Linear(in_features=s1, out_features=s2)
            self.fc4 = torch.nn.Linear(in_features=s2, out_features=config['num_outputs'])
        elif self.arch == 2:
            self.rnn = torch.nn.LSTM(input_size=embed_dim, hidden_size=embed_dim*2, num_layers=1, batch_first=True)
            self.fc1 = torch.nn.Linear(in_features=embed_dim*2, out_features=embed_dim*2)
            self.fc2 = torch.nn.Linear(in_features=embed_dim*2, out_features=config['num_outputs'])
        elif self.arch == 3:
            cnn_size = 200
            self.conv1 = torch.nn.Conv1d(embed_dim, out_channels=cnn_size, kernel_size=3)
            self.fc1 = torch.nn.Linear(in_features=cnn_size, out_features=config['num_outputs'])
        elif self.arch == 4:
            self.encoder = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
            self.fc1 = torch.nn.Linear(in_features=embed_dim*config['max_len'], out_features=config['num_outputs'])
        else:
            raise NotImplementedError()

    def forward(self, input_ids: torch.LongTensor, outputs: Optional[torch.LongTensor] = None):
        x = self.embedding(input_ids)

        if self.arch == 1:
            x = torch.flatten(x, start_dim=1, end_dim=-1)

            x = self.fc1(x)
            x = torch.relu(x)

            x = self.fc2(x)
            x = torch.relu(x)

            x = self.fc3(x)
            x = torch.relu(x)

            x = torch.nn.functional.dropout(input=x, p=0.1)

            x = self.fc4(x)
        elif self.arch == 2:
            out, (hidden, cell) = self.rnn(x)
            v = out[:, -1, :]
            x = self.fc1(v)
            x = torch.sigmoid(x)
            x = self.fc2(x)
        elif self.arch == 3:
            x = x.transpose(1, 2).contiguous()
            x = self.conv1(x)
            x = torch.relu(x).transpose(1, 2).contiguous()
            x, _ = torch.max(x, 1)
            x = self.fc1(x)
        elif self.arch == 4:
            x = self.encoder(x)
            x = torch.flatten(x, start_dim=1, end_dim=-1)
            x = self.fc1(x)
        else:
            raise NotImplementedError()

        y = torch.nn.functional.softmax(x, dim=1)

        if outputs is None:
            return y
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(input=y, target=outputs)
            return (loss, y)


class AccentuatorWrapper(object):
    def __init__(self, model_name_or_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if model_name_or_path == 'inkoziev/accentuator':
            config_filepath = huggingface_hub.hf_hub_download(repo_id=model_name_or_path, filename="config.json")
            model_filepath = huggingface_hub.hf_hub_download(repo_id=model_name_or_path, filename='pytorch_model.pth')
        else:
            config_filepath = os.path.join(model_name_or_path, 'config.json')
            model_filepath = os.path.join(model_name_or_path, 'pytorch_model.pth')

        with open(config_filepath, 'r') as f:
            self.config = json.load(f)

        self.vectorizer = Vectorizer.from_pretrained(self.config)
        self.model = AccentuatorModel(self.config)
        self.model.load_state_dict(torch.load(model_filepath, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, word):
        """Вернет порядковый номер гласной, на которую падает ударение"""
        input_ids = self.vectorizer.encode(word).to(self.device)
        y = self.model.forward(input_ids)
        i = torch.argmax(y).detach().item()

        nvowels = 0
        for ic, c in enumerate(word):
            if c.lower() in 'аеёиоуыэюя':
                nvowels += 1
                if ic == i:
                    return nvowels

        return -1

    def render_stress(self, word, stress_pos):
        out_chars = []
        nvowels = 0
        for c in word:
            out_chars.append(c)
            if c.lower() in 'аеёиоуыэюя':
                nvowels += 1
                if nvowels == stress_pos:
                    out_chars.append('\u0301')
        return ''.join(out_chars)


class AccentuatorWrapperWithDictionary(AccentuatorWrapper):
    def __init__(self, model_name_or_path='inkoziev/accentuator', device=None):
        super().__init__(model_name_or_path, device)
        if model_name_or_path == 'inkoziev/accentuator':
            dict_filepath = huggingface_hub.hf_hub_download(repo_id=model_name_or_path, filename='accents.pkl')
        else:
            dict_filepath = os.path.join(model_name_or_path, 'accents.pkl')

        with open(dict_filepath, 'rb') as f:
            self.ambiguous_accents = pickle.load(f)
            self.ambiguous_accents2 = pickle.load(f)
            self.word_accents_dict = pickle.load(f)
            self.yo_words = pickle.load(f)

    def predict2(self, word):
        # вернет варианты порядкового номера гласных, на которые может упасть ударение.

        if word in self.ambiguous_accents2:
            return self.ambiguous_accents2[word]

        if word in self.ambiguous_accents:
            stress_posx = []
            for accent in self.ambiguous_accents[word].keys():
                stress_pos = -1
                n_vowels = 0
                for c in accent:
                    if c.lower() in 'уеыаоэёяию':
                        n_vowels += 1

                    if c in 'АЕЁИОУЫЭЮЯ':
                        stress_pos = n_vowels
                        break

                if stress_pos == -1:
                    raise ValueError('Could not find stressed position in word "{}" in ambiguous_accents2'.format(accent))

                stress_posx.append(stress_pos)
            return stress_posx

        if word in self.word_accents_dict:
            return [self.word_accents_dict[word]]

        if re.match(r'^[бвгджзклмнпрстфхцчшщ]{2,}$', word):
            # Считаем, что в аббревиатурах, состоящих из одних согласных,
            # ударение падает на последний "слог":
            # ГКЧП -> Гэ-Ка-Че-П^э
            return [len(word)]

        return [self.predict(word)]


def render(word):
    px = accentuator.predict2(word)
    print(' | '.join(accentuator.render_stress(word, pos) for pos in px))


if __name__ == '__main__':
    accentuator = AccentuatorWrapperWithDictionary()
    render('моя')
    render('насыпать')
    render('кошка')
    render('ничегошеньки')



