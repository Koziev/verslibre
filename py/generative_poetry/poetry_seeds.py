"""
Генератор затравок для генератора стихов и хайку.

30.01.2022 Добавлены затравки из файла key_extraction_dataset.txt
04.02.2022 Рефакторинг и переработка механизма: подготовка списка заранее, сохранение его в pickle-файле
"""

import random
import collections
import pickle
import os


class SeedGenerator(object):
    def __init__(self, models_dir):
        self.user_seeds = collections.defaultdict(set)
        with open(os.path.join(models_dir, 'seeds.pkl'), 'rb') as f:
            self.common_seeds = pickle.load(f)

    def generate_seeds(self, user_id):
        seeds = set()
        n_trial = 0
        while n_trial < 1000:
            n_trial += 1
            seed = random.choice(self.common_seeds)

            if seed not in self.user_seeds[user_id]:
                seeds.add(seed)
                self.user_seeds[user_id].add(seed)
                if len(seeds) >= 3:
                    break

        return list(seeds)

