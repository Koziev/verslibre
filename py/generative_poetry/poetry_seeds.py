"""
Генератор затравок для генератора стихов и хайку.

30.01.2022 Добавлены затравки из файла key_extraction_dataset.txt
04.02.2022 Рефакторинг и переработка механизма: подготовка списка заранее, сохранение его в pickle-файле
04.04.2022 Первая (для данного пользователя) порция седжестов генерируется особым образом, а не берется
           из подготовленного списка.
03.05.2022 Отдельный список затравок для бусидо
"""

import random
import collections
import pickle
import os
import datetime


class SeedGenerator(object):
    def __init__(self, models_dir):
        self.user_seeds = collections.defaultdict(set)
        # Списки саджестов грузится из заранее сформированного файла.
        with open(os.path.join(models_dir, 'seeds.pkl'), 'rb') as f:
            self.common_seeds = pickle.load(f)
            self.month_2_data = pickle.load(f)
            self.busido_seeds = pickle.load(f)

    def generate_seeds(self, user_id, domain=None):
        seeds = set()
        if domain and domain == 'бусидо':
            # Для бусидо сначала пытаемся использовать отдельный список.
            n_trial = 0
            while n_trial < 20:
                n_trial += 1
                seed = random.choice(self.busido_seeds)

                if seed not in self.user_seeds[user_id]:
                    seeds.add(seed)
                    self.user_seeds[user_id].add(seed)
                    if len(seeds) >= 3:
                        break
            if len(seeds) >= 2:
                return list(seeds)

        if len(self.user_seeds) == 0:
            # 04.04.2022 отдельная ветка генерации первой порции саджестов.
            cur_month = datetime.datetime.now().month
            adj_m, adj_f, adj_n, adj_p, noun_m, noun_f, noun_n, noun_p = self.month_2_data[cur_month]

            for _ in range(3):
                adjs, nouns = random.choice([(adj_m, noun_m), (adj_f, noun_f), (adj_n, noun_n), (adj_p, noun_p)])
                adj = random.choice(adjs)
                noun = random.choice(nouns)
                colloc = adj + ' ' + noun
                seeds.add(colloc)
                self.user_seeds[user_id].add(colloc)
        else:
            # Вторая и последующие порции саджестов.
            if len(seeds) < 3:
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

