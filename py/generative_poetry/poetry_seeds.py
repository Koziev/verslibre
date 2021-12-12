import random
import collections
import itertools


user_seeds = collections.defaultdict(set)


a_m = ['зеленый', 'тусклый', 'шустрый', 'обворожительный', 'неожиданный', 'молодой', 'снисходительный',
'мудрый', 'безрассудный', 'застенчивый', 'вальяжный', 'сообразительный', 'воодушевленный',
'веселый', 'семейный', 'одинокий', 'незадачливый', 'агрессивный', 'настойчивый', 'толерантный',
'февральский', 'сентябрьский', 'ежедневный', 'пятничный', 'доброкачественный', 'кожаный',
'заунывный', 'занудный', 'звёздный', 'лунный', 'добрососедский', 'породистый', 'лютый',
'болтливый', 'бессмертный', 'задумчивый', 'воображаемый', 'вкрадчивый', 'симфонический', 'горбатый',
'нецензурный', 'душный', 'сказочный', 'мутный', 'небесный', 'невинный', 'вишнёвый', 'слепой', 'усталый',
'напевный', 'птичий', 'усатый', 'богатырский', 'лиственный', 'сказочный', 'страстный', 'безудержный',
'седой', 'добродушный', 'музыкальный', 'беспристрастный', 'вертлявый', 'рукотворный', 'зловещий',
'синеокий', 'первозданный', 'богемный', 'тлетворный', 'заморский', 'причудливый',
]

n_m = ['дом', 'аванс', 'ковид', 'друг', 'идиот', 'ребенок', 'воробей',
'лист', 'сон', 'день', 'идиот', 'дуб', 'тополь', 'зуб', 'дьявол', 'бог',
'кашель', 'кошелек', 'путь', 'тупик', 'бродяга', 'интеллект', 'совет', 'тупик', 'ктулху',
'зачёс', 'маникюр', 'понедельник', 'вторник', 'четверг', 'январь', 'февраль',
'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь',
'декабрь', 'микроб', 'кузнечик', 'вурдалак', 'ухряб', 'кабанчик', 'стон', 'вскрик',
'самурай', 'сабантуй', 'апофеоз', 'автопортрет', 'юмор']


a_f = ['престарелая', 'угрюмая', 'очаровательная', 'снисходительная', 'воодушевленная', 'застенчивая', 'привлекательная',
'мрачная', 'волнительная', 'оранжевая', 'доверительная', 'родная', 'замерзшая', 'уставшая',
'коренная', 'кокетливая', 'игривая', 'полуночная', 'апрельская', 'октябрьская', 'вечерняя', 'субботняя',
'зловредная', 'силиконовая', 'солнечная', 'вечнозелёная', 'сапфировая', 'храбрая', 'колченогая', 'суровая',
'зимняя', 'космическая', 'лживая', 'послеобеденная', 'голубоглазая', 'хромая', 'беспородная', 'лютая',
'вдумчивая', 'измождённая', 'кудрявая', 'джазовая', 'пышнотелая', 'розовощёкая', 'пышногрудая',
'виртуальная', 'лазурная', 'яркая', 'желчная', 'прозрачная', 'благая', 'пенная', 'легендарная', 'неприступная',
'шальная', 'тягучая', 'суровая', 'луноликая',
]

n_f = ['вакцина', 'прививка', 'девушка', 'женщина', 'идея', 'ночь', 'голова',
'дурочка', 'рябина', 'ива', 'чайка', 'смерть', 'жизнь', 'судьба', 'рифма',
'кровь', 'дорога', 'могила', 'еда', 'гроза', 'осень', 'зима', 'весна',
'прическа', 'беседа', 'рекомендация', 'коза', 'подружка', 'совесть',
'ресничка', 'расческа', 'пятница', 'суббота', 'кракозябра', 'оттепель', 'капель',
'депрессия', 'инфляция', 'опухоль', 'эпитафия', 'импровизация', 'девушка']


a_n = ['обезвоженное', 'счастливое', 'последнее', 'новое', 'фиолетовое', 'мерцающее', 'снисходительное',
'кредитное', 'культурное', 'иностранное', 'ночное', 'утреннее', 'злорадное', 'игривое', 'обеденное',
'синеглазое', 'июльское', 'декабрьское', 'утреннее', 'воскресное', 'полноводное', 'застенчивое',
'райское', 'чистосердечное', 'любовное', 'косматое', 'смутное', 'нервное', 'глазастое', 'неисхоженное',]

n_n = ['утро', 'похмелье', 'зелье', 'лекарство', 'окно', 'чудо',
'явление', 'марево', 'зарево', 'счастье', 'лето', 'очарование', 'вдохновение', 'воскресенье',
'зазеркалье', 'харакири', 'анимэ', 'послевкусие', 'пробуждение', 'смятение', 'сердцебиение'
]


a_p = ['железные', 'острые', 'последние', 'верные', 'ржавые', 'кургузые', 'простые', 'забытые', 'семейные', 'вечные',
'сладострастные', 'опухшие', 'январские', 'мартовские', 'алюминиевые', 'запредельные', 'сумасбродные',
'непокорные', 'первородные', 'запредельные', 'ошалевшие', 'светлоокие', 'гранитные', 'семейные', 'школьные',
'бесстрашные', 'будничные', 'апрельские', 'непоседливые', 'юные', 'многоголосые', 'ласково-сонные', 'кудрявые',
'незрелые',]

n_p = ['пассатижи', 'ножницы', 'дожди', 'деньги', 'друзья', 'долги', 'слезы',
'мысли', 'люди', 'листья', 'березы', 'волосы', 'воробьи', 'чайки', 'клочки',
'птицы', 'заморозки', 'замыслы']


n_gen = ['судьбы', 'страданий', 'любви', 'страсти', 'раздумий', 'одиночества', 'ночи', 'дьявола',
         'лета', 'осени', 'мечтаний']

collocs2 = ['русский чай', 'ласковый стон', 'весенние приметы', 'дешёвый шоколад',
'седой дирижёр', 'добродушный паучок', 'опасный рассвет',
'музыкальный свист', 'вертлявый бес', 'мысленный волк',
'холодный обрывок', 'сжатый размах', 'решительная драма',
'рукотворный рельеф', 'необъятная природа', 'внутренний иностранец',
'весенняя гроза', 'людская кровь', 'небесный купол', 'светлый ангел',
'чужие лица', 'звёздный свет', 'осенняя прохлада', 'исторический миг',
'белый дым', 'долгожданный век', 'молодёжный смех', 'вчерашний король',
'юные надежды', 'шальная слава', 'жёлтые листья', 'праздный бег',
'последний снег', 'первозданный ручей', 'вражий стан', 'небесная власть',
'желанный лучик', 'торжественный парад', 'безответная любовь',
'богемный сброд', 'тлетворный запашок', 'кареглазая соседка',
'глупый мяч', 'счастливая минута', 'багрово-красная река',
'безбожная поправка', 'пустая канитель', 'звездный кит',
'неисхоженное поле', 'тяжкий груз', 'суровая улика', 'зелёный июнь',
'торжественный наряд', 'безвредная трусиха', 'тупая обида',
'луноликая мечта', 'причудливый узор', 'добродушный человек',
'ярый звездопад', 'теплый ветер', 'цветущий сад', 'огненная стружка',
'чёрная река', 'стеклянная стена', 'перевернутый газон',
'незваный гость', 'тупой взгляд', 'земная реальность',
'печальный взлёт', 'мягкая кошка', 'многоголосые леса', 'вечная драка',
'горячий сон', 'холодная интриганка', 'непростая жизнь',
'здешние цветы', 'живая душа', 'прекрасный вид', 'верхние ноты',
'инородное тело', 'стотысячный клиент', 'древний дух',
'старый драндулет', 'папиллярный узор', 'солёный пот',
'воскресный день', 'кудрявые лозы', 'злой вечер', 'странное тело',
'яркие хвосты', 'незаслуженные плети', 'седой месяц',
'прощальная фраза', 'сырой ветер', 'пятый класс', 'божья благодать',
'старый дом', 'изысканный гранит', 'священная война',
'нездоровая бледность', 'незрелые юнцы', 'почтенный возраст',
'тёплый бриз', 'прошедшее счастье', 'культурный шок', 'крутые скалы',
'собачий лай', 'огненный шатер', 'сплошная суета', 'дряхлый дуб',
'вечное окно', 'янтарное колечко', 'высший суд', 'тонкий стан',
'мерзкая крыса', 'звездный кот', 'дивная краса', 'злая ночь',
'беззвучная ночь', 'расписные купола', 'седой итог', 'тёплый сон',
'березовый запах', 'непокорный завиток', 'добрые боги',
'пожелтевшая прелесть', 'лунный кристалл', 'промозглый вой',
'злые морозы', 'пустые дни', 'пламенный хорал', 'здоровый зуд',
'небесная прямота', 'трудный подросток', 'невинные младенцы',
'разноцветные олени', 'свободный дух', 'семейные преданья',
'серебряная рябь', 'пустая плоть', 'бескрайняя тайга', 'жаркое солнце',
'оранжевый глаз', 'сердечные заботы', 'безумная ложь',
'дождевые слизни', 'надежный друг', 'родные руки', 'точный дым',
'кудрявый локон', 'прохладная водица', 'пустынный путь',
'морское диво',]


def generate_seeds(user_id):
    seeds = set()

    if random.random() > 0.85:
        seeds.append(random.choice(collocs2))

    for _ in range(100):
        template = random.choice('an_m an_f an_n an_p n_n'.split())
        seed = None
        if template == 'an_m':
            a = random.choice(a_m)
            n = random.choice(n_m)
            seed = a + ' ' + n
        elif template == 'an_f':
            a = random.choice(a_f)
            n = random.choice(n_f)
            seed = a + ' ' + n
        elif template == 'an_n':
            a = random.choice(a_n)
            n = random.choice(n_n)
            seed = a + ' ' + n
        elif template == 'an_p':
            a = random.choice(a_p)
            n = random.choice(n_p)
            seed = a + ' ' + n
        elif template == 'n_n':
            n = random.choice(list(itertools.chain(n_m, n_f, n_n, n_p)))
            n2 = random.choice(n_gen)
            seed = n + ' ' + n2

        if seed and seed not in user_seeds[user_id]:
            seeds.add(seed)
            user_seeds[user_id].add(seed)
            if len(seeds) >= 3:
                break

    return list(seeds)
