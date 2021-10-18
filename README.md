# Генерация стихов с помощью больших языковых моделей


Репозиторий содержит код инференса для генерации коротких стихов.


## Генерация трехстрочников ("хайку")

![телеграм бот для генерации хайку](haiku_telegram.png)

Доступен докер-образ [inkoziev/haiku:latest](https://hub.docker.com/repository/docker/inkoziev/haiku) для запуска генератора как телеграм-бота.

Загружаем образ и запускаем:

```
sudo docker pull inkoziev/haiku:latest
sudo docker run -it inkoziev/haiku
```

Программа попросит ввести токен телеграм-бота. Затем загрузятся модели (примерно минута) и можно 
общаться с ботом. Вводите затравку - существительное или словосочетание. Генерация нескольких вариантов
на CPU идет примерно 30 секунд. Затем бот выведет первый вариант и предложит оценить его,
либо вывести следующий вариант.

Этот бот также доступен в телеграмме как [@haiku_guru_bot](http://t.me/haiku_guru_bot).


## Генерация четырехстрочников


![телеграм бот для генерации четырехстрочников](verslibre_telegram.png)

Бинарные файлы моделей из-за своего большого размера не выложены, но доступны
в докер-образе [inkoziev/verslibre:latest](https://hub.docker.com/repository/docker/inkoziev/verslibre).

Скачиваем и запускаем образ:

```
sudo docker pull inkoziev/verslibre:latest
sudo docker run -it inkoziev/verslibre:latest
```

После запуска программа запросит ввод токена для телеграм-бота.

После загрузки всех моделей можно запустить бота в его чате командой /start. Бот предложит выбрать одну из трех
случайных тем для сочинения либо ввести свою тему. Темой может быть любое
словосочетание с существительным в главной роли, например "генератор стихов".

Этот бот доступен в телеграмме как [@verslibre_bot](http://t.me/verslibre_bot)

Примеры генерации:

```
--- Прохладный ветерок ---

Прохладный ветерок,
Трепет и прохлада.
Я, как и прежде, не смог
Помолиться, глядя


--- А дождь проливал проливной ---

А дождь проливал проливной,
В окна дома глядел.
И я, как будто, был счастлив с тобой
И чего-то, чего-то хотел
```

Подробное описание всего пайплайна можно [прочитать тут](https://kelijah.livejournal.com/288594.html).


## Обучающие данные

В подкаталоге [tmp](https://github.com/Koziev/verslibre/tmp) лежат файлы с частью обучающих данных:

[poetry_corpus.txt](https://github.com/Koziev/verslibre/tmp/poetry_corpus.txt) - корпус отфильтрованных четверостиший, символ | в качестве разделителя строк; используется для дотренировки модели ruT5.  
[poem_generator_dataset.dat](https://github.com/Koziev/verslibre/tmp/poem_generator_dataset.dat) - датасет для тренировки ruGPT, выдающей текст стиха по теме (ключевому словосочетанию).  
[captions_generator_rugpt.dat](https://github.com/Koziev/verslibre/tmp/captions_generator_rugpt.dat) - датасет для тренировки ruGPT, генерирующей заголовок стиха по его содержимому.  

Описание процесса подготовки обучающего корпуса можно [найти здесь](https://kelijah.livejournal.com/288594.html).





