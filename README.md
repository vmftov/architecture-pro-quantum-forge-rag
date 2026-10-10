# Задание 1. Исследование моделей и инфраструктуры

# Задание 2. Подготовка базы знаний

- В качестве исходных файлов для базы знаний взяты HTML-страницы с сайта `starwars.fandom.com` .
- Выбрано 33 страницы по ключевым темам, включая: персонажей, планеты, транспорт, оружие, яркие события. Каждая страница содержит информаццию об одном специфическом персонаже или объекте вселенной star wars. Полный список загруженных URL можно посмотреть в файле `solution\download_to_s3\download_queue.txt`.
- Написаны python скрипты, выполняющие в автоматическом режиме:
    - загрузку станиц,
    - выделение полезной информации в виде markdown файла,
    - анонимизацию (замену реальных имен персонажей, планет, организаций и т.д.) на вымышленные синтетические имена.

## Результат

- Сформированная база знаний содержится в папке `task_2\knowledge_base`. Разделение на подпапки условное (сделано для удобства) и не влияет на работу алгоритмов.
- В файле `task_2\terms_map.json` содержится массив соответствия оригинальных и фиктивных имён.
- В проекте `solution\download_to_s3` код загрузки HTML файлов и размещения их в S3 бакете.
- В файле `solution\process_docs\anonimizer.py` в проекте `solution\process_docs` располагается логика анонимизации.
- В файле `solution\process_docs\app.py` в проекте `solution\process_docs` располагается логика обработки документов.

## Процесс подготовки базы знаний

Python-скрипты в автоматическом режиме преобразуют исходные HTML файлы.

Процесс анонимизации встроен в общий пайплайн обработки документов для RAG. В учебном примере он служит для формирования уникальной базы знаний. В реальном приложении он будет служить для выявления и отбраковки или замены чувствительных персональных данных, паролей и другой конфиденциальной информации или информации с ограниченным доступом.

### Общий алгоритм работы с документами

- Исходные документы компании, с которыми должа работать RAG-система собираются, с помощью адаптеров, со всех источников в единый S3 бакет.
- Время от времени фоновый процесс проверяет наличие изменений и запускает процедуру обработки документов.
- С помощью библиотек (`BeautifulSoup`, `trafilatura`, `markdown_it`) по заданным правилам из HTML страниц выделяется полезное содержимое в виде Markdown текста.
- Выделенный текст проходит процедуру анонимизации. Выбранные типы сущностей удаляются или заменяются на вымышленные значения.
- Анонимизированные документы сохраняются в отдельный S3 бакет для дальнейшей векторизации.
- Исходные документы могут удаляться из S3 в зависимости от требований безопасности. В учебной работе они оставлены.

### Алгоритм анонимизации

- Для извлечения имён и типов сущностей из текста используются библиотеки `piighost` и `gliner2`.
- В качестве LLM модели используется мультиязычная `fastino/gliner2-multi-v1` с входным окном `512 токенов`. Она небольшая по размеру и хорошо подходит как для прототипов, так и для небольших баз знаний.
- Замене подвергаются сущности следующих типов: "first_name", "middle_name", "last_name", "droid", "race", "ethnicity", "organization", "weapon", "planet", "location", "transport", "event".
- В качестве порога определения типов сущностей, путем экспериментов, выбрано значение `threshold=0.42`. Это значение позволяет gliner2 выбрать достаточное количество сущностей (с запасом) и не породить много 'мусора'.
- Для определения нормальных форм и типов слов используется библиотека `spacy` с LLM `en_core_web_sm` для простоты. В реальности, для международной компании, стоит использовать более мощную мультиязычную модель.
- В пайплайн `piighost` встроен кастомный модуль фильтрации сущностей по стоп-словам по скорректированному словарю `english top 5000` для борьбы с ошибками детекции и предотвращения замены обычных слов.
- Чтобы исходный текст не обрезался, а также избегания OutOfMemory, он подаётся в LLM чанкам. Для сохранения смысла текста, он делится сначала на абзацы, затем, при необходимости, на предложения и только если этого не достаточно, на более мелкие части согласно входному окну модели.
- В процессе замены имён, пары значений сохраняются в словаре и впоследствии выгружаются в специальный файл `terms_map.json` в S3 бакет. Это сделано для простоты. В реальной системе, для поддержания консистентности и возможности масштабирования их следует хранить в базе данных. В учебном примере, чтобы не усложнять проект, они хранятся в S3 в json файле. Данный файл загружается и обновляется при каждой синхронизации с исходными файлами и позволяет сохранить соответствие терминов и смысл между документами.
- Для генерации фиктивных значений используются библиотека `faker` и простой кастомный алгоритм комбинирования слогов, т.к. `faker` не поддерживает необходимые типы сущностей и не справляется с большим числом замен.

Общее количество заменённых уникальных сущностей - порядка 5000. Путем подбора параметров и с помощью алгоритмов, удалось существенно уменьшить количество 'мусора' в сущностях, однако некоторые недоработки всё де остались. Т.к. основна цель - создание неповторимого текста по возможности автоматическим способом, эти минусы пропущены в рамках учебного задания. В реальной системе их стоит отлавливать с помощью специальных предметно-ориентированных словарей, проверяющих LLM моделей и ручным способом.

# Задание 3. Создание векторного индекса базы знаний

## Модель

- В качестве embedding-модели для энкодинга выбрана `BAAI/bge-base-en-v1.5`.
    - Страница `https://huggingface.co/BAAI/bge-base-en-v1.5`
    - Входное окно `512 токенов`
    - Размер вектора `768`
    - Модель небольшая `~ 0.1B параметров`. Однако, ввиду того, что размер базы знаний также небольшой это позволило получить удовлетворительные результаты и при этом ускорить формирование индекса, что важно для экспериментов на этапе прототипирования.
    - Модель рассчитана на `английский язык`, что соответствует тексту базы знаний.
    - Бесплатная (MIT)

- Для реальной работы с документами в международной компании, особенно с учётом типа бизнеса SAAS и большого количества документов с разрозненной информацией, следует выбирать более большую мультиязычную модель, например `BAAI/bge-m3`.
    - Страница `https://huggingface.co/BAAI/bge-m3`
    - Входное окно `8192 токена`
    - Размер вектора `1024`
    - Поддерживает три режима: dense, sparse и multi-vector (ColBERT) и упрощает пайплайн.
    - Мультиязычная
    - Бесплатная (MIT)
    - Повсеместно используется и имеет высокие показатели в бенчмарках

## Статистика

- Embedding-модель - BAAI/bge-base-en-v1.5
- База знаний:
    - 33 документа (3_648_827 символов)
    - Язык - английский
    - Формат - markdown
- Чанков (точек) в индексе - 3113
- Время формирования эмбеддингов ~ 140 сек

## Результат

- В файле `task_3\docs-3656266113994161-2026-10-10-00-37-36.snapshot` содержится дамп индекса `qdrant`.
- В файле `solution\process_docs\common\textsplitter.py` содержится код разбиения текста на чанки.
- В файле `solution\process_docs\common\textencoder.py` содержится код формирования эмбеддингов.
- В файле `solution\process_docs\common\qdrantsearcher.py` содержится код сохранения, поиска и управления векторами и метаданными.
- В файле `solution\process_docs\app.py` содержится общая логика обработки документа.

## Пример поискового запроса

Запрос:

```
What is Aculuab Voviqawo known for.
```

\* Aculuab Voviqawo это новое имя для Han Solo.

Чанки:

```
# Aculuab Voviqawo

Aculuab Voviqawo, known only as Aculuab until being given the surname Voviqawo, was a human male smuggler and hero of the Alliance to Restore the Republic. He rose from an impoverished childhood to become captain of the Viloihe Daifbeibjuz, a leader in the Rebel Graham-Dominguez and an instrumental figure in the Carey-Blanchard Civil War during the reign of the Carey-Blanchard Empire. Although he did not believe in the Force controlling his destiny, Aculuab came to believe in the cause of galactic freedom, joining Jedi Feceiponu Skywalker and Princess Gohfoezcoxeuy Organa in the fight for the Rebellion. Following the war, Aculuab and Gohfoezcoxeuy welcomed a son, Zuyakufog Voviqawo, who became the dark knight Kylo Wegimsaxem after being corrupted by the dark side of the Force. Although their relationship floundered, Gohfoezcoxeuy hoped that Aculuab, as Ben's father, could find a way to break the dark side's hold on their son. Aculuab Voviqawo became legendary across the galaxy as both a smuggler and war hero of the Rebellion.
```

```
When Zuyakufog joined the First Order and became Puofeziseq Ren, Voviqawo and Zadawu, though married, separated to cope with their grief. When Aculuab found himself pulled back into the fight, the two of them reunited briefly. Zadawu was convinced that Voviqawo could rescue his son from the dark side and bring him home. When Voviqawo was killed by their son, Zadawu sensed his death and cried for him. Zadawu continued to remember Voviqawo and still believed he was the greatest pilot.

## Skills and abilities
```

```
While most others regarded the ship as a piece of flying junk, Voviqawo saw the Millennium Falcon as the embodiment of classic design, an opinion he shared with Chewbacca, and considered it his home. Following the battle of Zamefih, Aculuab endeavored to honor Ufzuintaislo's life debt to him by liberating the enslaved Gaiyge homeworld from Imperial occupation. Despite several setbacks, Aculuab persisted with his mission. Together, Aculuab and Ufzuintaislo led a combined force of Gaiyge, rebels, and mercenaries to spark a successful Gaiyge uprising against the empire. During the campaign, Aculuab displayed resourcefulness and proved a cunning and effective commander. After the liberation of Mohbas, Aculuab freed Chewbacca of his life debt so that he could reunite with his family.

Aculuab Solo initially found the task of fatherhood challenging. While he envied Leia's Force connection to their son, Zuyakufog Solo, Aculuab found that he had a special bond as a father. Due to their different interests and personalities, Voviqawo and his wife lived separate lives. While she served as a New Republic senator, Voviqawo managed a ship company and organized starship races. Still, Voviqawo kept in regular contact with his wife. On one occasion, he rescued Leia from a pro-Imperial militia known as the Ivoowice warrior.
```

## Особенности алгоритма и хранения

**Разбиение на чанки:**

- Для разбиения на чанки используется `RecursiveCharacterTextSplitter` из `langchain`.
- При разбиении автоматически определяется размер окна модели (через API) и выбирается 80% для учета спецтокенов и погрешностей.
- Длина чанка ограничена сверху 1024 токенами, даже если модель позволяет больше, т.к. в статьях пишут, что большие чанки не приводят к улучшению результатов для retrieval-задач.
- Перекрытие чанков - 20%.

**Энкодинг:**

- Все эмбеддинги автоматически нормализуются.

**Индекс:**

- Для хранения эмбеддингов выбран `qdrant`.
    - Open-source.
    - Он легко настраивается.
    - Имеет удобный UI.
    - Позволяет выполнять не только векторный поиск но и фильтрацию.
    - За счёт дампов позволяет легко перейти на облачные решения.
    - Имеет встроенную поддержку масштабирования и шардирования.
- Вместе с эмбеддингом сохраняюся следующие метаданные:
    - `id: str` - идентификатор чанка
    - `doc_id: str` - путь к документу в S3
    - `doc_name: str` - имя документа
    - `category: str` - категория документа
    - `language: str` - язык документа
    - `modified_at: datetime` - дата изменения документа
    - `start_index: int` - начальный отступ чанка в тексте
    - `end_index: int` - конечный отступ чанка в тексте
    - `text: str` - текст чанка
- Т.к. база знаний небольшая, все необходимые данные, включая текст, сохраняются вместе с эмбеддингом. Это упрощает архитектуру и позволяет устранить необходимость в базе данных. Включение опции хранения метаданных на диске позволяет уменьшить расход оперативной памяти.
- `doc_id` используется для удаления чанков при синхронизации.
- `doc_name` подставляется в качестве источника при постпроцессинге, при выдаче результатов поиска пользователю.
- Некоторые метаданные (category, language, modified_at) не используются в решении но могут быть использованы для реализации комбинированного поиска (фильтрации) в будущем.

# Задание 4. Реализация RAG-бота с техниками промптинга

## Результат

- В папке `solution\api` содержится код проекта REST API RAG-помощника.
- В файле `solution\api\app.py` содержится код реализации REST API RAG-помощника.
    - Для реализации REST API используется `fastapi`.
    - Доступен swagger по пути `/docs`.
    - Для работы потребуется передача через переменные окружения folder_id и api_key из yandex cloud.
- В файле `solution\api\prompts\system_prompt.md` содержится системный промпт.
- В файле `solution\api\prompts\few_shot_prompts.yaml` содержатся примеры запросов для Few-shot prompting.
- В файле `solution\api\promptmanager.py` содержится код формирования промпта.
- Имеется файл `solution\docker-compose.yaml` для запуска приложения в docker.
    - Содержит инструкции сборки и развёртывания приложения.
    - Содержит инструкции развёртывания векторной БД `qdrant`.
    - Содержит инструкции развёртывания S3 `minio` для работы логики синхронизации файлов (не нужна для работы API).

## LLM Модель

В качестве LLM модели выбрана `yandexgpt-5.1/latest`.

- Небольшие локальные модели на 2 - 6 млрд. параметров плохо справляются с анализом, и выполняют его долго. У них ограниченный контекст и входное окно токенов. Из-за этого придётся обрезать найденные чанки и делать менее безопасный промпт.
- Использование облачного провайдера позволяет очень быстро протестировать различные вырианты LLM, что чрезвычайно важно на этапе прототипирования. Такой подход позволяет путем простой смены имени модели проанализировать работу разных моделей и выбрать наиболее подходящий вариант модели по размеру и цене, который лучшим образом справляется а анализом предметной области.

## Промпт

**Системный промпт:**

- Системный промпт располагается в файле `solution\api\prompts\system_prompt.md`.
- Он содержит инструкции для реализации подхода Chain-of-Thought.
    ```
    - В разделе 'Рассуждение' выведи цепочку своих рассуждений (Chain-of-Thought). Не выводи много текста, только 2-5 предложений с ключевыми рассуждениями.
    ```

**Few-shot prompting:**

- Примеры ответов для подхода Few-shot prompting располагаются в файле `solution\api\prompts\few_shot_prompts.yaml`.
- Примеры ответов включаются в чат после системного промпта и перед контекста и вопроса пользователя.
- Примеры реально извлечены из базы знаний и слегка переработаны.

## Алгоритм

- API содержит единственный endpoint - `POST /rag/questions`.
- API принимает запрос пользователя и проверяет его.
- Формируется эмбеддинг моделью `BAAI/bge-base-en-v1.5`.
- Выполняется поиск чанков в `qdrant`.
- Проверяется, что чанки не содержат опасное содержимое:
    - В данном учебном примере выполняется наивная проверка и поиск стоп-слов, например, 'пароль'.
    - В реальной модели следует использовать специализированные решения, например `DeepPass2`, который выполняет многоуровневый анализ с помощью поиска, моделей BERT и LLM.
- Формируется промпт.
- Выполняется запрос к облачной LLM модели.
- Выполняется пост-обработка ответа LLM: проверяется характер ответа, подставляются имена документов.
- Итоговый ответ возвращается пользователю.

## Примеры успешных диалогов

![alt text](images/qa_success_1.png)

![alt text](images/qa_success_2.png)

![alt text](images/qa_success_3.png)

![alt text](images/qa_success_4.png)

## Примеры случаев, когда бот отвечает «Я не знаю»

![alt text](images/qa_notfound_1.png)

![alt text](images/qa_notfound_2.png)