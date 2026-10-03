"""Prompt sets used across the scripts.

Kept in one place so every figure and table draws from the same lists.
"""

# Single-token continuations we score against " Paris" (feature-specificity table).
COMPARISON_TOKENS = [" Paris", " London", " Madrid", " Berlin", " the", " known"]

# Tokens checked in the direct-logit-attribution ranking table.
DLA_CHECK_TOKENS = [
    " Paris", " France", " French", " Pierre", " Eiffel",
    " London", " Berlin", " Madrid", " Spanish", " German",
]

# Heatmap prompts: the concept across languages/scripts, cultural markers, and
# matched negatives (other countries, math, weather). This is the hero figure.
HEAT_PROMPTS = [
    "The capital of France is Paris",
    "Paris is famous for the Eiffel Tower",
    "I love French cuisine and wine",
    "The capital of Germany is Berlin",
    "Berlin is famous for the Brandenburg Gate",
    "I love Italian pizza and pasta",
    "The weather today is sunny and warm",
    "She wrote three numbers on paper",

    "The capital of Italy is Rome",
    "Rome is famous for the Colosseum",
    "I love Spanish tapas and paella",
    "The capital of Japan is Tokyo",
    "Tokyo is famous for sushi and Shibuya",
    "The capital of Russia is Moscow",
    "Moscow is famous for Red Square",
    "I love Greek moussaka and baklava",
    "The weather forecast says it will rain tomorrow",
    "She wrote five equations on the whiteboard",

    "Столица Франции — Париж",
    "Париж славится Эйфелевой башней",
    "Я люблю французские сыры и вино",
    "Столица России — Москва",
    "Москва известна Красной площадью",
    "Берлин — столица Германии",
    "Я обожаю итальянскую пиццу и пасту",
    "Сегодня солнечная и теплая погода",
    "Маша путешествовала по Европе и посетила Лувр",
    "Клод Моне был талантливым живописцем",

    "Die Hauptstadt von Frankreich ist Paris",
    "フランスの首都はパリ",
]

# Short story-style prompts for the additive-steering sweep.
STORY_PROMPTS = [
    "Once upon a time there was a",
    "Let me tell you about my last vacation. We went to",
    "The most beautiful city I have ever visited is",
    "She opened the old book and started reading about",
    "He walked through the streets thinking about his trip to",
    "The coffee cup was half empty when he noticed that",
    "In the middle of the forest, there was a cabin where",
    "The professor looked at the data again and realized that",
]

# Mixed prompts for the closed-loop demo: some pull toward France on their own,
# some do not — the controller should only inject when the sensor reads low.
MIXED_PROMPTS = [
    "Какая столица Франции? Ответь кратко",
    "Расскажи про художника-импрессиониста. Ответь кратко",
    "Какой твой любимый город? Ответь кратко",
    "Расскажи про какого нибудь композитора. Ответь кратко",
    "What is the capital of Italy? Answer in one word",
    "Назови известную реку. Ответь кратко",
    "Назови известный французский десерт. Ответь кратко",
    "Кто написал 'Три мушкетёра'? Ответь кратко",
    "Какой праздник во Франции отмечают 14 июля? Ответь кратко",
    "Назови самую высокую гору в мире. Ответь кратко",
    "Кто изобрёл телефон? Ответь кратко",
    "Какой газ выделяют растения при фотосинтезе? Ответь кратко",
]

# Feature selection for the multi-layer ablation (07, 08): on every layer, the
# France feature must fire on all SELECT_FRANCE prompts and stay below its level
# on every SELECT_CONTROL prompt.
SELECT_FRANCE = [
    "The capital of France is",
    "I spent the summer traveling across France",
    "Die Hauptstadt von Frankreich ist",
    "Столица Франции —",
    "フランスの首都は",
]
SELECT_CONTROL = [
    "The capital of Germany is",
    "I spent the summer traveling across Japan",
    "Die Hauptstadt von Italien ist",
    "Столица России —",
    "スペインの首都は",
    "The capital of Spain is",
]

# Held-out evaluation (08): no prompt here was used to select features.
# (prompt, target) — scored on the target's first token.
HELDOUT_FRANCE = [
    ("La capitale de la France est", " Paris"),
    ("The Louvre museum is located in the city of", " Paris"),
    ("Notre-Dame cathedral stands in the heart of", " Paris"),
    ("The Seine river flows through the city of", " Paris"),
    ("Napoleon Bonaparte was the emperor of", " France"),
    ("Croissants and baguettes are typical food from", " France"),
    ("Лувр находится в городе", " Париж"),
    ("エッフェル塔がある都市は", "パリ"),
]
HELDOUT_CONTROL = [
    ("The capital of Italy is", " Rome"),
    ("La capital de España es", " Madrid"),
    ("The Colosseum is located in the city of", " Rome"),
    ("The Brandenburg Gate is located in the city of", " Berlin"),
    ("Big Ben is located in the city of", " London"),
    ("Столица Японии —", " Токио"),
]

# Generic text with no country in it: mean next-token loss checks that an
# ablation removes a concept rather than damaging the model.
NEUTRAL_TEXT = [
    "The quick brown fox jumps over the lazy dog and runs into the forest.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "To install the package, run pip install and then restart the kernel.",
    "Вчера мы гуляли в парке и долго обсуждали новый фильм.",
    "She opened the window, made a cup of tea and started reading her emails.",
]
