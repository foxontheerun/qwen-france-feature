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
