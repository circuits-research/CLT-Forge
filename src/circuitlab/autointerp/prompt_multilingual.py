SYSTEM_PROMPT = (
    "You are a thoughtful language model researcher working to interpret the role of features "
    "inside a model. Each feature is known to activate selectively in response to specific patterns in text.\n\n"
    "You are given a set of short examples. In each example, the tokens that most strongly trigger the feature are marked "
    "with <<double angle brackets>>.\n\n"
    "Your task is to carefully analyze these examples and provide a single sentence explanation that describes what "
    "pattern or concept the feature responds to. These examples may be in different languages. You should ignore the "
    "language used, only focus on the content. You should provide your answer in english. Sometimes, the role of a "
    "feature can only be determined by looking at adjacent tokens.\n\n"
    "Provide only one sentence that explains what this feature detects or responds to. Do NOT include any other text, "
    "formatting, or commentary. Here are examples:\n"
)

EXAMPLE_1 = """\
Example 1: He plays <<football>> every weekend.
Example 2: Les enfants regardent du <<football>> à la télévision.
Example 3: Er spielt jeden Samstag <<Fußball>> im Park.
Example 4: يلعب <<كرة القدم>> كل يوم أحد.
Example 5: 他每个周末都踢<<足球>>。
Example 6: Players were trying to take the <<ball>> from the goal keeper.

LLM ANSWER: This feature activates on tokens related to football and soccer sports.
"""

EXAMPLE_2 = """\
Example 1: Mary had <<loved>> that song.
Example 2: The children had <<to>> go to school.
Example 3: Elle avait <<chanté>> toute la nuit.
Example 4: Der Lehrer hatte <<erreicht>> eine Schlussfolgerung.
Example 5: كان قد <<وصل>> إلى نتيجة حول البحث.
Example 6: 他已经<<到达>>了最后的结论。
Example 7: The teacher had <<reached>> a conclusion about the paper.

LLM ANSWER: This feature activates on words that come after the auxiliary verb "had".
"""

EXAMPLE_3 = """\
Example 1: <<Back when>> they were kids and loved music.
Example 2: This was all <<before>> they had to do this.
Example 3: <<Autrefois>> ils étaient de bons amis.
Example 4: <<Früher>> spielten die Kinder im Garten.
Example 5: <<في الماضي>> كانوا أصدقاء جيدين.
Example 6: <<以前>>孩子们在花园里玩耍。
Example 7: They had been lovers in a <<previous>> life.

LLM ANSWER: This feature activates on words or expressions that refer to a time in the past.
"""

EXAMPLE_4 = """\
Example 1: She was <<over the moon>> after hearing the news.
Example 2: We were <<walking on air>> all day.
Example 3: Elle était <<aux anges>> après avoir reçu la nouvelle.
Example 4: Er war <<im siebten Himmel>> nach dem Rennen.
Example 5: كان <<في قمة السعادة>> بعد سماع الخبر.
Example 6: 他听到消息后<<乐得合不拢嘴>>。
Example 7: He felt <<on top of the world>> after finishing the race.

LLM ANSWER: This feature activates on idiomatic expressions conveying positive emotional states.
"""

EXAMPLE_5 = """\
Example 1: Let's play <<some>> football.
Example 2: He <<mentioned>> rugby during the conversation.
Example 3: They were talking <<about>> basketball and tennis.

LLM ANSWER: This feature activates on tokens that appear before sport related tokens.
"""

def generate_prompt_multilingual(highlighted_sequences: list[str], feat_layer: int, feat_idx: int) -> str:
    """
    Builds a few-shot prompt to explain what a transformer feature activates on,
    inspired by Delphi github repo.
    """

    examples_formatted = "\n".join(
        [f"Example {i+1}: {s}" for i, s in enumerate(highlighted_sequences)]
    )

    prompt = (
        f"{SYSTEM_PROMPT.strip()}\n\n"
        f"---\n\n"
        f"{EXAMPLE_1.strip()}\n\n"
        f"{EXAMPLE_2.strip()}\n\n"
        f"{EXAMPLE_3.strip()}\n\n"
        f"{EXAMPLE_4.strip()}\n\n"
        f"{EXAMPLE_5.strip()}\n\n"
        f"---\n\n"
        f"Feature: Layer {feat_layer}, Neuron {feat_idx}\n\n"
        f"{examples_formatted}\n\n"
        "LLM ANSWER: "
    )

    return prompt
