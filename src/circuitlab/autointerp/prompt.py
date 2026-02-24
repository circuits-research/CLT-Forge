SYSTEM_PROMPT = (
    "You are a thoughtful language model researcher working to interpret the role of features "
    "inside a model. Each feature is known to activate selectively in response to specific patterns in text.\n\n"
    "You are given a set of short examples. In each example, the tokens that most strongly trigger the feature are marked "
    "with <<double angle brackets>>.\n\n"
    "Your task is to carefully analyze these examples and provide a description in three words maximum that summarizes the underlying "
    "pattern or concept the feature responds to. You should then provide an explanation in one sentence that explains this description. Sometimes, the role of a feature can only be determined by looking at adjacent tokens.\n\n"
    "CRITICAL: You MUST format your response exactly as follows:\n"
    "[DESCRIPTION]: Your three word description here\n"
    "[EXPLANATION]: Your one sentence explanation here\n\n"
    "Do NOT include any other text, formatting, or commentary. Here are examples of the exact format required:\n"
)
# "Note: Features in early layers (≤ 4) often activate on specific token types or lexical forms.\n"
# "Features in middle layers (5–8) may capture broader semantic or stylistic patterns.\n"
# "Features in later layers (> 8) often relate to predictive dynamics, such as preparing to generate specific kinds of words.\n"
# "These are tendencies, not hard rules."

EARLY_EXAMPLE_1 = """\
Example 1: He plays <<football>> every weekend.
Example 2: The kids are watching <<football>> on TV.
Example 3: Les enfants regardent du <<football>> à la télévision.
Example 4: Er spielt jeden Samstag <<Fußball>> im Park.
Example 5: يلعب <<كرة القدم>> كل يوم أحد.
Example 6: Players were trying to take the <<ball>> from the goal keeper.

[DESCRIPTION]: 'Football'
[EXPLANATION]: This feature activates on tokens related to football. 
"""

EARLY_EXAMPLE_2 = """\
Example 1: Marry had <<loved>> that song.
Example 2: The children had <<to>> go to school. 
Example 3: Elle avait <<chanté>> toute la nuit.
Example 4: Der Lehrer hatte <<erreicht>> eine Schlussfolgerung.
Example 5: 他已经<<到达>>了最后的结论。
Example 6: The teacher had <<reached>> a conclusion about the paper.

[DESCRIPTION]: after 'had' token 
[EXPLANATION]: This feature activates on words that come after "had"
"""

EARLY_EXAMPLE_3 = """\
Example 1: <<Back when>> they were kids and loved music.
Example 2: This was all <<before>> they had to do this. 
Example 3: <<Autrefois>> ils étaient de bons amis.
Example 4: <<Früher>> spielten die Kinder im Garten.
Example 5: <<في الماضي>> كانوا أصدقاء جيدين.
Example 6: They had been lover in a <<previous>> life.

[DESCRIPTION]: in the past
[EXPLANATION]: This feature activates on words or expressions that refer to a time in the past. 
"""

NORMAL_EXAMPLE_1 = """\
Example 1: He plays <<football>> every weekend.
Example 2: The kids are watching <<football>> on TV.
Example 3: Les enfants regardent du <<football>> à la télévision.
Example 4: Er spielt jeden Samstag <<Fußball>> im Park.
Example 5: 他每个周末都踢<<足球>>。
Example 6: Players were trying to take the <<ball>> from the goal keeper.

[DESCRIPTION]: 'Football'
[EXPLANATION]: This feature activates on tokens related to football. 
"""

NORMAL_EXAMPLE_2 = """\
Example 1: She was <<over the moon>> after hearing the news.
Example 2: We were <<walking on air>> all day.
Example 3: Elle était <<aux anges>> après avoir reçu la nouvelle.
Example 4: Er war <<im siebten Himmel>> nach dem Rennen.
Example 5: كان <<في قمة السعادة>> بعد سماع الخبر.
Example 6: He felt <<on top of the world>> after finishing the race.

[DESCRIPTION]: Positive emotional expressions
[EXPLANATION]: Idiomatic expressions conveying positive emotional states.
"""

NORMAL_EXAMPLE_3 = """\
Feature: Layer 11, Neuron 10003

Example 1: Let's play <<some>> football.
Example 2: He <<mentioned>> rugby during the conversation.
Example 3: Ils parlaient <<de>> basketball et tennis.
Example 4: Sie sprachen <<über>> Basketball und Tennis.
Example 5: يتحدثون <<عن>> كرة السلة والتنس.
Example 6: They were talking <<about>> basketball and tennis.

[DESCRIPTION]: 'say a sport'
[EXPLANATION]: This feature activates on tokens that are preceded by a sport token.
"""

def generate_prompt(highlighted_sequences: list[str], feat_layer: int, feat_idx: int) -> str:
    """
    Builds a few-shot prompt to explain what a transformer feature activates on,
    inspired by Delphi github repo.
    """

    examples_formatted = "\n".join(
        [f"Example {i+1}: {s}" for i, s in enumerate(highlighted_sequences)]
    )

    if feat_layer == 0: 
        prompt = (
            f"{SYSTEM_PROMPT.strip()}\n\n"
            f"---\n\n"
            f"{EARLY_EXAMPLE_1.strip()}\n\n"
            f"{EARLY_EXAMPLE_2.strip()}\n\n"
            f"{EARLY_EXAMPLE_3.strip()}\n\n"
            f"---\n\n"
            f"Feature: Layer {feat_layer}, Neuron {feat_idx}\n\n"
            f"{examples_formatted}\n\n"
            "Now provide your analysis using the EXACT format shown above. Include both [DESCRIPTION]: and [EXPLANATION]: lines:"
        )
    else: 
        prompt = (
            f"{SYSTEM_PROMPT.strip()}\n\n"
            f"---\n\n"
            f"{NORMAL_EXAMPLE_1.strip()}\n\n"
            f"{NORMAL_EXAMPLE_2.strip()}\n\n"
            f"{NORMAL_EXAMPLE_3.strip()}\n\n"
            f"---\n\n"
            f"Feature: Layer {feat_layer}, Neuron {feat_idx}\n\n"
            f"{examples_formatted}\n\n"
            "Now provide your analysis using the EXACT format shown above. Include both [DESCRIPTION]: and [EXPLANATION]: lines:"
        )

    return prompt
