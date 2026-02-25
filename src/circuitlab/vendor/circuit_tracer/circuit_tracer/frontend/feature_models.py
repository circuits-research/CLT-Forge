from pydantic import BaseModel


class Example(BaseModel):
    tokens_acts_list: list[float]
    train_token_ind: int
    is_repeated_datapoint: bool
    tokens: list[str]


class ExamplesQuantile(BaseModel):
    quantile_name: str
    examples: list[Example]


class Model(BaseModel):
    transcoder_id: str
    index: int
    examples_quantiles: list[ExamplesQuantile]
    top_logits: list[str]
    bottom_logits: list[str]
    act_min: float
    act_max: float
    quantile_values: list[float]
    histogram: list[float]
    activation_frequency: float
