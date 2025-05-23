# TODO:
# * Bring all datasets to the unified interface i.e./
# * Protocol or ABC HfTokenDataset that
#   * loads from hub (hence the "source" argument and such
#   * has tokenizer field/property
#   * yields tokens
#   * is Dataset (naturally)
#   * (?) has /build or load method that yileds dataloader based on config?


from ppotune.data.sets.gsm8k import (
    gsm8k_dataset,
    chat_gsm8k_dataset,
    plain_gsm8k_dataset,
    eval_gsm8k_dataset
)
from ppotune.data.sets.alpaca import (
    alpaca_dataset,
)
from ppotune.data.sets.tldr import (
    tldr_dataset
)
from ppotune.data.sets.helpsteer import (
    helpsteer_dataset
)
from ppotune.data.sets.wiki_lingua import (
    wiki_lingua_dataset
)

__all__ = [
    "gsm8k_dataset",
    "chat_gsm8k_dataset",
    "plain_gsm8k_dataset",
    "alpaca_dataset",
    "tldr_dataset",
    "eval_gsm8k_dataset",
    "helpsteer_dataset",
    "wiki_lingua_dataset",
]
