# -*- coding: utf-8 -*-
"""0.1.63: the in-training card keeps a release card (v1.2's launch had replaced the v1.1 release card whole, 2026-10-19): the training block
is inserted before '## Use', replaced in place when present, and the rest of the card stays byte for byte; without a release card the
in-training card stands alone as before."""
from geolip.nano_beatrix import nano_v1

V12 = ["nano-pollux-dual-v1.2"]
RELEASE = """---
license: cc-by-nc-sa-4.0
---

# AlephLLM nano: Castor and Pollux, v1.1

Four small byte-level AlephLLM models.

| model | the body | parameters |
|---|---|---|
| [`nano-castor-v1.1`](./nano-castor-v1.1) | Castor | 9,215,057 |

## Use

```python
print("load")
```

## Training

The recipe.
"""


def test_the_training_block_joins_a_release_card_and_leaves_the_rest():
    card = nano_v1.card_in_training(V12, existing=RELEASE)
    assert card.count(nano_v1.TRAINING_OPEN) == 1 and card.count(nano_v1.TRAINING_CLOSE) == 1
    a, u = card.index("## Training now: v1.2"), card.index("## Use")
    assert a < u and "nano-pollux-dual-v1.2" in card[a:u] and "nano-castor-v1.1" in card[:a]
    head, tail = card.split(nano_v1.TRAINING_OPEN)[0], card.split(nano_v1.TRAINING_CLOSE)[1].lstrip("\n")
    assert RELEASE == head + tail                          # everything but the block is the release card, untouched
    again = nano_v1.card_in_training(V12, existing=card)   # a second launch replaces the block in place
    assert again == card
    other = nano_v1.card_in_training(["nano-pollux-dual-v1.1"], existing=card)
    assert other.count(nano_v1.TRAINING_OPEN) == 1 and "## Training now: v1.1" in other and "nano-castor-v1.1" in other


def test_without_a_release_card_the_in_training_card_stands_alone():
    fresh = nano_v1.card_in_training(V12)
    assert fresh.startswith("---\nlicense") and "# AlephLLM nano: Castor and Pollux, v1.2" in fresh and nano_v1.TRAINING_OPEN not in fresh
    assert nano_v1.card_in_training(V12, existing="") == fresh and nano_v1.card_in_training(V12, existing=fresh) == fresh
    assert nano_v1.card_in_training() == nano_v1._card_fresh()
