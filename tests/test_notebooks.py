"""Every notebook in colab/ ships with its Open-in-Colab link to its own file on GitHub and pins a released version of this package
no newer than the installed one; its version assert never demands more than the pin installs."""
import glob
import json
import os
import re

import geolip.nano_beatrix as N

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "colab")
GH = "https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/"


def _notebooks():
    paths = sorted(glob.glob(os.path.join(ROOT, "*.ipynb")))
    assert paths, "no notebooks under colab/"
    return [(p, json.load(open(p, encoding="utf-8"))) for p in paths]


def test_head_carries_the_colab_link_to_its_own_file():
    for p, nb in _notebooks():
        head = nb["cells"][0]
        assert head["cell_type"] == "markdown", p
        assert GH + os.path.basename(p) in "".join(head["source"]), p


def test_c1_pins_a_released_version_and_asserts_no_more_than_it():
    current = tuple(int(x) for x in N.__version__.split("."))
    for p, nb in _notebooks():
        c1 = "".join(nb["cells"][1]["source"])
        pin = re.search(r"geolip-nano-beatrix-trainer@v(\d+)\.(\d+)\.(\d+)", c1)
        assert pin, p
        pinned = tuple(int(x) for x in pin.groups())
        assert pinned <= current, (p, pinned, current)
        floor = re.search(r">= \((\d+), (\d+), (\d+)\)", c1)
        assert floor and tuple(int(x) for x in floor.groups()) <= pinned, p


def test_code_cells_are_clean_of_outputs():
    for p, nb in _notebooks():
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                assert c.get("outputs") == [] and c.get("execution_count") is None, p
