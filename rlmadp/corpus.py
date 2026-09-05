"""Builds a synthetic Rajputana-history haystack with planted multi-hop facts.

The point of a synthetic corpus is CONTROL: we know exactly where each fact
sits, so `coverage` and "did it find the bridge" are checkable rather than
vibes. Three facts are planted far apart, and hop N's search term only becomes
knowable once hop N-1 has been read.

The filler is generated chronicle-ish prose. It is deliberately full of Rajput
proper nouns so that naive greps hit hundreds of irrelevant places -- a haystack
made of needles is the honest test, not a haystack made of hay.
"""

from __future__ import annotations

import random

# --- The three planted facts -------------------------------------------------
#
# Q: "After Haldighati, a minister restored Maharana Pratap's finances with a
#     personal donation. Name the capital Pratap founded with those funds, and
#     the year he died there."
#
#   hop 1  the minister            -> "Bhama Shah"
#   hop 2  the capital he funded   -> "Chavand"
#   hop 3  the year Pratap died    -> "1597"
#
# Note the trap in hop 1: the corpus spells it "Bhama Shah" (two words), so a
# root that greps the modern one-word "Bhamashah" gets ZERO hits and must
# recover with a looser pattern. This is not artificial -- 19th-century
# transliterations of Rajput names vary constantly.

HOP1 = (
    "In the season following the field of Haldighati, when the Rana's treasury "
    "stood exhausted and his captains had begun to disperse, the minister Bhama "
    "Shah came to him at Chulia and laid his whole accumulated fortune before "
    "him -- a sum reported by the bards as sufficient to maintain five and twenty "
    "thousand men for the space of twelve years. The Rana at first refused it. "
    "Being pressed a second time he accepted, and with it re-raised the army of "
    "Mewar and began the recovery of the western districts."
)

HOP2 = (
    "The resources thus restored to him by the gift of Bhama Shah, Pratap Singh "
    "passed the following years "
    "in the reduction of the hill country, retaking Kumbhalmer, Gogunda and "
    "Mandalgarh in succession. His seat during this period was fixed at Chavand, "
    "some forty miles to the south-east of the lake of Udaipur, and Chavand "
    "remained the capital of Mewar for the remainder of his reign and the whole "
    "of his son's minority."
)

HOP3 = (
    "At Chavand, in the fifty-seventh year of his age, the Rana was seized with "
    "an internal injury sustained while drawing a stiff bow at the chase. He "
    "lingered some weeks, and died on the nineteenth of January 1597, having "
    "first bound his chiefs by oath never to suffer the Mughal to hold the hills "
    "of Mewar. Amar Singh succeeded him."
)

# --- Filler generation -------------------------------------------------------

_HOUSES = ["Sisodia", "Rathore", "Kachwaha", "Chauhan", "Hada", "Bhati", "Jhala", "Tomar", "Parmar"]
_PLACES = ["Chittor", "Kumbhalmer", "Gogunda", "Mandalgarh", "Amber", "Marwar", "Ranthambhor",
           "Jodhpur", "Bikaner", "Bundi", "Mewar", "Merta", "Sirohi", "Jaisalmer", "Dungarpur"]
_TITLES = ["Rana", "Rao", "Raja", "Thakur", "Maharana", "Rawat", "Kunwar"]
_NAMES = ["Sanga", "Udai Singh", "Man Singh", "Jaimal", "Patta", "Amar Singh", "Kumbha",
          "Rai Singh", "Surtan", "Duda", "Bida", "Kalyan", "Askaran", "Chandrasen"]
_VERBS = ["gave battle to", "made submission to", "held the passes against", "sent tribute to",
          "raised the standard against", "contracted alliance with", "laid siege to",
          "withdrew before", "exchanged hostages with"]
_TAILS = ["in the year of the Samvat then current", "as the annals of the house record",
          "though the bards differ upon the number", "before the rains had broken",
          "and the fact is confirmed by the inscription at the gate",
          "whereupon the clans were again dispersed"]


def _sentence(rng: random.Random) -> str:
    return (
        f"{rng.choice(_TITLES)} {rng.choice(_NAMES)} of the {rng.choice(_HOUSES)} house "
        f"{rng.choice(_VERBS)} the chief of {rng.choice(_PLACES)}, {rng.choice(_TAILS)}."
    )


def build(target_chars: int = 3_000_000, seed: int = 7) -> tuple[str, dict[str, int]]:
    """Return (document, planted_offsets).

    The three facts are placed at roughly 15%, 45% and 85% of the way through,
    so no single contiguous read can pick up more than one of them. That is what
    forces genuine hopping rather than one lucky large slice.
    """
    rng = random.Random(seed)
    parts: list[str] = ["ANNALS AND ANTIQUITIES OF RAJASTHAN -- A COMPILED CHRONICLE\n\n"]
    plant_at = {0.15: HOP1, 0.45: HOP2, 0.85: HOP3}
    offsets: dict[str, int] = {}
    size = len(parts[0])
    remaining = sorted(plant_at.items())

    while size < target_chars:
        if remaining and size >= remaining[0][0] * target_chars:
            _, fact = remaining.pop(0)
            offsets[fact.split()[0] + "..."] = size
            parts.append("\n\n" + fact + "\n\n")
            size += len(parts[-1])
            continue
        chunk = " ".join(_sentence(rng) for _ in range(40)) + "\n\n"
        parts.append(chunk)
        size += len(chunk)

    doc = "".join(parts)
    # Re-locate exactly, since the running total above counts pre-join.
    offsets = {
        "hop1 (Bhama Shah)": doc.find("Bhama Shah"),
        "hop2 (Chavand)": doc.find("His seat during this period"),
        "hop3 (1597)": doc.find("nineteenth of January 1597"),
    }
    return doc, offsets


TASK = (
    "After the battle of Haldighati, a minister restored Maharana Pratap's "
    "finances with a personal donation. Name the capital Pratap founded with "
    "those funds, and the year he died there."
)

if __name__ == "__main__":
    doc, offsets = build()
    print(f"document: {len(doc):,} chars")
    for k, v in offsets.items():
        print(f"  {k:22s} at char {v:,}  ({v / len(doc):.1%} through)")
