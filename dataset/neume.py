import random
from typing import List

from common import Distrubution


FONTS = [
    'N',    # Neanes
    'NSS'   # NeanesStathisSeries
]


def load_classes() -> List[str]:
    with open('byztex/standalone_neumes.txt', 'r') as f:
        classes = f.readlines()

    classes = [c.rstrip() for c in classes]

    return classes

class NeumeGenerator:
    classes = load_classes()

    def __init__(
        self,
        distribution: dict = None,
        font_distribution: dict = None,
        seed: int = None
    ):
        random.seed(seed)
        self.dist = NeumeGenerator.init_cumulative_distribution(
            NeumeGenerator.classes, distribution
        )
        self.font_dist = NeumeGenerator.init_cumulative_distribution(FONTS, font_distribution)

    @staticmethod
    def init_cumulative_distribution(values: list, distribution: dict) -> dict:
        distribution = dict() if distribution is None else distribution.copy()
        if 'denominator' in distribution:
            denom = distribution.pop('denominator')
        else:
            denom = len(values)
        dist = Distrubution(denom, distribution)
        undef_classes = [c for c in values if c not in distribution]
        dist.make_uniform(undef_classes)
        cum_dist = dist.make_cumulative()
        return cum_dist

    def next(self) -> (str, str):
        font_rnd_val = random.random() * len(self.font_dist)
        font = None
        for key in FONTS:
            if font_rnd_val < self.font_dist[key]:
                font = key
                break
        if font is None:
            font = FONTS[-1]

        neume_rnd_val = random.random() * len(self.dist)
        for key in NeumeGenerator.classes:
            if neume_rnd_val < self.dist[key]:
                return font, key

        return font, self.classes[-1]
