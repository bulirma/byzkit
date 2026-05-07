import random
import re
from typing import List


def load_classes() -> List[str]:
    with open('byztex/template_standalone.tex', 'r') as f:
        text = f.read()

    neume_pattern = re.compile(r'\\providecommand\\(\w+)\{')
    occurrences = neume_pattern.findall(text)
    return occurrences


class NeumeGenerator:
    classes = load_classes()
    num_classes = len(classes)

    def __init__(self, distribution: dict = None, seed: int = None):
        if distribution is None:
            distribution = dict()
        self.denom = distribution.get('denominator', NeumeGenerator.num_classes)
        self.dist = dict()
        num_missing = NeumeGenerator.num_classes - len(distribution) + (1 if 'denominator' in distribution else 0)
        for key in NeumeGenerator.classes:
            if key not in distribution:
                distribution[key] = num_missing / self.denom
        cumVal = 0
        for key in NeumeGenerator.classes:
            cumVal += distribution[key]
            self.dist[key] = cumVal
        random.seed(seed)

    def next(self):
        rnd_val = random.random() * self.denom
        for key in NeumeGenerator.classes:
            if rnd_val < self.dist[key]:
                return key
        return self.classes[-1]
