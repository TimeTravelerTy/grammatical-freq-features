import re
from difflib import SequenceMatcher

from datasets import load_dataset
from torch.utils.data import Dataset


_WORD_RE = re.compile(r"[A-Za-z0-9]+|[^A-Za-z0-9\\s]")
DEFAULT_FEATURES = (
    "noun_plural",
    "third_person_singular",
    "past_tense",
    "past_tense_irregular",
)


def _tokenize_words(text):
    return [tok for tok in _WORD_RE.findall(text.strip()) if not tok.isspace()]


def _diff_word_indices(clean_words, patch_words):
    matcher = SequenceMatcher(a=clean_words, b=patch_words)
    clean_indices = set()
    patch_indices = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        clean_indices.update(range(i1, i2))
        patch_indices.update(range(j1, j2))
    return sorted(clean_indices), sorted(patch_indices)


def load_lingualens_pairs(
    split="train",
    language="English",
    categories=("syntax", "morphology"),
    features=None,
    max_samples=None,
    seed=42,
):
    dataset = load_dataset("THU-KEG/LinguaLens-Data")[split]

    if language:
        dataset = dataset.filter(lambda x: x["language"] == language)
    if categories:
        allowed = set(categories)
        dataset = dataset.filter(lambda x: any(c in allowed for c in x["categories"]))
    if features:
        allowed_features = set(features)
        dataset = dataset.filter(lambda x: x["feature"] in allowed_features)

    if max_samples:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))

    pairs = []
    for row in dataset:
        clean = row["sentence1"]
        patch = row["sentence2"]
        clean_words = _tokenize_words(clean)
        patch_words = _tokenize_words(patch)
        clean_indices, patch_indices = _diff_word_indices(clean_words, patch_words)
        if not clean_indices or not patch_indices:
            continue
        pairs.append(
            {
                "sentence": clean,
                "patch_sentence": patch,
                "clean_words": clean_words,
                "patch_words": patch_words,
                "clean_word_indices": clean_indices,
                "patch_word_indices": patch_indices,
                "feature": row["feature"],
                "categories": row["categories"],
                "pair_index": row["pair_index"],
            }
        )
    return pairs


class LinguaLensDataset(Dataset):
    def __init__(
        self,
        pairs=None,
        split="train",
        language="English",
        categories=("syntax", "morphology"),
        features=None,
        max_samples=None,
        seed=42,
    ):
        if pairs is None:
            pairs = load_lingualens_pairs(
                split=split,
                language=language,
                categories=categories,
                features=features,
                max_samples=max_samples,
                seed=seed,
            )
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
