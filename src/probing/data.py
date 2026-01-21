import logging
from functools import partial

import numpy as np
import pyconll
from sklearn.utils import resample
from torch.utils.data import Dataset

from src.probing.utils import NOUN_POS, VERB_POS, NUMBER_SUBTYPES, concept_filter

try:
    from lemminflect import getInflection
except ImportError:  # pragma: no cover - optional dependency
    getInflection = None

_VOWELS = set("aeiou")
_IRREGULAR_VERBS = {
    ("be", "Pres", "Sing"): "is",
    ("be", "Pres", "Plur"): "are",
    ("be", "Past", "Sing"): "was",
    ("be", "Past", "Plur"): "were",
    ("have", "Pres", "Sing"): "has",
    ("have", "Pres", "Plur"): "have",
    ("have", "Past", None): "had",
    ("do", "Pres", "Sing"): "does",
    ("do", "Pres", "Plur"): "do",
    ("do", "Past", None): "did",
}


def _normalize_feat_values(values):
    if not values:
        return set()
    if isinstance(values, str):
        return {values}
    return set(values)


def _is_real_token(token):
    token_id = str(token.id)
    return token_id.isdigit()


def _space_after(token):
    misc = getattr(token, "misc", None)
    if not misc:
        return True
    try:
        return misc.get("SpaceAfter") != "No"
    except Exception:
        return "SpaceAfter=No" not in str(misc)


def _preserve_case(source, target):
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def _simple_plural(form):
    lower = form.lower()
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return form + "es"
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in _VOWELS:
        return form[:-1] + "ies"
    return form + "s"


def _simple_singular(form):
    lower = form.lower()
    if lower.endswith("ies") and len(lower) > 3:
        return form[:-3] + "y"
    if lower.endswith(("ses", "xes", "zes", "ches", "shes")):
        return form[:-2]
    if lower.endswith("s") and not lower.endswith("ss"):
        return form[:-1]
    return form


def _inflect_noun(form, lemma, number, is_proper):
    lemma = lemma if lemma and lemma != "_" else form
    if getInflection is not None:
        if number == "Plur":
            tag = "NNPS" if is_proper else "NNS"
        else:
            tag = "NNP" if is_proper else "NN"
        inflected = getInflection(lemma, tag)
        if inflected:
            return _preserve_case(form, inflected[0])
    if number == "Plur":
        return _preserve_case(form, _simple_plural(form))
    return _preserve_case(form, _simple_singular(form))


def _verb_tag(tense, number):
    if tense == "Past":
        return "VBD"
    if tense == "Pres":
        if number == "Sing":
            return "VBZ"
        return "VBP"
    if number == "Sing":
        return "VBZ"
    if number == "Plur":
        return "VBP"
    return None


def _simple_past(form):
    lower = form.lower()
    if lower.endswith("e"):
        return form + "d"
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in _VOWELS:
        return form[:-1] + "ied"
    return form + "ed"


def _inflect_verb(form, lemma, tense=None, number=None):
    lemma = lemma if lemma and lemma != "_" else form
    lemma_lower = lemma.lower()
    if tense:
        key = (lemma_lower, tense, number)
        if key in _IRREGULAR_VERBS:
            return _preserve_case(form, _IRREGULAR_VERBS[key])
        fallback_key = (lemma_lower, tense, None)
        if fallback_key in _IRREGULAR_VERBS:
            return _preserve_case(form, _IRREGULAR_VERBS[fallback_key])
    if getInflection is not None:
        tag = _verb_tag(tense, number)
        if tag:
            inflected = getInflection(lemma, tag)
            if inflected:
                return _preserve_case(form, inflected[0])
    if tense == "Past":
        return _preserve_case(form, _simple_past(lemma))
    if number == "Sing":
        return _preserve_case(form, _simple_plural(lemma))
    return _preserve_case(form, lemma)


def _render_sentence(sentence, overrides=None):
    overrides = overrides or {}
    parts = []
    for token in sentence:
        if not _is_real_token(token):
            continue
        token_id = str(token.id)
        form = overrides.get(token_id, token.form)
        parts.append(form)
        if _space_after(token):
            parts.append(" ")
    return "".join(parts).strip()


def _number_target_and_focus(concept_value):
    if concept_value in NUMBER_SUBTYPES:
        spec = NUMBER_SUBTYPES[concept_value]
        return spec["number"], spec["pos_tags"]
    return concept_value, None


def build_minimal_pair(sentence, concept_key, concept_value):
    if concept_key == "Number":
        target_number, _ = _number_target_and_focus(concept_value)
        if target_number not in {"Sing", "Plur"}:
            return None
        patch_number = "Plur" if target_number == "Sing" else "Sing"
        overrides = {}
        changed = False
        for token in sentence:
            if not _is_real_token(token):
                continue
            values = _normalize_feat_values(token.feats.get("Number"))
            if not values or "Ptan" in values:
                continue
            if token.upos in NOUN_POS:
                new_form = _inflect_noun(token.form, token.lemma, patch_number, token.upos == "PROPN")
            elif token.upos in VERB_POS:
                tense_values = _normalize_feat_values(token.feats.get("Tense"))
                tense = None
                if "Past" in tense_values:
                    tense = "Past"
                elif "Pres" in tense_values:
                    tense = "Pres"
                new_form = _inflect_verb(token.form, token.lemma, tense=tense, number=patch_number)
            else:
                continue
            if new_form and new_form != token.form:
                overrides[str(token.id)] = new_form
                changed = True
        if not changed:
            return None
        clean_text = _render_sentence(sentence)
        patch_text = _render_sentence(sentence, overrides=overrides)
        if clean_text == patch_text:
            return None
        return clean_text, patch_text
    if concept_key == "Tense":
        if concept_value not in {"Past", "Pres"}:
            return None
        patch_tense = "Pres" if concept_value == "Past" else "Past"
        overrides = {}
        changed = False
        for token in sentence:
            if not _is_real_token(token):
                continue
            if token.upos not in VERB_POS:
                continue
            values = _normalize_feat_values(token.feats.get("Tense"))
            if not values:
                continue
            number_values = _normalize_feat_values(token.feats.get("Number"))
            number = None
            if "Sing" in number_values:
                number = "Sing"
            elif "Plur" in number_values:
                number = "Plur"
            new_form = _inflect_verb(token.form, token.lemma, tense=patch_tense, number=number)
            if new_form and new_form != token.form:
                overrides[str(token.id)] = new_form
                changed = True
        if not changed:
            return None
        clean_text = _render_sentence(sentence)
        patch_text = _render_sentence(sentence, overrides=overrides)
        if clean_text == patch_text:
            return None
        return clean_text, patch_text
    return None


def balance_dataset(dataset, seed):
    """
    Balance the dataset by undersampling the majority class.
    """
    labels = np.array([item['label'] for item in dataset])
    positive_samples = [item for item in dataset if item['label'] == 1]
    negative_samples = [item for item in dataset if item['label'] == 0]
    
    logging.info(f"Before balancing:")
    logging.info(f"  Total samples: {len(dataset)}")
    logging.info(f"  Positive samples: {len(positive_samples)}")
    logging.info(f"  Negative samples: {len(negative_samples)}")
    
    if not positive_samples or not negative_samples:
        logging.warning("No positive samples found.")
        return None
    
    if len(positive_samples) < len(negative_samples):
        negative_samples = resample(negative_samples, n_samples=len(positive_samples), random_state=seed)
    else:
        positive_samples = resample(positive_samples, n_samples=len(negative_samples), random_state=seed)
    
    balanced_dataset = positive_samples + negative_samples
    np.random.shuffle(balanced_dataset)
    
    logging.info(f"After balancing:")
    logging.info(f"  Total samples: {len(balanced_dataset)}")
    logging.info(f"  Positive samples: {len(positive_samples)}")
    logging.info(f"  Negative samples: {len(negative_samples)}")
    
    return balanced_dataset

class ProbingDataset(Dataset):
    def __init__(self, conll_file, filter_criterion):
        self.sentences = []
        self.labels = []
        self.filter_criterion = filter_criterion
        self.load_data(conll_file)

    def load_data(self, conll_file):
        conll_files = conll_file if isinstance(conll_file, (list, tuple)) else [conll_file]
        for path in conll_files:
            data = pyconll.load_from_file(path)
            for sentence in data:
                label = self.filter_criterion(sentence)
                if label is None:
                    continue
                label = 1 if label else 0
                self.sentences.append(sentence.text)
                self.labels.append(label)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {"sentence": self.sentences[idx], "label": self.labels[idx]}


class MinimalPairDataset(Dataset):
    def __init__(
        self,
        conll_file,
        concept_key,
        concept_value,
        pos_tags=None,
        exclude_values=None,
        exclude_other_values=False,
        drop_conflicts=False,
    ):
        self.sentences = []
        self.patch_sentences = []
        self.labels = []
        self.concept_key = concept_key
        self.concept_value = concept_value
        self.filter_criterion = partial(
            concept_filter,
            concept_key=concept_key,
            concept_value=concept_value,
            pos_tags=pos_tags,
            exclude_values=exclude_values,
            exclude_other_values=exclude_other_values,
            drop_conflicts=drop_conflicts,
        )
        self.load_data(conll_file)

    def load_data(self, conll_file):
        conll_files = conll_file if isinstance(conll_file, (list, tuple)) else [conll_file]
        for path in conll_files:
            data = pyconll.load_from_file(path)
            for sentence in data:
                label = self.filter_criterion(sentence)
                if label is None:
                    continue
                pair = build_minimal_pair(sentence, self.concept_key, self.concept_value)
                if pair is None:
                    continue
                clean_text, patch_text = pair
                self.sentences.append(clean_text)
                self.patch_sentences.append(patch_text)
                self.labels.append(1 if label else 0)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "sentence": self.sentences[idx],
            "patch_sentence": self.patch_sentences[idx],
            "label": self.labels[idx],
        }
