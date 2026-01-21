from itertools import islice

from src.probing.lingualens import DEFAULT_FEATURES, LinguaLensDataset

dataset = LinguaLensDataset(features=DEFAULT_FEATURES, max_samples=200)
by_feature = {}
for item in dataset:
    by_feature.setdefault(item["feature"], []).append(item)

for feature in DEFAULT_FEATURES:
    print(f"\nFEATURE: {feature}")
    for item in islice(by_feature.get(feature, []), 3):
        print("clean:", item["sentence"])
        print("patch:", item["patch_sentence"])
        print("clean_words:", item["clean_words"])
        print("patch_words:", item["patch_words"])
        print("clean_span_tokens:", [item["clean_words"][i] for i in item["clean_word_indices"]])
        print("patch_span_tokens:", [item["patch_words"][i] for i in item["patch_word_indices"]])
        print("clean_span_words:", item["clean_word_indices"])
        print("patch_span_words:", item["patch_word_indices"])
        print("---")
