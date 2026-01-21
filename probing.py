import argparse
import glob
import logging
import os
from functools import partial

import joblib
import torch
from nnsight import LanguageModel
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import HF_TOKEN
from src.probing.utils import extract_activations, concept_filter, get_features_and_values, concept_label_stats
from src.probing.data import ProbingDataset, balance_dataset

# Constants
TRACER_KWARGS = {'scan': False, 'validate': False}
LOG_DIR = 'logs'
UD_BASE_FOLDER = "./data/universal_dependencies/"

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, 'probing.txt'), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def setup_model(model_name):
    """Initialize and return the language model."""
    return LanguageModel(model_name, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN)

def find_ud_file(ud_base_folder, pattern, override=None):
    if override:
        return override
    matches = glob.glob(os.path.join(ud_base_folder, "**", pattern), recursive=True)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple matches for {pattern} under {ud_base_folder}. "
            "Specify the exact file with --ud_train_file/--ud_test_file."
        )
    return matches[0]

def prepare_datasets(
    train_filepath,
    test_filepath,
    concept_key,
    concept_value,
    seed,
    pos_tags=None,
    exclude_values=None,
    exclude_other_values=False,
    drop_conflicts=False,
):
    """Prepare and balance the training and test datasets."""
    filter_criterion = partial(
        concept_filter,
        concept_key=concept_key,
        concept_value=concept_value,
        pos_tags=pos_tags,
        exclude_values=exclude_values,
        exclude_other_values=exclude_other_values,
        drop_conflicts=drop_conflicts,
    )
    train_dataset = ProbingDataset(train_filepath, filter_criterion)
    test_dataset = ProbingDataset(test_filepath, filter_criterion)

    train_stats = concept_label_stats(
        train_filepath,
        concept_key,
        concept_value,
        pos_tags=pos_tags,
        exclude_values=exclude_values,
        exclude_other_values=exclude_other_values,
        drop_conflicts=drop_conflicts,
    )
    test_stats = concept_label_stats(
        test_filepath,
        concept_key,
        concept_value,
        pos_tags=pos_tags,
        exclude_values=exclude_values,
        exclude_other_values=exclude_other_values,
        drop_conflicts=drop_conflicts,
    )
    if train_stats["total"]:
        print(
            "Train label breakdown:"
            f" only_target={train_stats['only_target']} ({train_stats['only_target'] / train_stats['total']:.2%})"
            f" both={train_stats['both']} ({train_stats['both'] / train_stats['total']:.2%})"
            f" only_excluded={train_stats['only_excluded']} ({train_stats['only_excluded'] / train_stats['total']:.2%})"
            f" neither={train_stats['neither']} ({train_stats['neither'] / train_stats['total']:.2%})"
        )
        if drop_conflicts:
            print(f"Train dropped conflicts: {train_stats.get('dropped', 0)}")
    if test_stats["total"]:
        print(
            "Test label breakdown:"
            f" only_target={test_stats['only_target']} ({test_stats['only_target'] / test_stats['total']:.2%})"
            f" both={test_stats['both']} ({test_stats['both'] / test_stats['total']:.2%})"
            f" only_excluded={test_stats['only_excluded']} ({test_stats['only_excluded'] / test_stats['total']:.2%})"
            f" neither={test_stats['neither']} ({test_stats['neither'] / test_stats['total']:.2%})"
        )
        if drop_conflicts:
            print(f"Test dropped conflicts: {test_stats.get('dropped', 0)}")

    train_pos = sum(train_dataset.labels)
    test_pos = sum(test_dataset.labels)
    train_total = len(train_dataset)
    test_total = len(test_dataset)
    train_ratio = train_pos / train_total if train_total else 0
    test_ratio = test_pos / test_total if test_total else 0
    print(f"Train positives: {train_pos}/{train_total} ({train_ratio:.2%})")
    print(f"Test positives: {test_pos}/{test_total} ({test_ratio:.2%})")

    print("Balancing training dataset...")
    train_dataset = balance_dataset(train_dataset, seed)
    print("Balancing test dataset...")
    test_dataset = balance_dataset(test_dataset, seed)

    return train_dataset, test_dataset

def train_and_evaluate_probe(train_activations, train_labels, test_activations, test_labels, seed):
    """Train a logistic regression probe and evaluate its performance."""
    print("Training logistic regression model...")
    classifier = LogisticRegression(random_state=seed, max_iter=1000, class_weight="balanced", solver="newton-cholesky")
    classifier.fit(train_activations, train_labels)

    train_accuracy = classifier.score(train_activations, train_labels)
    test_accuracy = classifier.score(test_activations, test_labels)

    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

    return classifier

def process_dataset(args, train_filepath, test_filepath):
    """Process a single dataset for probing."""
    print("\nProcessing dataset")
    logging.info("Processing dataset")

    model = setup_model(args.model_name)

    if not train_filepath or not test_filepath:
        print("Skipping dataset: Missing train or test file")
        logging.warning("Skipping dataset: Missing train or test file")
        return

    features = get_features_and_values(train_filepath)
    if args.concept_keys:
        keys = {k.strip() for k in args.concept_keys.split(",") if k.strip()}
        features = {k: v for k, v in features.items() if k in keys}

    output_dir = f"outputs/probing/probes/{'llama' if 'llama' in args.model_name else 'aya'}"
    for concept_key, values in features.items():
        for concept_value in values:
            print(f"\nProcessing {concept_key}: {concept_value}")
            logging.info(f"Processing {concept_key}: {concept_value}")

            model_filename = f"{concept_key}_{concept_value}.joblib"
            model_path = os.path.join(output_dir, model_filename)
            
            if os.path.exists(model_path):
                print(f"Probe already exists. Skipping.")
                logging.info(f"Probe already exists for {concept_key}: {concept_value}. Skipping.")
                continue

            train_dataset, test_dataset = prepare_datasets(
                train_filepath,
                test_filepath,
                concept_key,
                concept_value,
                args.seed,
                pos_tags=args.pos_tags,
                exclude_values=args.exclude_values,
                exclude_other_values=args.exclude_other_values,
                drop_conflicts=args.drop_conflicts,
            )

            if train_dataset is None or len(train_dataset) < 128 or test_dataset is None:
                print(f"Not enough samples. Skipping.")
                logging.warning(f"Skipping {concept_key}: {concept_value}: Not enough samples")
                continue

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            print("Extracting activations...")
            train_activations, train_labels = extract_activations(model, train_dataloader, args.layer_num)
            test_activations, test_labels = extract_activations(model, test_dataloader, args.layer_num)

            classifier = train_and_evaluate_probe(train_activations, train_labels, test_activations, test_labels, args.seed)

            os.makedirs(output_dir, exist_ok=True)
            joblib.dump(classifier, model_path)
            print(f"Saved trained model to {model_path}")
            logging.info(f"Saved trained model to {model_path}")

def main(args):
    if args.pos_tags:
        args.pos_tags = [tag.strip() for tag in args.pos_tags.split(",") if tag.strip()]
    if args.exclude_values:
        args.exclude_values = [val.strip() for val in args.exclude_values.split(",") if val.strip()]
    if args.exclude_other_values:
        args.exclude_other_values = True
    if args.drop_conflicts:
        args.drop_conflicts = True

    try:
        train_filepath = find_ud_file(args.ud_base_folder, "*-ud-train.conllu", args.ud_train_file)
        test_filepath = find_ud_file(args.ud_base_folder, "*-ud-test.conllu", args.ud_test_file)
    except ValueError as exc:
        print(f"Error: {exc}")
        logging.error(str(exc))
        return

    if not train_filepath or not test_filepath:
        print("Error: Missing UD train/test file. Provide --ud_train_file/--ud_test_file or a base folder.")
        logging.error("Missing UD train/test file.")
        return

    process_dataset(args, train_filepath, test_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probing script for language models")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the language model to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for data loading")
    parser.add_argument("--layer_num", type=int, default=16, help="Layer number to extract activations from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--concept_keys", type=str, default=None, help="Comma-separated concept keys to probe (e.g., Number,Tense)")
    parser.add_argument("--pos_tags", type=str, default=None, help="Comma-separated UPOS tags to filter by (e.g., VERB,AUX)")
    parser.add_argument("--exclude_values", type=str, default=None, help="Comma-separated concept values to exclude (e.g., Plur)")
    parser.add_argument("--exclude_other_values", action="store_true", help="Exclude sentences that contain any other values of the concept")
    parser.add_argument("--drop_conflicts", action="store_true", help="Drop sentences that contain both target and excluded values")
    parser.add_argument("--ud_base_folder", type=str, default=UD_BASE_FOLDER, help="Base folder for UD treebanks")
    parser.add_argument("--ud_train_file", type=str, default=None, help="Override UD training .conllu path")
    parser.add_argument("--ud_test_file", type=str, default=None, help="Override UD test .conllu path")
    args = parser.parse_args()

    main(args)
