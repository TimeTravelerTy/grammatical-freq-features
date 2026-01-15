import argparse
import glob
import json
import logging
import os
from functools import partial

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from nnsight import LanguageModel

from src.config import HF_TOKEN
from src.utils import setup_model, setup_autoencoder, dict_to_json
from src.probing.attribution import attribution_patching
from src.probing.data import ProbingDataset, balance_dataset
from src.probing.utils import get_features_and_values, concept_filter, convert_probe_to_pytorch
from src.utils import get_available_languages, get_available_concepts
# Constants
TRACER_KWARGS = {'scan': False, 'validate': False}
LOG_DIR = 'logs'
UD_BASE_FOLDER = "./data/universal_dependencies/"
AYA_AE_PATH = ""

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, 'probe_features.txt'), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


import traceback
import torch
import gc

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()

def setup_model_and_autoencoder(model_name, device_map="cuda", torch_dtype=torch.float16):
    """Set up the language model and autoencoder."""
    model = LanguageModel(model_name, torch_dtype=torch_dtype, device_map=device_map, token=HF_TOKEN)
    submodule = model.model.layers[16]
    if "llama" in model_name:
        autoencoder = setup_autoencoder(device=device_map)
    else:
        autoencoder = setup_autoencoder(checkpoint_path=AYA_AE_PATH, device=device_map)
    print(f"Autoencoder dictionary size: {autoencoder.dict_size}")
    return model, submodule, autoencoder

def load_progress(output_dir, concept_key, concept_value):
    """Load progress from a previous run if it exists."""
    progress_file = os.path.join(output_dir, f"{concept_key}_{concept_value}.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            outputs = json.load(f)
        print(f"Loaded progress from {progress_file}")
        return outputs
    return {}

def process_concept(args, concept_key, concept_value, model, submodule, autoencoder, probe_dir, output_dir, verbose=True):
    """Process a single concept across all languages."""
    if verbose:
        print(f"Processing concept: {concept_key}:{concept_value}")
    
    outputs = load_progress(output_dir, concept_key, concept_value)
    if args.language:
        languages = [args.language]
    else:
        languages = get_available_languages(args.ud_base_folder)

    for language in tqdm(languages, desc=f"Processing languages for {concept_key}:{concept_value}", leave=False, disable=not verbose):
        if language in outputs:
            if verbose:
                print(f"Skipping {language} - already processed.")
            continue

        if args.ud_train_file:
            ud_train_filepath = args.ud_train_file
        else:
            ud_folder = f"{args.ud_base_folder}UD_{language}/"
            ud_train_filepath = glob.glob(os.path.join(ud_folder, "*-ud-train.conllu"))
            if not ud_train_filepath:
                if verbose:
                    print(f"Training file not found for {language}. Skipping.")
                continue
            ud_train_filepath = ud_train_filepath[0]

        # Check if the concept exists for this language
        features = get_features_and_values(ud_train_filepath)
        if concept_key not in features or concept_value not in features[concept_key]:
            if verbose:
                print(f"Concept {concept_key}:{concept_value} not found in the data for {language}. Skipping.")
            continue

        # Load and convert probe to PyTorch
        probe_file = os.path.join(probe_dir, f"{language}_{concept_key}_{concept_value}.joblib")
        if not os.path.exists(probe_file):
            if verbose:
                print(f"Probe file not found for {language}_{concept_key}_{concept_value}. Skipping.")
            continue
        
        probe = joblib.load(probe_file)
        torch_probe = convert_probe_to_pytorch(probe)

        # Prepare dataset
        train_dataset = prepare_dataset(
            ud_train_filepath,
            concept_key,
            concept_value,
            args.seed,
            pos_tags=args.pos_tags,
            exclude_values=args.exclude_values,
            exclude_other_values=args.exclude_other_values,
            drop_conflicts=args.drop_conflicts,
        )
        if train_dataset is None or len(train_dataset) < 128:
            if verbose:
                print(f"Not enough samples in training set for {language}_{concept_key}_{concept_value}. Skipping.")
            continue

        # Perform attribution patching
        effects = attribution_patching_loop(train_dataset, model, torch_probe, submodule, autoencoder)

        # Select top and bottom features
        top_effects, bottom_effects = select_top_bottom_features(effects[submodule])

        outputs[language] = {
            "top_1_percent": top_effects,
            "bottom_1_percent": bottom_effects
        }

        # Save progress after each language
        output_file = os.path.join(output_dir, f"{concept_key}_{concept_value}.json")
        dict_to_json(outputs, output_file)
        if verbose:
            print(f"Saved progress to {output_file}")

    return outputs

def prepare_dataset(
    ud_train_filepath,
    concept_key,
    concept_value,
    seed,
    pos_tags=None,
    exclude_values=None,
    exclude_other_values=False,
    drop_conflicts=False,
):
    """Prepare and balance the dataset."""
    filter_criterion = partial(
        concept_filter,
        concept_key=concept_key,
        concept_value=concept_value,
        pos_tags=pos_tags,
        exclude_values=exclude_values,
        exclude_other_values=exclude_other_values,
        drop_conflicts=drop_conflicts,
    )
    train_dataset = ProbingDataset(ud_train_filepath, filter_criterion)
    return balance_dataset(train_dataset, seed)

def attribution_patching_loop(dataset, model, torch_probe, submodule, autoencoder):
    """Perform attribution patching on the dataset."""
    effects = {}
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, example in enumerate(tqdm(dataloader, desc="Attribution patching")):
        if i >= 128:
            break
        tokens = model.tokenizer(example["sentence"][0], return_tensors="pt", padding=False)
        e, _, _, _ = attribution_patching(tokens, model, torch_probe, [submodule], {submodule: autoencoder})
        if submodule not in effects:
            effects[submodule] = e[submodule].sum(dim=0)
        else:
            effects[submodule] += e[submodule].sum(dim=0)
    return effects

def select_top_bottom_features(effects):
    """Select top 1% and bottom 1% features based on attribution effects."""
    feature_acts = torch.sum(effects.act, dim=0)
    total_features = feature_acts.numel()
    num_features_to_select = max(1, int(0.01 * total_features))
    
    top_v, top_i = torch.topk(feature_acts.flatten(), num_features_to_select)
    bottom_v, bottom_i = torch.topk(feature_acts.flatten(), num_features_to_select, largest=False)
    
    top_effects = list(zip(top_i.flatten().tolist(), top_v.cpu().numpy().tolist()))
    bottom_effects = list(zip(bottom_i.flatten().tolist(), bottom_v.cpu().numpy().tolist()))
    
    return top_effects, bottom_effects

def feature_selection(args):
    """Main function for feature selection across concepts and languages."""
    model, submodule, autoencoder = setup_model_and_autoencoder(
        args.model_name,
        device_map=args.device_map,
        torch_dtype=torch.float16 if args.device_map != "cpu" else torch.float32,
    )
    
    probe_dir = f"outputs/probing/probes/{'llama' if 'llama' in args.model_name else 'aya'}"
    output_dir = f"outputs/probing/features/{'llama' if 'llama' in args.model_name else 'aya'}"
    print(f"Probe directory: {probe_dir}")
    print(f"Output directory: {output_dir}")

    if args.concept_key and args.concept_value:
        concepts = [(args.concept_key, args.concept_value)]
    else:
        concepts = get_available_concepts(probe_dir)

    for concept_key, concept_value in tqdm(concepts, desc="Processing concepts"):
        #process_concept(args, concept_key, concept_value, model, submodule, autoencoder, probe_dir, output_dir)
        try:
            process_concept(args, concept_key, concept_value, model, submodule, autoencoder, probe_dir, output_dir)
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory error for concept {concept_key}_{concept_value}")
            cleanup_memory()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            print("Cleaned up memory and continuing with next concept...")
            continue
        except Exception as e:
            print(f"Error processing concept {concept_key}_{concept_value}: {str(e)}")
            traceback.print_exc()
            print("Continuing with next concept...")
            continue
        finally:
            cleanup_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature selection for probing")
    parser.add_argument('--concept_key', type=str, default=None, help="Concept key for probing")
    parser.add_argument('--concept_value', type=str, default=None, help="Concept value for probing")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the language model")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--language', type=str, default=None, help="Specific language to process (optional)")
    parser.add_argument("--ud_base_folder", type=str, default=UD_BASE_FOLDER, help="Base folder for UD treebanks")
    parser.add_argument("--ud_train_file", type=str, default=None, help="Override UD training .conllu path")
    parser.add_argument("--pos_tags", type=str, default=None, help="Comma-separated UPOS tags to filter by (e.g., VERB,AUX)")
    parser.add_argument("--exclude_values", type=str, default=None, help="Comma-separated concept values to exclude (e.g., Plur)")
    parser.add_argument("--exclude_other_values", action="store_true", help="Exclude sentences that contain any other values of the concept")
    parser.add_argument("--drop_conflicts", action="store_true", help="Drop sentences that contain both target and excluded values")
    parser.add_argument("--device_map", type=str, default="cuda", help="Device map for model/SAE (e.g., cuda, cpu, auto)")
    args = parser.parse_args()

    if args.pos_tags:
        args.pos_tags = [tag.strip() for tag in args.pos_tags.split(",") if tag.strip()]
    if args.exclude_values:
        args.exclude_values = [val.strip() for val in args.exclude_values.split(",") if val.strip()]
    if args.exclude_other_values:
        args.exclude_other_values = True
    if args.drop_conflicts:
        args.drop_conflicts = True

    feature_selection(args)
