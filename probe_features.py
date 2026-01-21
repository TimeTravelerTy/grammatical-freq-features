import argparse
import glob
import json
import logging
import os
from collections import defaultdict
from functools import partial

import joblib
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from nnsight import LanguageModel

from src.config import HF_TOKEN
from sparsify import Sae
from src.utils import setup_model, setup_autoencoder, dict_to_json
from src.probing.attribution import attribution_patching
from src.probing.data import MinimalPairDataset, ProbingDataset, balance_dataset
from src.probing.lingualens import DEFAULT_FEATURES, LinguaLensDataset, load_lingualens_pairs
from src.probing.utils import get_features_and_values, concept_filter, convert_probe_to_pytorch, logprob_sum_from_logits
from src.utils import get_available_concepts
# Constants
TRACER_KWARGS = {'scan': False, 'validate': False}
LOG_DIR = 'logs'
UD_BASE_FOLDER = "./data/UniversalDependencies"
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


def logprob_diff_metric(
    model,
    submodule,
    probe,
    input_ids=None,
    clean_positions=None,
    patch_logprob=None,
    **_,
):
    if input_ids is None:
        raise ValueError("input_ids required for logprob_diff_metric")
    logits = model.output[0]
    clean_logprob = logprob_sum_from_logits(logits, input_ids, clean_positions)
    if patch_logprob is None:
        return clean_logprob
    if isinstance(patch_logprob, torch.Tensor):
        patch_logprob = patch_logprob.to(clean_logprob.device)
        if patch_logprob.dim() == 0:
            patch_logprob = patch_logprob.unsqueeze(0)
        if patch_logprob.numel() == 1 and clean_logprob.numel() > 1:
            patch_logprob = patch_logprob.expand_as(clean_logprob)
    return clean_logprob - patch_logprob


def word_indices_to_token_positions(encoding, word_indices):
    if not word_indices:
        return []
    word_ids = None
    if hasattr(encoding, "word_ids"):
        try:
            word_ids = encoding.word_ids()
        except TypeError:
            try:
                word_ids = encoding.word_ids(batch_index=0)
            except Exception:
                word_ids = None
        except Exception:
            word_ids = None
    if word_ids is None:
        return []
    word_index_set = set(word_indices)
    return [i for i, wid in enumerate(word_ids) if wid in word_index_set]



def resolve_ud_base_folder(ud_base_folder):
    if os.path.exists(ud_base_folder):
        return ud_base_folder
    return ud_base_folder


def parse_ud_paths(value):
    if not value:
        return None
    paths = [path.strip() for path in value.split(",") if path.strip()]
    return paths or None


def find_ud_files(ud_base_folder, pattern, override=None):
    override_paths = parse_ud_paths(override)
    if override_paths:
        return override_paths
    matches = glob.glob(os.path.join(ud_base_folder, "**", pattern), recursive=True)
    return sorted(set(matches))


def dedupe_paths(paths):
    seen = set()
    deduped = []
    for path in paths:
        normalized = os.path.abspath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(path)
    return deduped

def resolve_submodule_device(model, submodule, fallback=None):
    for param in submodule.parameters():
        if param.device.type != "meta":
            return param.device
    model_root = getattr(model, "model", model)
    for param in model_root.parameters():
        if param.device.type != "meta":
            return param.device
    if isinstance(fallback, torch.device):
        return fallback
    if isinstance(fallback, str) and (fallback == "cpu" or fallback.startswith("cuda")):
        return torch.device(fallback)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_device(model, submodule, autoencoder):
    device = resolve_submodule_device(model, submodule)
    if device is not None:
        return device
    for param in autoencoder.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _build_model_with_transformers(model_name, device_map, torch_dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        token=HF_TOKEN,
    )
    tokenizer.pad_token = tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        token=HF_TOKEN,
    )
    if device_map not in ["auto", None]:
        hf_model.to(device_map)
    hf_model.eval()
    return LanguageModel(hf_model, tokenizer=tokenizer)


def _has_meta_params(module):
    for param in module.parameters():
        if param.device.type == "meta":
            return True
    return False


def setup_model_and_autoencoder(model_name, device_map="cuda", torch_dtype=torch.float16, load_with_transformers=False):
    """Set up the language model and autoencoder."""
    model = None
    if load_with_transformers:
        try:
            model = _build_model_with_transformers(model_name, device_map, torch_dtype)
        except Exception:
            model = None
    if model is None:
        model_device_map = None if device_map in ["cuda", "cuda:0", "cpu"] else device_map
        model = LanguageModel(
            model_name,
            torch_dtype=torch_dtype,
            device_map=model_device_map,
            token=HF_TOKEN,
            low_cpu_mem_usage=False if model_device_map is None else True,
        )
        if model_device_map is None:
            try:
                model.to(device_map)
            except Exception:
                pass
        if _has_meta_params(model.model):
            model = _build_model_with_transformers(model_name, device_map, torch_dtype)
    if _has_meta_params(model.model):
        raise RuntimeError("Model parameters are still on meta device after loading; check device_map/model loading.")
    use_llama31_sae = "llama-3.1-8b" in model_name.lower()
    layer_index = 2 if use_llama31_sae else 16
    submodule = model.model.layers[layer_index]
    ae_device = resolve_submodule_device(model, submodule, fallback=device_map)
    if use_llama31_sae:
        autoencoder = Sae.load_from_hub(
            "EleutherAI/sae-llama-3.1-8b-32x",
            hookpoint="layers.2.mlp",
        ).to(ae_device)
    elif "llama" in model_name:
        autoencoder = setup_autoencoder(device=ae_device)
    else:
        autoencoder = setup_autoencoder(checkpoint_path=AYA_AE_PATH, device=ae_device)
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
    """Process a single concept."""
    if verbose:
        print(f"Processing concept: {concept_key}:{concept_value}")

    outputs = load_progress(output_dir, concept_key, concept_value)
    if outputs.get("top_1_percent") and outputs.get("bottom_1_percent"):
        if verbose:
            print("Skipping concept - already processed.")
        return outputs

    ud_train_filepaths = find_ud_files(args.ud_base_folder, "*-ud-train.conllu", args.ud_train_file)
    if args.include_ud_dev:
        dev_paths = find_ud_files(args.ud_base_folder, "*-ud-dev.conllu")
        ud_train_filepaths = dedupe_paths((ud_train_filepaths or []) + dev_paths)

    if not ud_train_filepaths:
        if verbose:
            print("Training file not found. Skipping.")
        return outputs

    # Check if the concept exists for this dataset
    features = get_features_and_values(ud_train_filepaths)
    if concept_key not in features or concept_value not in features[concept_key]:
        if verbose:
            print(f"Concept {concept_key}:{concept_value} not found in the data. Skipping.")
        return outputs

    # Load and convert probe to PyTorch
    probe_file = os.path.join(probe_dir, f"{concept_key}_{concept_value}.joblib")
    if not os.path.exists(probe_file):
        if verbose:
            print(f"Probe file not found for {concept_key}_{concept_value}. Skipping.")
        return outputs

    probe = joblib.load(probe_file)
    torch_probe = convert_probe_to_pytorch(probe, device=resolve_device(model, submodule, autoencoder))
    for p in torch_probe.parameters():
        p.requires_grad_(False)
    try:
        print("Probe param device:", next(torch_probe.parameters()).device)
    except Exception:
        print("Probe param device:", None)

    # Prepare dataset
    train_dataset = prepare_dataset(
        ud_train_filepaths,
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
            print(f"Not enough samples in training set for {concept_key}_{concept_value}. Skipping.")
        return outputs

    # Perform attribution patching
    effects = attribution_patching_loop(train_dataset, model, torch_probe, submodule, autoencoder)

    # Select top and bottom features
    top_effects, bottom_effects = select_top_bottom_features(effects[submodule])

    outputs = {
        "top_1_percent": top_effects,
        "bottom_1_percent": bottom_effects
    }

    output_file = os.path.join(output_dir, f"{concept_key}_{concept_value}.json")
    dict_to_json(outputs, output_file)
    if verbose:
        print(f"Saved progress to {output_file}")

    return outputs


def process_lingualens_feature(args, feature, model, submodule, autoencoder, output_dir, pairs=None, verbose=True):
    if verbose:
        print(f"Processing LinguaLens feature: {feature}")

    dataset = prepare_lingualens_dataset(args, feature, pairs=pairs)
    if len(dataset) < 16:
        if verbose:
            print(f"Not enough pairs for {feature}. Skipping.")
        return None

    effects = attribution_patching_loop_lingualens(
        dataset,
        model,
        submodule,
        autoencoder,
        max_examples=args.lingualens_max_examples,
    )
    top_effects, bottom_effects = select_top_bottom_features(effects[submodule])
    outputs = {
        "feature": feature,
        "top_1_percent": top_effects,
        "bottom_1_percent": bottom_effects,
    }
    output_file = os.path.join(output_dir, f"lingualens_{feature}.json")
    dict_to_json(outputs, output_file)
    if verbose:
        print(f"Saved progress to {output_file}")
    return outputs



def prepare_dataset(
    ud_train_filepaths,
    concept_key,
    concept_value,
    seed,
    pos_tags=None,
    exclude_values=None,
    exclude_other_values=False,
    drop_conflicts=False,
):
    """Prepare and balance the dataset."""
    use_minimal_pairs = concept_key in {"Number", "Tense"}
    if use_minimal_pairs:
        train_dataset = MinimalPairDataset(
            ud_train_filepaths,
            concept_key,
            concept_value,
            pos_tags=pos_tags,
            exclude_values=exclude_values,
            exclude_other_values=exclude_other_values,
            drop_conflicts=drop_conflicts,
        )
    else:
        filter_criterion = partial(
            concept_filter,
            concept_key=concept_key,
            concept_value=concept_value,
            pos_tags=pos_tags,
            exclude_values=exclude_values,
            exclude_other_values=exclude_other_values,
            drop_conflicts=drop_conflicts,
        )
        train_dataset = ProbingDataset(ud_train_filepaths, filter_criterion)
    return balance_dataset(train_dataset, seed)


def prepare_lingualens_dataset(args, feature, pairs=None):
    if pairs is not None:
        feature_pairs = [p for p in pairs if p["feature"] == feature]
        return LinguaLensDataset(pairs=feature_pairs)
    return LinguaLensDataset(
        split=args.lingualens_split,
        language="English",
        categories=("syntax", "morphology"),
        features=[feature],
        max_samples=args.lingualens_max_samples,
        seed=args.seed,
    )

def attribution_patching_loop(dataset, model, torch_probe, submodule, autoencoder):
    """Perform attribution patching on the dataset."""
    effects = {}
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, example in enumerate(tqdm(dataloader, desc="Attribution patching")):
        if i >= 128:
            break
        if i > 0 and i % 8 == 0:
            print(f"Attribution patching: processed {i} examples")
        clean_text = example["sentence"][0]
        patch_text = None
        if "patch_sentence" in example and example["patch_sentence"]:
            patch_text = example["patch_sentence"][0]
        tokens = model.tokenizer(
            clean_text,
            return_tensors="pt",
            padding=False,
            return_attention_mask=True,
        )
        patch_tokens = None
        if patch_text:
            patch_tokens = model.tokenizer(
                patch_text,
                return_tensors="pt",
                padding=False,
                return_attention_mask=True,
            )
        e, _, _, _ = attribution_patching(
            tokens,
            model,
            torch_probe,
            [submodule],
            {submodule: autoencoder},
            patch_prefix=patch_tokens,
        )
        if submodule not in effects:
            effects[submodule] = e[submodule].sum(dim=0)
        else:
            effects[submodule] += e[submodule].sum(dim=0)
    return effects


def attribution_patching_loop_lingualens(dataset, model, submodule, autoencoder, max_examples=128):
    """Perform attribution patching on LinguaLens minimal pairs."""
    effects = {}
    for i, example in enumerate(tqdm(dataset, desc="Attribution patching (LinguaLens)")):
        if i >= max_examples:
            break
        clean_words = example.get("clean_words") or example["sentence"].split()
        patch_words = example.get("patch_words") or example["patch_sentence"].split()
        clean_tokens = model.tokenizer(
            clean_words,
            return_tensors="pt",
            padding=False,
            is_split_into_words=True,
            return_attention_mask=True,
        )
        patch_tokens = model.tokenizer(
            patch_words,
            return_tensors="pt",
            padding=False,
            is_split_into_words=True,
            return_attention_mask=True,
        )
        clean_positions = word_indices_to_token_positions(clean_tokens, example.get("clean_word_indices", []))
        patch_positions = word_indices_to_token_positions(patch_tokens, example.get("patch_word_indices", []))
        if not clean_positions or not patch_positions:
            continue

        e, _, _, _ = attribution_patching(
            clean_tokens,
            model,
            probe=None,
            submodules=[submodule],
            dictionaries={submodule: autoencoder},
            patch_prefix=patch_tokens,
            metric_fn=logprob_diff_metric,
            metric_kwargs={
                "clean_positions": clean_positions,
                "patch_positions": patch_positions,
            },
        )
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
    """Main function for feature selection across concepts."""
    model, submodule, autoencoder = setup_model_and_autoencoder(
        args.model_name,
        device_map=args.device_map,
        torch_dtype=torch.float16 if args.device_map != "cpu" else torch.float32,
        load_with_transformers=args.load_with_transformers,
    )
    for p in model.model.parameters():
        p.requires_grad_(False)
    for p in autoencoder.parameters():
        p.requires_grad_(False)

    def _dev(mod):
        try:
            return next(mod.parameters()).device
        except Exception:
            return None

    print("CUDA available:", torch.cuda.is_available())
    print("Model param device:", _dev(model.model))
    print("Submodule param device:", _dev(submodule))
    print("AE param device:", _dev(autoencoder))
    
    output_dir = f"outputs/probing/features/{'llama' if 'llama' in args.model_name else 'aya'}"
    if args.lingualens:
        output_dir = os.path.join(output_dir, "lingualens")
        os.makedirs(output_dir, exist_ok=True)
        features = args.lingualens_features or list(DEFAULT_FEATURES)
        pairs = load_lingualens_pairs(
            split=args.lingualens_split,
            language="English",
            categories=("syntax", "morphology"),
            features=features,
            max_samples=args.lingualens_max_samples,
            seed=args.seed,
        )
        grouped = defaultdict(list)
        for pair in pairs:
            grouped[pair["feature"]].append(pair)
        for feature in features:
            try:
                process_lingualens_feature(
                    args,
                    feature,
                    model,
                    submodule,
                    autoencoder,
                    output_dir,
                    pairs=grouped.get(feature, []),
                )
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA out of memory error for feature {feature}")
                cleanup_memory()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                print("Cleaned up memory and continuing with next feature...")
                continue
            except Exception as e:
                print(f"Error processing feature {feature}: {str(e)}")
                traceback.print_exc()
                print("Continuing with next feature...")
                continue
            finally:
                cleanup_memory()
        return

    probe_dir = f"outputs/probing/probes/{'llama' if 'llama' in args.model_name else 'aya'}"
    print(f"Probe directory: {probe_dir}")
    print(f"Output directory: {output_dir}")

    if args.concept_key and args.concept_value:
        concepts = [(args.concept_key, args.concept_value)]
    else:
        concepts = get_available_concepts(probe_dir)

    for concept_key, concept_value in tqdm(concepts, desc="Processing concepts"):
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
    parser.add_argument("--ud_base_folder", type=str, default=UD_BASE_FOLDER, help="Base folder for UD treebanks")
    parser.add_argument("--ud_train_file", type=str, default=None, help="Override UD training .conllu paths (comma-separated)")
    parser.add_argument("--include_ud_dev", action="store_true", help="Include UD dev files in the training set")
    parser.add_argument("--pos_tags", type=str, default=None, help="Comma-separated UPOS tags to filter by (e.g., VERB,AUX)")
    parser.add_argument("--exclude_values", type=str, default=None, help="Comma-separated concept values to exclude (e.g., Plur)")
    parser.add_argument("--exclude_other_values", action="store_true", help="Exclude sentences that contain any other values of the concept")
    parser.add_argument("--drop_conflicts", action="store_true", help="Drop sentences that contain both target and excluded values")
    parser.add_argument("--device_map", type=str, default="cuda", help="Device map for model/SAE (e.g., cuda, cpu, auto)")
    parser.add_argument("--load_with_transformers", action="store_true", help="Load model with transformers before wrapping with nnsight")
    parser.add_argument("--lingualens", action="store_true", help="Use LinguaLens minimal pairs instead of UD probes")
    parser.add_argument(
        "--lingualens_features",
        type=str,
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated LinguaLens features to use",
    )
    parser.add_argument("--lingualens_split", type=str, default="train", help="LinguaLens split to use")
    parser.add_argument("--lingualens_max_samples", type=int, default=None, help="Optional cap on LinguaLens pairs to load")
    parser.add_argument("--lingualens_max_examples", type=int, default=128, help="Max LinguaLens pairs per feature")
    args = parser.parse_args()

    args.ud_base_folder = resolve_ud_base_folder(args.ud_base_folder)
    if args.pos_tags:
        args.pos_tags = [tag.strip() for tag in args.pos_tags.split(",") if tag.strip()]
    if args.exclude_values:
        args.exclude_values = [val.strip() for val in args.exclude_values.split(",") if val.strip()]
    if args.exclude_other_values:
        args.exclude_other_values = True
    if args.drop_conflicts:
        args.drop_conflicts = True
    if args.lingualens_features:
        args.lingualens_features = [feat.strip() for feat in args.lingualens_features.split(",") if feat.strip()]

    feature_selection(args)
