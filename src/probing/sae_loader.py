import json
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import load_file


def _first_tensor(state, keys):
    for key in keys:
        if key in state:
            return state[key]
    return None


def _load_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_hook_name(*configs):
    for cfg in configs:
        if not cfg:
            continue
        for key in ("hook_name", "hookpoint", "hook_point"):
            if key in cfg:
                return cfg[key]
    return None


class LocalSAE(nn.Module):
    def __init__(self, state, cfg):
        super().__init__()
        self.cfg = cfg

        w_enc = _first_tensor(state, ("W_enc", "encoder.weight", "encoder_W"))
        w_dec = _first_tensor(state, ("W_dec", "decoder.weight", "decoder_W"))
        if w_enc is None or w_dec is None:
            raise KeyError("Missing encoder/decoder weights in SAE state dict.")

        b_enc = _first_tensor(state, ("b_enc", "encoder.bias", "encoder_b"))
        b_dec = _first_tensor(state, ("b_dec", "decoder.bias", "decoder_b"))

        self.w_enc = nn.Parameter(w_enc, requires_grad=False)
        self.w_dec = nn.Parameter(w_dec, requires_grad=False)
        self.b_enc = nn.Parameter(b_enc, requires_grad=False) if b_enc is not None else None
        self.b_dec = nn.Parameter(b_dec, requires_grad=False) if b_dec is not None else None

        self._enc_transpose = None
        self._dec_transpose = None
        self._infer_shapes()

    @property
    def dict_size(self):
        return int(self._d_sae)

    def _infer_shapes(self):
        w_enc = self.w_enc
        w_dec = self.w_dec
        if w_enc.dim() != 2 or w_dec.dim() != 2:
            raise ValueError("Expected 2D SAE weights.")
        b_enc = self.b_enc
        b_dec = self.b_dec

        # Prefer bias shapes to infer orientation.
        if b_enc is not None:
            if b_enc.numel() == w_enc.shape[0]:
                self._enc_transpose = True
                self._d_sae = w_enc.shape[0]
                self._d_model = w_enc.shape[1]
            elif b_enc.numel() == w_enc.shape[1]:
                self._enc_transpose = False
                self._d_sae = w_enc.shape[1]
                self._d_model = w_enc.shape[0]
            else:
                raise ValueError(
                    f"Encoder bias shape {tuple(b_enc.shape)} does not match W_enc {tuple(w_enc.shape)}"
                )
        else:
            # Fall back to assuming encoder is (d_sae, d_model).
            self._enc_transpose = True
            self._d_sae = w_enc.shape[0]
            self._d_model = w_enc.shape[1]

        expected_enc_shape = (self._d_sae, self._d_model) if self._enc_transpose else (self._d_model, self._d_sae)
        if tuple(w_enc.shape) != expected_enc_shape:
            raise ValueError(
                f"Encoder weight shape {tuple(w_enc.shape)} does not match expected {expected_enc_shape}"
            )

        if b_dec is not None:
            if b_dec.numel() == w_dec.shape[0]:
                self._dec_transpose = True
            elif b_dec.numel() == w_dec.shape[1]:
                self._dec_transpose = False
            else:
                raise ValueError(
                    f"Decoder bias shape {tuple(b_dec.shape)} does not match W_dec {tuple(w_dec.shape)}"
                )
        else:
            # Assume decoder is (d_model, d_sae), requiring transpose.
            self._dec_transpose = True

        expected_dec_shape = (self._d_model, self._d_sae) if self._dec_transpose else (self._d_sae, self._d_model)
        if tuple(w_dec.shape) != expected_dec_shape:
            raise ValueError(
                f"Decoder weight shape {tuple(w_dec.shape)} does not match expected {expected_dec_shape}"
            )

    def _linear(self, x, weight, bias, transpose):
        if transpose:
            weight = weight.t()
        out = x @ weight
        if bias is not None:
            out = out + bias
        return out

    def encode(self, x):
        z = self._linear(x, self.w_enc, self.b_enc, self._enc_transpose)
        return torch.relu(z)

    def decode(self, f):
        return self._linear(f, self.w_dec, self.b_dec, self._dec_transpose)


def load_local_sae(path, device="cpu"):
    state_path = os.path.join(path, "final.safetensors")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing SAE weights at {state_path}")

    state = load_file(state_path, device=device)
    hyperparams = _load_json(os.path.join(path, "hyperparams.json"))
    lm_config = _load_json(os.path.join(path, "lm_config.json"))
    hook_name = _extract_hook_name(hyperparams, lm_config)
    cfg = SimpleNamespace(**hyperparams)
    cfg.hook_name = hook_name
    if not hasattr(cfg, "d_sae"):
        cfg.d_sae = None

    sae = LocalSAE(state, cfg)
    sae.to(device)
    return sae
