"""Model identity is independent of inference hardware; unknown choices fail closed."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from .backends.precision_contract import precision_metadata


@dataclass(frozen=True)
class DecoderModel:
    name: str
    backbone: str
    arch: str
    model_id: str
    convention: str = "v2"
    apple_supported: bool = False


MODELS = {
    name: DecoderModel(name, backbone, arch, model_id, convention, apple)
    for name, backbone, arch, model_id, convention, apple in [
        ("DecoderTCR-ESMC_300M", "esmc", "DecoderTCRC_300M",
         "decodertcr@1.5.0:300M", "v2", True),
        ("DecoderTCR-ESMC_600M", "esmc", "DecoderTCRC_600M",
         "decodertcr@1.5.0:600M", "v2", True),
        ("DecoderTCR-ESMC_6B", "esmc", "DecoderTCRC_6B",
         "decodertcr@1.5.0:6B", "v2", True),
    ]
}
WEIGHT_SOURCES = ("registry", "huggingface")
ALIASES = {"esmc-300m": "DecoderTCR-ESMC_300M", "esmc-600m": "DecoderTCR-ESMC_600M",
           "esmc-6b": "DecoderTCR-ESMC_6B"}

# All three DecoderTCR 1.5.0 members are published together as a single registry
# mirror on HuggingFace. `--weight-source huggingface` resolves weights through the
# DecoderTCR package's Hub registry (hf://OWNER/REPOSITORY/PREFIX) by exact model
# ID, exactly like a local mirror, with the same manifest sha256/size verification.
# `--hf-repo OWNER/NAME` overrides the repository; the prefix is the registry root.
DEFAULT_HF_REPO = "Decoder-CHI/DecoderTCR-staging"
HF_REGISTRY_PREFIX = "registry"


def resolve_model(name: str) -> DecoderModel:
    canonical = ALIASES.get(name.lower(), name) if isinstance(name, str) else name
    if canonical not in MODELS:
        raise ValueError(f"unsupported model {name}; choose one of {sorted(MODELS)}")
    return MODELS[canonical]


def registry_model_id(name: str) -> str:
    """Return the DecoderTCR registry model ID for a Workbench model name."""
    return resolve_model(name).model_id


def normalize_weight_source(source) -> str:
    """Weights come from the internal registry by default, or HuggingFace on request."""
    value = "registry" if source is None else str(source).strip().lower()
    if value not in WEIGHT_SOURCES:
        raise ValueError(f"unsupported weight source {source!r}; choose one of {WEIGHT_SOURCES}")
    return value


def huggingface_registry(override: Optional[str] = None) -> str:
    """Resolve the ``hf://OWNER/REPOSITORY/PREFIX`` registry spec for the HF source.

    An explicit ``OWNER/NAME`` override selects a different Hub repository; the
    default is the published DecoderTCR staging mirror that carries every member.
    A spec already written as ``hf://...`` is accepted and returned unchanged.
    """
    repo = (override or "").strip() or DEFAULT_HF_REPO
    if repo.startswith("hf://"):
        return repo
    parts = [part for part in repo.split("/") if part]
    if len(parts) != 2:
        raise ValueError(
            "HuggingFace repository must be OWNER/NAME (or a full hf://OWNER/NAME/PREFIX spec); "
            f"got {override!r}"
        )
    return f"hf://{repo}/{HF_REGISTRY_PREFIX}"


def normalize_device(device: str) -> str:
    if not isinstance(device, str):
        raise ValueError("device must be cpu, gpu, cuda, cuda:N, or apple")
    value = device.strip().lower()
    if value in ("apple", "mlx"):
        return "apple"
    if value == "gpu":
        return "cuda"
    if value == "cpu" or re.fullmatch(r"cuda(?::(?:0|[1-9][0-9]*))?", value):
        return value
    raise ValueError("unsupported device; use cpu, gpu, cuda, cuda:N, or apple (MLX Metal)")


def validate_precision(device: str, precision: str) -> str:
    precision_metadata(precision)
    if precision == "float16" and device != "apple":
        raise ValueError("float16 precision is supported only on Apple ESM-C 300M; use --precision float32 for CPU/CUDA")
    return precision


def validate_backend(model: str, device: str, checkpoint, mlx_python, precision: str = "float32",
                     *, require_checkpoint: bool = True) -> DecoderModel:
    validate_precision(device, precision)
    spec = resolve_model(model)
    if device == "apple":
        if not spec.apple_supported:
            raise ValueError(f"Apple MLX supports ESM-C 300M, 600M and 6B architectures only, not {spec.name}")
        if spec.name != "DecoderTCR-ESMC_300M" and precision != "float32":
            raise ValueError(f"Apple {spec.name} currently requires float32 precision")
        if mlx_python is None:
            raise ValueError("Apple MLX requires --mlx-python PYTHON")
        # Preparation converts the bundle from the registry artifact, so the bundle
        # need not exist yet; running inference still requires the converted bundle.
        if require_checkpoint and checkpoint is None:
            raise ValueError("Apple MLX requires --checkpoint CONVERTED_BUNDLE and --mlx-python PYTHON")
    else:
        # CPU/CUDA always load the registry (or --weight-source huggingface) weights
        # selected by --model. A checkpoint is silently ignored there, so reject it
        # rather than score with weights the configuration did not actually choose.
        if checkpoint is not None:
            raise ValueError("--checkpoint applies only to --device apple; CPU/CUDA load the registry "
                             "(or --weight-source huggingface) weights selected by --model. Remove "
                             "--checkpoint and clear it from any saved settings.")
        if mlx_python is not None:
            raise ValueError("--mlx-python applies only to --device apple")
    return spec
