"""CPU initialization for LAVIS ViT-g using a file-backed float32 state."""
from functools import partial
import hashlib
import math
from pathlib import Path
from unittest.mock import patch


def verify(path, digest):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    if result.hexdigest() != digest:
        raise ValueError(f"Checkpoint checksum mismatch: {path.name}")


def load_cpu_model(weights, scratch):
    import torch
    from torch import nn
    from lavis.models.blip2_models.blip2 import LayerNorm
    from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
    from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
    from lavis.models.eva_vit import VisionTransformer, convert_weights_to_fp16
    from lavis.processors.blip_processors import BlipCaptionProcessor, BlipImageEvalProcessor
    from transformers import BertTokenizer

    weights, scratch = Path(weights), Path(scratch)
    stage = weights / "blip2_pretrained.pth"
    eva = weights / "eva_vit_g.pth"
    verify(stage, "f31f96e4a97ce9ac7a140ea1961b8a871822ae1cd0a23f0f1122fbefd70db9f0")
    verify(eva, "99d2bb36c6b52c94fe6e2e12373afb27de57ae81378c3d8c53bf0e83b0f4275f")
    bert = weights / "bert-base-uncased"
    verify(bert / "config.json", "7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20")
    verify(bert / "vocab.txt", "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3")

    class CheckpointModel(Blip2Qformer):
        # LAVIS v1.0.2 constructors fetch, initialize and copy multiple large
        # checkpoints. Only substitute initialization, all forward and feature
        # extraction methods are inherited from the release.
        @classmethod
        def init_tokenizer(cls):
            tokenizer = BertTokenizer.from_pretrained(str(bert))
            tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            return tokenizer

        @classmethod
        def init_vision_encoder(cls, name, size, drop, checkpoint, precision):
            model = VisionTransformer(
                img_size=size, patch_size=14, use_mean_pooling=False, embed_dim=1408,
                depth=39, num_heads=16, mlp_ratio=4.3637, qkv_bias=True,
                drop_path_rate=drop, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                use_checkpoint=checkpoint,
            )
            convert_weights_to_fp16(model)
            return model, LayerNorm(model.num_features)

        @classmethod
        def init_Qformer(cls, count, width, cross_attention_freq=2):
            config = BertConfig.from_pretrained(str(bert))
            config.encoder_width = width
            config.add_cross_attention = True
            config.cross_attention_freq = cross_attention_freq
            config.query_length = count
            return BertLMHeadModel(config), nn.Parameter(torch.empty(1, count, config.hidden_size))

    # LAVIS calls .item() on a 39-element linspace for drop path.
    cpu_linspace = partial(torch.linspace, device="cpu")
    with torch.device("meta"), patch.object(torch, "linspace", cpu_linspace):
        model = CheckpointModel()

    expected = model.state_dict()
    mapped = {}
    for source, prefix in ((eva, "visual_encoder."), (stage, "")):
        state = torch.load(str(source), map_location="cpu", mmap=True, weights_only=True)
        if source == stage:
            state = state["model"]
        for name, tensor in state.items():
            key = prefix + name
            # The checkpoint includes block 39, final norm and head. LAVIS
            # constructs 39 blocks (zero-based 0 through 38).
            if key not in expected:
                if source == eva and (name.startswith("blocks.39.") or
                                      name in ("norm.weight", "norm.bias", "head.weight", "head.bias")):
                    continue
                raise KeyError(f"Unexpected checkpoint tensor {key}")
            target = expected[key]
            if tensor.shape != target.shape:
                raise ValueError(f"Shape mismatch for {key}: {tensor.shape} vs {target.shape}")
            if key in mapped:
                raise ValueError(f"Duplicate checkpoint tensor {key}")
            # Reproduce LAVIS' initial conversion of vision Conv/Linear weights
            # to fp16 followed by load_model_and_preprocess(..., device='cpu').float().
            dtype = torch.float32 if target.is_floating_point() else target.dtype
            path = scratch / f"{len(mapped)}.bin"
            storage = torch.from_file(str(path), shared=True, size=math.prod(target.shape), dtype=dtype)
            mapped[key] = storage.view(target.shape)
            mapped[key].copy_(tensor.to(target.dtype))
        del tensor, state
    model.load_state_dict(mapped, strict=True, assign=True)
    if any(p.is_meta or p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("Incomplete CPU model materialization")
    return model.eval(), BlipImageEvalProcessor(image_size=224), BlipCaptionProcessor()
