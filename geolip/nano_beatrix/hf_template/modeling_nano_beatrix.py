"""The transformers AutoModel wrapper of the nano Beatrix v1 crafts: small byte-level AlephLLM models (splat hubs with an aleph address,
rotary softmax, dual hubs, Canon layers, an aleph-addressed expert bank with a byte table).

Loads with:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("AbstractPhil/alephllm-nano-castor-pollux", subfolder="<craft>",
                                                 trust_remote_code=True)
The model reads raw UTF-8 bytes: input_ids are byte values 0..255 (no tokenizer). Generation re-reads the whole window each step
(no cache: the Canon layers and the dual hubs decode by the full forward).
"""
from __future__ import annotations

from dataclasses import fields

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .presets import AlephLMConfig
from .alephlm import AlephLM

_FIELDS = [f.name for f in fields(AlephLMConfig)]


class NanoBeatrixConfig(PretrainedConfig):
    """Every field of the craft's AlephLMConfig as an attribute (the training run's model_config), plus the transformers keys."""
    model_type = "nano-beatrix"
    attribute_map = {"num_hidden_layers": "n_layers", "hidden_size": "d_model", "num_attention_heads": "n_heads",
                     "max_position_embeddings": "context"}

    def __init__(self, **kwargs):
        defaults = AlephLMConfig().to_dict()
        for k in _FIELDS:
            v = kwargs.pop(k, defaults[k])
            setattr(self, k, list(v) if isinstance(v, tuple) else v)
        kwargs.setdefault("use_cache", False)
        super().__init__(**kwargs)

    def to_aleph(self) -> AlephLMConfig:
        d = {k: getattr(self, k) for k in _FIELDS}
        d["hub_ckpt"] = 0                      # inference: no recompute
        return AlephLMConfig.from_dict(d)


class NanoBeatrixForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NanoBeatrixConfig
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _tied_weights_keys = {}

    def __init__(self, config: NanoBeatrixConfig):
        super().__init__(config)
        self.model = AlephLM(config.to_aleph())
        self.post_init()

    def _computed_buffers(self) -> dict:
        """The buffers the craft computes at construction (never stored in the weight files), from a fresh build on the CPU."""
        if "_computed" not in self.__dict__:
            with torch.random.fork_rng(devices=[]), torch.device("cpu"):
                ref = AlephLM(self.config.to_aleph())
            self.__dict__["_computed"] = {"model." + n: b for n, b in ref.named_buffers()}
        return self.__dict__["_computed"]

    @torch.no_grad()
    def _init_weights(self, module):
        # Every parameter and every stored buffer comes from the weight file; nothing is re-initialized. The loader re-creates the
        # buffers that are not stored (the byte-class table) as empty memory and asks this method to fill them: each is copied from
        # a fresh build of the same craft.
        own = [n for n in getattr(module, "_non_persistent_buffers_set", ()) if getattr(module, n, None) is not None]
        if not own or all(getattr(module, n).device.type == "meta" for n in own):
            return
        name = next((n for n, m in self.named_modules() if m is module), None)
        if name is None:
            return
        ref = self._computed_buffers()
        for n in own:
            buf, key = getattr(module, n), (name + "." if name else "") + n
            if buf.device.type != "meta" and key in ref:
                buf.copy_(ref[key].to(dtype=buf.dtype))

    def forward(self, input_ids=None, labels=None, attention_mask=None, past_key_values=None, use_cache=None, **kwargs):
        # attention_mask is ignored: a causal byte model trained on packed rows; right-pad with masked labels.
        if input_ids.shape[1] > self.config.context:
            input_ids = input_ids[:, -self.config.context:]
            if labels is not None:
                labels = labels[:, -self.config.context:]
        out = self.model(input_ids, labels=labels)
        return CausalLMOutputWithPast(loss=out[1] if labels is not None else None, logits=out[0])

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids[:, -self.config.context:]}
