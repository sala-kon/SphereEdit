import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import AttnProcessor
from diffusers.models.attention_processor import Attention


class CrossAttnWeightProcessor(AttnProcessor):
    """
    Drop-in replacement that stores raw cross-attention probabilities
    in .saved_probs each time the layer runs:
        shape = (batch*heads, spatial, tokens)
    """
    def __init__(self):
        super().__init__()
        self.saved_probs = None      # will be overwritten each step

    def __call__(                    # ← signature required by diffusers
        self,
        attn,                        # the Attention layer that owns us
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs                     # diffusers ≥0.27 passes scale, etc.
    ):
        bsz, seq_len, _ = hidden_states.shape

        # ---- standard q, k, v projections ---------------------------------
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key   = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
        value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))

        # ---- scaled-dot attention -----------------------------------------
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, bsz)
        probs = attn.get_attention_scores(query, key, attention_mask)  # (B·H, S, T)
        self.saved_probs = probs.detach()

        # ---- finish forward pass exactly like diffusers default -----------
        hidden_states = torch.bmm(probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AttentionHook:
    """Captures cross-attention maps from UNet without modifying forward pass"""
    def __init__(self, locate_middle_block: bool = False):
        self.attention_maps = []
        self.hook_handles = []
        self.layer_names = []
        self.locate_middle_block = locate_middle_block

    def __call__(self, module, input, output):
        # Store attention weights from cross-attention layers
        if isinstance(module.processor, torch.nn.Module):  # For xFormers compatibility
            self.attention_maps.append(output[1].detach())
        else:
            self.attention_maps.append(output.detach())

    def attach(self, unet):
        # Attach to all cross-attention layers
        for name, module in unet.named_modules():
            if "attn2" in name and "to_" not in name:  # Cross-attention blocks
                if "mid" in name and not self.locate_middle_block:
                    continue
                handle = module.register_forward_hook(self)
                self.hook_handles.append(handle)
                self.layer_names.append(name)

    def remove(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    @staticmethod
    def get_spatial_dims(seq_len: int):
        """Find factor pair closest to square for sequence length."""
        pairs = []
        for i in range(1, int(seq_len**0.5) + 1):
            if seq_len % i == 0:
                pairs.append((i, seq_len // i))
        return max(pairs, key=lambda x: x[0] / x[1])  # Most square-like pair


class AttentionGrabber:
    """
    Installs CrossAttnWeightProcessor on each cross-attention layer and
    tracks them so you can retrieve saved_probs later.
    """
    def __init__(self, unet, include_mid=False):
        self.processors = {}          # {layer_name: processor}
        for name, attn in unet.named_modules():
            if not (isinstance(attn, Attention) and "attn2" in name and "to_" not in name):
                continue
            if "mid" in name and not include_mid:
                continue
            proc = CrossAttnWeightProcessor()
            attn.set_processor(proc)
            self.processors[name] = proc

    def clear(self):
        """Call at the start of a new diffusion run if you need a blank slate."""
        for p in self.processors.values():
            p.saved_probs = None

    def first_probs(self):
        """Returns one tensor (usually the highest-res layer) or None."""
        for p in self.processors.values():
            if p.saved_probs is not None:
                return p.saved_probs
        return None

    def all_probs(self):
        """Returns dict {name: tensor} for all layers with saved probs."""
        return {n: p.saved_probs for n, p in self.processors.items() if p.saved_probs is not None}


class AttentionAggregator:
    """
    Convert raw UNet cross-attention tensors
       (B·H,  S, 77)  →  (77, H₀, W₀)
    and aggregate across layers.

    Parameters
    ----------
    batch_cfg : int
        Batch multiplier from classifier-free guidance (default 2).
    n_heads   : int
        Number of heads per layer (8 for SD-v1/v2, 12 for SD-XL).
    target_hw : int
        Spatial resolution for output maps (64 ⇒ 64×64 grid).
    pool      : str
        'mean' or 'sum' aggregation across layers.
    """
    def __init__(self, batch_cfg=2, n_heads=8, target_hw=64, pool="mean"):
        assert pool in {"mean", "sum"}
        self.batch_cfg  = batch_cfg
        self.n_heads    = n_heads
        self.target_hw  = target_hw
        self.pool       = pool

        self.layer_maps = []     # will hold each (77, H₀, W₀)
        self.global_map = None

    def add_raw(self, raw: torch.Tensor):
        """
        Feed one raw attention tensor of shape (B·H, S, 77).
        """
        self.layer_maps.append(self._to_heat_map(raw))

    def compute(self):
        """
        Aggregate all added layers into self.global_map and return it.
        """
        if not self.layer_maps:
            raise RuntimeError("No raw tensors added")

        stack = torch.stack(self.layer_maps, 0)          # (L, 77, H₀, W₀)
        if self.pool == "mean":
            self.global_map = stack.mean(0)
        else:
            self.global_map = stack.sum(0)

        return self.global_map

    def _to_heat_map(self, raw):
        """
        Convert (B·H, S, 77) → (77, target_hw, target_hw).
        """
        bh, S, T = raw.shape
        h = w = int(S ** 0.5)

        raw = raw.view(self.batch_cfg, self.n_heads, S, T)   # (2, H, S, 77)
        cond = raw[1].mean(0).transpose(0, 1)                # (77, S)
        maps = cond.view(T, h, w)                            # (77, h, w)

        if h != self.target_hw:
            maps = F.interpolate(
                maps.unsqueeze(0),
                size=(self.target_hw, self.target_hw),
                mode="bilinear",
                align_corners=False,
            )[0]
        return maps
