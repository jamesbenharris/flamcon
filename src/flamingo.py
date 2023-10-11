import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many

def exists(val):
    return val is not None

def FeedForward(dim, mult = 4):
    """
    A feedforward neural network module with layer normalization, linear transformations,
    GELU activation, and linear transformation layers.

    Args:
        dim (int): Input and output dimension.
        mult (int): Multiplier for the hidden dimension.

    Returns:
        nn.Sequential: Feedforward module.
    """
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class PerceiverAttention(nn.Module):
    """
    Perceiver attention mechanism.

    Args:
        dim (int): Input dimension.
        dim_head (int): Dimension per head.
        heads (int): Number of attention heads.
    """
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = self.dim_head * self.heads
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents):
        """
        Forward pass of the Perceiver attention mechanism.

        Args:
            x (torch.Tensor): Input tensor (media).
            latents (torch.Tensor): Latent tensor (text).

        Returns:
            torch.Tensor: Output tensor after attention.
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * self.scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler module for image/video captioning.

    Args:
        dim (int): Input and output dimension.
        depth (int): Number of attention and feedforward layers.
        dim_head (int): Dimension per head.
        heads (int): Number of attention heads.
        num_latents (int): Number of latent vectors.
        num_media_embeds (int): Number of media positional embeddings.
        ff_mult (int): Multiplier for the hidden dimension in feedforward layers.
    """
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_media_embeds = 20,
        ff_mult = 4
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Forward pass of the Perceiver Resampler module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through attention and feedforward layers.
        """
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')
        times = x.shape[1]
        x = x + self.media_pos_emb[:times]

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents)

# gated cross attention

class MaskedCrossAttention(nn.Module):
    """
    Masked Cross-Attention mechanism for gated attention between text and media data.

    Args:
        dim (int): Input dimension.
        dim_head (int): Dimension per head.
        heads (int): Number of attention heads.
        only_attend_immediate_media (bool): Whether to attend only to immediate media.
    """
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        only_attend_immediate_media = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.only_attend_immediate_media = only_attend_immediate_media
        inner_dim = dim_head * self.heads
        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        media,
        media_locations = None
    ):
        b, t, m = media.shape[:3]
        h = self.heads
        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, 'b t n d -> b (t n) d')
        k, v = self.to_kv(media).chunk(2, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        if exists(media_locations):
            text_time = media_locations.cumsum(dim = -1) # at each boolean of True, increment the time counter (relative to media time)
            
            media_time = torch.arange(t, device = x.device) + 1
            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(rearrange(text_time, 'b i -> b 1 i 1'), repeat(media_time, 'j -> 1 1 1 (j m)', m = m))
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
            attn = attn.masked_fill(text_without_media_mask, 0.)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class GatedCrossAttentionBlock(nn.Module):
    """
    Gated Cross-Attention block.

    Args:
        dim (int): Input and output dimension.
        dim_head (int): Dimension per head.
        heads (int): Number of attention heads.
        ff_mult (int): Multiplier for the hidden dimension in feedforward layers.
        only_attend_immediate_media (bool): Whether to attend only to immediate media.
        device (str): Device to use ('cuda' or 'cpu').
    """
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        only_attend_immediate_media = True,
        device = 'cuda'
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.ff_mult = ff_mult
        self.only_attend_immediate_media = only_attend_immediate_media
        self.attn = MaskedCrossAttention(dim = self.dim, dim_head = self.dim_head, heads = self.heads, only_attend_immediate_media = self.only_attend_immediate_media)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(self.dim, mult = self.ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(
        self,
        x,
        media,                  # media tensor, encoded by perceiver resample - (batch, time, latents, dim)
        media_locations = None  # boolean tensor indicating positions of media - (batch, sequence)
    ):
        
        """
        Forward pass of the Gated Cross-Attention block.

        Args:
            x (torch.Tensor): Input tensor.
            media (torch.Tensor): Media tensor.
            media_locations (torch.Tensor): Boolean tensor indicating media positions.

        Returns:
            torch.Tensor: Output tensor after gated cross-attention and feedforward operations.
        """
        x = self.attn(x, media, media_locations = media_locations) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh()  + x
        return x