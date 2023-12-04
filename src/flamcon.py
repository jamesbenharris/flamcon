import os 
import math
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from src.flamingo import GatedCrossAttentionBlock, PerceiverResampler
from src.misc import genTokenize
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from labml_nn.sampling.nucleus import NucleusSampler
from labml_nn.sampling.temperature import TemperatureSampler
from labml.logger import Text

# Helper function to check if a value exists
def exists(val):
    return val is not None

# Helper function to set requires_grad flag for module's parameters
def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

# Helper function to freeze all layers of a module
def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

# Helper function to unfreeze all layers of a module
def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

# Helper function to freeze the model and set it to evaluation mode
def freeze_model_and_make_eval_(model):
    freeze_all_layers_(model)
    
# Layer normalization class with learnable scaling and bias
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# Main Flamingo model for multimodal fusion
class Flamcon(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        media_token_id=3,
        cross_attn_every=3,
        img_encoder=None,
        perceiver_num_latents=64,
        perceiver_depth=2,
        max_video_frames=10,
        only_attend_immediate_media=True,
        lang_model=None,
    ):
        super().__init__()
        # Initialize model parameters and components
        self.dim = dim
        self.media_token_id = media_token_id
        self.img_encoder = img_encoder
        self.dim_head = dim_head
        self.heads = heads
        self.model = lang_model
        self.only_attend_immediate_media = only_attend_immediate_media

        # Create video frame positional embeddings if applicable
        self.video_frame_pos_emb = nn.Parameter(torch.randn(max_video_frames, self.dim)) if exists(max_video_frames) else None
        
        # Embedding layer for text tokens
        self.token_emb = lang_model.get_input_embeddings()

        # Init PerceiverResampler
        self.perceiver_resampler = PerceiverResampler(
            dim=self.dim,
            depth=perceiver_depth,
            dim_head=self.dim_head,
            heads=self.heads,
            num_latents=perceiver_num_latents,
            num_media_embeds = max_video_frames
        )
        self.layers = nn.ModuleList([])
        for ind in range(depth):
            self.layers.append(self.get_flamingo_layer(ind, cross_attn_every))
            
    def prep_images(self,images,batch):
        images = rearrange(images, 'b t ... -> (b t) ...')
        with torch.no_grad():
            embeds = self.img_encoder(images)
        return rearrange(embeds, '(b t) ... -> b t ...', b=batch)
    
    def prep_videos(self,videos):
            batch, media, num_times, *_ = videos.shape
            videos = rearrange(videos, '... c h w -> (...) c h w')

            with torch.no_grad():
                embeds = self.img_encoder(videos)

            embeds = rearrange(embeds, '(b m t) ... -> b m t ...', b=batch, m=media, t=num_times)
            video_time_pos_emb = repeat(self.video_frame_pos_emb[:num_times], 't d -> b m t n d', b=batch, m=media, n=embeds.shape[-2]).to(embeds.device)
            embeds = embeds + video_time_pos_emb
            return rearrange(embeds, 'b m t n d -> b m (t n) d')
    
    # Clip gradient norms for all layers
    def get_fsdp(self,wrapper_kwargs):
        layers = nn.ModuleList([]) 
        with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
            self.perceiver_resampler = wrap(wrap(self.perceiver_resampler))
            self.token_emb = wrap(wrap(self.token_emb))
            self.img_encoder = wrap(wrap(self.img_encoder))
            for transformer,gated in self.layers:
                flamingo = nn.ModuleList([])
                transformer = wrap(wrap(transformer))
                if gated is not None:
                    gated = wrap(wrap(gated))
                flamingo.append(transformer)
                flamingo.append(gated)
                layers.append(flamingo)
            self.layers = layers
            
        for transformer in self.model.transformer.h:      
            for p in transformer.parameters():
                p.exclude_from_optimizer = True    
            
        def clip_grad_norm_(max_norm):
            self.perceiver_resampler.clip_grad_norm_(max_norm)
            for layer in self.layers:
                gated = layer[1]
                if isinstance(gated, FSDP):
                    gated.clip_grad_norm_(max_norm)
        self.clip_grad_norm_ = clip_grad_norm_

    # Construct a Flamingo layer consisting of transformer and gated cross-attention
    def get_flamingo_layer(self, ind, cross_attn_every):
        flamingo = nn.ModuleList([])
        transformer = self.model.transformer.h[ind]
        gated = GatedCrossAttentionBlock(dim=self.dim, dim_head=self.dim_head, heads=self.heads, only_attend_immediate_media=self.only_attend_immediate_media) if not (ind % cross_attn_every) else None
        flamingo.append(transformer)
        flamingo.append(gated)
        return flamingo
        
    def loop(self,embeds,text_tokens,media_locations,attention_mask):
        #alibi = build_alibi_tensor(attention_mask,71,text_tokens.dtype).to(attention_mask.device)
        for attn_ff, flamingo_cross_attn in self.layers:
            if exists(flamingo_cross_attn) and exists(embeds):
                text_tokens = flamingo_cross_attn(
                    text_tokens,
                    embeds,
                    media_locations=media_locations
                )
            text_tokens = attn_ff(text_tokens, alibi=None, attention_mask=attention_mask)[0]
        return text_tokens
    
    # Forward pass through the Flamingo model
    def forward(
        self,
        text,
        *,
        gen=False,
        images=None,
        videos=None,
        embeds=None,
        attention_mask=None
    ):
        batch = text.shape[0]

        # Determine if the model is in Flamingo mode
        flamingo_mode = any([exists(t) for t in (images, videos, embeds)])
        
        # Freeze or unfreeze layers based on Flamingo mode
        if (flamingo_mode and not gen):
            
            freeze_all_layers_(self)
            unfreeze_all_layers_(self.perceiver_resampler)
            unfreeze_all_layers_(self.img_encoder)
            [unfreeze_all_layers_(cross_attn) for _, cross_attn in self.layers if exists(cross_attn)]
        elif(gen):
            freeze_model_and_make_eval_(self)
        else:
            unfreeze_all_layers_(self)

        # Determine media token positions for masked cross-attention
        if flamingo_mode:
            media_locations = text == self.media_token_id

        # Embed text tokens
        text_tokens = self.token_emb(text)

        assert not (exists(embeds) and (exists(images) or exists(videos)))

        # Encode images or videos into embeddings
        if exists(images):
            assert exists(self.img_encoder), 'img_encoder must be passed in for automatic image encoding'
            embeds = self.prep_images(images,batch)
        if exists(videos):
            assert exists(self.img_encoder), 'img_encoder must be passed in for automatic video encoding'
            embeds = self.prep_videos(videos)
        if exists(embeds):
            embeds = self.perceiver_resampler(embeds)
        return self.loop(embeds,text_tokens,media_locations,attention_mask)
        
    def test(
        self,
        text,
        *,
        images=None,
        videos=None,
        embeds=None,
        gen=True,
        attention_mask=None
        
    ): 
        return self.forward(text,images=images,videos=videos,embeds=embeds,gen=gen,attention_mask=attention_mask)
    
    def generate(
        self,
        text,
        tokenizer,
        to_logits,
        rank,
        *,
        images=None,
        videos=None,
        embeds=None,
        gen=True,
        attention_mask=None,
        n_tokens = 5,        
    ):
        
        sampler = NucleusSampler(0.95, TemperatureSampler(1.))
        space = torch.Tensor([204]).to(rank).int()
        for i in range(n_tokens):
            data, attention_mask = genTokenize(text,tokenizer,rank)
            seq_len = len(data)
            data = data[-seq_len:]
            output = self.forward(data,images=images,videos=videos,embeds=embeds,gen=gen,attention_mask=attention_mask)
            logits = to_logits(output)
            logits = logits[:, -1]
            res = sampler(logits)
            data = torch.cat([data, space.reshape(space.shape[0],1)], dim=1)
            data = torch.cat([data, res.reshape(res.shape[0],1)], dim=1)
            text = tokenizer.decode(data[0]).replace('  ',' ')
        return text