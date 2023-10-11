import os
import argparse
import glob
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch import  nn, zeros, float32, float16, cuda, set_float32_matmul_precision, load, argmax, save
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler,DataLoader

from tqdm import tqdm
from src.flamcon import Flamcon, LayerNorm
from src.distributed import init_distributed_device, world_info_from_env
from src.misc import save_checkpoint
from src.dataloader import WebVidDataset, RandomVideos


from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from einops import rearrange
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam


set_float32_matmul_precision('medium')

def tokenize(tokenizer,text):
    tokenizer.padding_side = "right"
    text =  tokenizer(
        text,
        max_length=512,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )   
    return  text['input_ids'], text['attention_mask'].bool()

def getLoss(predicted, actual, logits):
    """
    Compute cross-entropy loss between predicted and actual tokens.

    Args:
        predicted (Tensor): Predicted tokens.
        actual (Tensor): Actual tokens.
        logits (Callable): Function to compute logits.

    Returns:
        loss (Tensor): Cross-entropy loss.
    """
    predicted = logits(predicted)
    predicted = rearrange(predicted, 'b n c -> b c n')
    actual = actual[:, 0:]
    loss = F.cross_entropy(predicted, actual, ignore_index=0)
    return loss
    
def test(args, model, rank, dialogue, media, attention_mask, tokenizer, to_logits):
    """
    Perform the training loop for one epoch.

    Args:
        args: Parsed command-line arguments.
        model: The Flamcon model.
        rank (int): Process rank.
        dialog: tokens
        media: image/video

    Returns:
        text_tokens: the output the prediction
    """
    input_ids, attention_mask = tokenize(tokenizer,dialogue)
    input_ids = input_ids.to(rank)
    media = media.to(rank)
    if args.video:
        text_tokens = model.generate(input_ids, tokenizer, to_logits, videos=media)
    else:
        text_tokens = model.generate(input_ids, tokenizer, to_logits, images=media)
    return text_tokens

        
def main():
    """
    Main entry point of the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--horovod", default=False, type=bool)
    parser.add_argument("--dist_backend", default="nccl", type=str)
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--no_set_device_rank", default=False, type=bool)
    parser.add_argument("--cpu_offload", default=True, type=bool)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--dim", default=4544, type=int)
    parser.add_argument("--num_tokens", default=65027, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--fsdp", default=True, type=bool)
    parser.add_argument("--video", default=True, type=bool)
    parser.add_argument("--max_frames", default=10, type=int)
    parser.add_argument("--max_tokens", default=512, type=int)
    parser.add_argument("--lang_model", default="tiiuae/falcon-7b", type=str)
    parser.add_argument("--run_name", default="flamcon", type=str)
    parser.add_argument("--my_group", default=None, type=str)
    parser.add_argument("--delete_previous_checkpoint", default=True, type=bool)
    parser.add_argument("--resume", default=True, type=bool)
    
    args = parser.parse_args()
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)

    if args.rank == 0:
        print("Loading ViT\n")
        
    #Load vision transformer
    vit = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = args.dim,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    vit = Extractor(vit, return_embeddings_only = True).to('cuda')

    if args.rank == 0:
        print("Loading Falcon\n")
        
    #Loads language model
    falcon = AutoModelForCausalLM.from_pretrained(
                args.lang_model,
                trust_remote_code=True
            )
    tokenizer = AutoTokenizer.from_pretrained(args.lang_model)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<media>"]}
    )
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    falcon.resize_token_embeddings(new_num_tokens=len(tokenizer))
    
    #set Mixes precision policy
    mp_policy = MixedPrecision(
        param_dtype=float32,
        reduce_dtype=float16,  # gradient communication
        buffer_dtype=float16,
    )

    if args.rank == 0:
        print("Loading Flamcon")
    
    #Print parameters per GPU
    print(f"ViT parameter num: {sum(p.numel() for p in vit.parameters())} on rank {args.rank}\n")
    print(f"Language parameter num: {sum(p.numel() for p in falcon.parameters())} on rank {args.rank}\n")
    
    model = Flamcon(
                    num_tokens = len(tokenizer),       # number of tokens
                    dim = args.dim,                     # dimensions
                    depth = 32,                         # depth
                    heads = 8,                          # attention heads
                    dim_head = 64,                      # dimension per attention head
                    img_encoder = vit,                  # plugin your image encoder (this can be optional if you pass in the image embeddings 
                    media_token_id = tokenizer.encode("<media>")[-1],                 # the token id representing the [media] or [image]
                    cross_attn_every = 3,               # how often to cross attend
                    perceiver_num_latents = 64,         # perceiver number of latents, should be smaller than the sequence length of the image tokens
                    perceiver_depth = 2,                # perceiver resampler depth
                    max_video_frames = args.max_frames, # max video frames
                    lang_model = falcon                 # llm
                    )

    del vit
           
    resume_from_epoch = 0
    checkpoint = None
    if os.path.exists(f"{args.run_name}") and args.resume:
        checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.run_name}.\n")
        else:
            resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(
                f"Found checkpoint {resume_from_checkpoint} for run {args.run_name}.\n"
            )
            checkpoint = load(resume_from_checkpoint, map_location="cpu")
            resume_from_epoch = checkpoint["epoch"] + 1
            if args.rank == 0:
                model.load_state_dict(checkpoint["model_state_dict"], False)
        
        
    if args.fsdp:
        fsdp_args = dict(
                    process_group=None,
                    cpu_offload=CPUOffload(offload_params=args.cpu_offload),
                    device_id=device_id,
                    sync_module_states =True,  # broadcast loaded ckpt from rank 0 -> all ranks
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    use_orig_params=False,
                    mixed_precision=mp_policy,
                    forward_prefetch=False,
                    backward_prefetch=BackwardPrefetch.BACKWARD_POST,
                    limit_all_gathers=True,
                )        
        model.get_fsdp(fsdp_args)
    else:
        model = model.to(device_id)
        model = DDP(model, device_ids=[device_id])
    #Load to_logits for loss
    
    to_logits = falcon.lm_head.to(args.rank)  
    del falcon
    print(f"Model parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}\n")
    print(f"Model {cuda.memory_allocated()/1024**3:.3} GB on rank {args.rank}\n")
    

    #Load data based on batch size
    if args.rank == 0:
        print("Loading Data with batch: "+str(args.batch)+"\n")
    
    #For quick validation use generated data.
    #data = RandomVideos(length=args.batch,frames=args.max_frames)

    data = WebVidDataset("test.csv","data",args.max_frames,tokenizer,args.max_tokens,test=True)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data,
                                        batch_size=args.batch,
                                        sampler=sampler)

    #Start training
    if args.rank == 0:
        print("Starting Testing\n")
        
    if checkpoint is not None:
        del checkpoint
        
    for batchid, data in enumerate(tqdm(dataloader,position=0, desc="Iters", leave=False, colour='green', ncols=80)):
        media, dialog, dialogue_test, file, = data
        dialog,attention_mask  = tokenize(tokenizer,dialog)
        text = test(args, model, args.rank, dialogue_test, media, attention_mask, tokenizer, to_logits)
        print(file[0],text)
        print('\n')
        if batchid > 3:
            break
        
if __name__ == "__main__":
    main()
