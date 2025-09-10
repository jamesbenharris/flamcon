import os
import argparse
import glob
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (
    MixedPrecision,
)
from torch import  nn, zeros, float32, float16, cuda, set_float32_matmul_precision, load, argmax, save
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from tqdm import tqdm
from src.flamcon import Flamcon, LayerNorm
from src.distributed import init_distributed_device, world_info_from_env
from src.dataloader import WebVidDataset, RandomVideos


from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

set_float32_matmul_precision('medium')

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
    parser.add_argument("--max_frames", default=40, type=int)
    parser.add_argument("--max_tokens", default=512, type=int)
    parser.add_argument("--lang_model", default="tiiuae/falcon-7b", type=str)
    parser.add_argument("--run_name", default="flamcon", type=str)
    parser.add_argument("--my_group", default=None, type=str)
    parser.add_argument("--delete_previous_checkpoint", default=True, type=bool)
    parser.add_argument("--resume", default=True, type=bool)
    
    args = parser.parse_args()
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    try:
        device_id = init_distributed_device(args)
    except:
        device_id = 0

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
    #Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.lang_model,do_lower_case=False)
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

    #Print parameters per GPU
    print(f"ViT parameter num: {sum(p.numel() for p in vit.parameters())} on rank {args.rank}\n")
    print(f"Language parameter num: {sum(p.numel() for p in falcon.parameters())} on rank {args.rank}\n")
    to_logits = falcon.lm_head.to(args.rank)  
    
    #Load Flamcon
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

    #Clear up vit and llm memory
    del vit
    del falcon

    #Load checkpoint if exists
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

    print("Loading model to GPU rank")
    model = model.to(args.rank)

    print('Model Loaded')
    print(f"Model parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}\n")
    print(f"Model {cuda.memory_allocated()/1024**3:.3} GB on rank {args.rank}\n")
    

    #Load data based on batch size
    if args.rank == 0:
        print("Loading Data with batch: "+str(args.batch)+"\n")
    
    #Load data, dataSampler, dataloader
    data = WebVidDataset("test_nw.csv","data",args.max_frames,tokenizer,args.max_tokens,test=True,samples=500)
    dataSampler = DistributedSampler(data, rank=args.rank, num_replicas=args.world_size, shuffle=True)
    dataloader = DataLoader(data,batch_size=1,sampler=dataSampler)

    #Start testing.
    if args.rank == 0:
        print("Starting Testing\n")
        
    if checkpoint is not None:
        del checkpoint
        
    for batchid, data in enumerate(tqdm(dataloader,position=0, desc="Iters", leave=False, colour='green', ncols=80)):
        media, X, y, file, = data
        media = media.to('cuda')
        text = model.generate(X,tokenizer,to_logits,args.rank, videos=media, n_tokens=5)
        print(file,X,text)
        print('\n')
        
if __name__ == "__main__":
    main()