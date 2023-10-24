import logging
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
from torch import  nn, zeros, float32, float16, cuda, set_float32_matmul_precision, load, argmax, Size, device, Tensor, BoolTensor,tensor
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from tqdm import tqdm
from src.flamcon import Flamcon, LayerNorm
from src.distributed import init_distributed_device, world_info_from_env
from src.misc import save_checkpoint,_prepare_attn_mask
from src.dataloader import WebVidDataset, RandomVideos

from labml_nn.sampling.nucleus import NucleusSampler
from labml_nn.sampling.temperature import TemperatureSampler

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


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)  
    
def getLoss(predicted, labels, logits, tokenizer):
    """
    Compute cross-entropy loss between predicted and actual tokens.

    Args:
        predicted (Tensor): Predicted tokens.
        actual (Tensor): Actual tokens.
        logits (Callable): Function to compute logits.

    Returns:
        loss (Tensor): Cross-entropy loss.
    """
    sampler_nuke = NucleusSampler(0.95, TemperatureSampler(1.))
    #predicted = rearrange(predicted, 'b n c -> b c n')
    labels[labels == tokenizer.pad_token_id] = -100
    labels[labels == tokenizer.eos_token] = -100
    labels[labels == tokenizer.encode("<media>")[-1]] = -100
    #labels = labels[labels != tokenizer.pad_token_id]
    #labels = labels[labels != tokenizer.eos_token]
    #labels = labels[labels != tokenizer.encode("<media>")[-1]]
    predicted = logits(predicted)[:,-labels.shape[1]:,:]
#     print(tokenizer.batch_decode(sampler_nuke(predicted)),'-',tokenizer.batch_decode(labels))
    loss_fct = nn.CrossEntropyLoss()
    loss = 0
    count =0
    for i in range(len(predicted)-1):
        losses = loss_fct(
            predicted[i], labels[i]
        )
        if losses > 0.0:
            loss+=losses
            count+=1
        else:
            print("error")
    if count > 0:
        loss=loss/count
        return loss
    else:
        return 100.0


def tokenize(tokenizer,text):
    tokenizer.padding_side = "right"
    text =  tokenizer(
        text,
        max_length=512,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )   
    return  text['input_ids'], text['attention_mask']

def train(args, model, rank, world_size, train_loader, optimizer, epoch, logits, pad_token, log, tokenizer, sampler=None):
    """
    Perform the training loop for one epoch.

    Args:
        args: Parsed command-line arguments.
        model: The Flamcon model.
        rank (int): Process rank.
        world_size (int): Total number of processes.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for gradient descent.
        epoch (int): Current epoch.
        logits (Callable): Function to compute logits.
        sampler: Optional DistributedSampler for data shuffling.

    Returns:
        None
    """
    model.train()
    ddp_loss = zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (media, X, y, file) in enumerate(tqdm(train_loader,position=1, desc="Iters", leave=False, colour='green', ncols=40)):
        optimizer.zero_grad()
        input_ids, attention_mask = tokenize(tokenizer,X)
        input_ids_test,_ = tokenize(tokenizer,y)
        input_ids = input_ids.to(rank)
        input_ids_test = input_ids_test.to(rank)
        attention_mask = attention_mask.to(rank)
        media = media.to(rank)
        batch_size, seq_length = input_ids.shape
        attention_mask = _prepare_attn_mask(attention_mask,
                                                     input_shape=(batch_size, seq_length),
                                                     past_key_values_length=0)
            
        if args.video:
            text_tokens = model(input_ids, videos=media, attention_mask=attention_mask)
        else:
            text_tokens = model(input_ids, images=media, attention_mask=attention_mask)
        
        loss = getLoss(text_tokens, input_ids_test, logits, tokenizer)
        if loss != 100.0:
            loss.backward()
            model.clip_grad_norm_(1.0)
            optimizer.step()
            
            if loss.item()> 0.0:
                ddp_loss[0] += loss.item()
                ddp_loss[1] += len(media)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        log.info('Train Epoch: {} \tLoss: {:.6f}\n'.format(epoch, ddp_loss[0] / ddp_loss[1]))

def test(args, model, rank, world_size, test_loader, optimizer, epoch, logits, pad_token, log, tokenizer, sampler=None):
    """
    Perform the training loop for one epoch.

    Args:
        args: Parsed command-line arguments.
        model: The Flamcon model.
        rank (int): Process rank.
        world_size (int): Total number of processes.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for gradient descent.
        epoch (int): Current epoch.
        logits (Callable): Function to compute logits.
        sampler: Optional DistributedSampler for data shuffling.

    Returns:
        None
    """
    model.eval()
    ddp_loss = zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (media, X, y, file) in enumerate(tqdm(test_loader,position=1, desc="Iters", leave=False, colour='green', ncols=40)):
        input_ids, attention_mask = tokenize(tokenizer,X)
        input_ids_test,_ = tokenize(tokenizer,y)
        input_ids = input_ids.to(rank)
        input_ids_test = input_ids_test.to(rank)
        attention_mask = attention_mask.to(rank)
        media = media.to(rank)
        batch_size, seq_length = input_ids.shape
        attention_mask = _prepare_attn_mask(attention_mask,
                                                 input_shape=(batch_size, seq_length),
                                                 past_key_values_length=0)
        
        if args.video:
            text_tokens = model.test(input_ids, videos=media, attention_mask=attention_mask)
        else:
            text_tokens = model.test(input_ids, images=media, attention_mask=attention_mask)
        loss = getLoss(text_tokens, input_ids_test, logits, tokenizer)
        
        if loss.item()> 0.0:
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(media)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        log.info('Test Epoch: {} \tLoss: {:.6f}\n'.format(epoch, ddp_loss[0] / ddp_loss[1]))
        
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
    parser.add_argument("--batch", default=4, type=int)
    parser.add_argument("--dim", default=4544, type=int)
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
           
    resume_from_epoch = 1
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
                    sync_module_states=True,  # broadcast loaded ckpt from rank 0 -> all ranks
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
    
    #Include specific params to optimizer
    paramsToOptimize = list(
            filter(
                lambda x: x[1].requires_grad
                and not getattr(x[1], "exclude_from_optimizer", False),
                model.named_parameters(),
            )
    )

    #Loading optimizers based on offload
    if args.rank == 0:
        print("Loading Optimizer\n")
        
    if args.cpu_offload:
        optimizer = DeepSpeedCPUAdam
    else:
        optimizer = FusedAdam
        
    optim = optimizer(
        (p for _, p in paramsToOptimize),
        lr=1e-4,
        weight_decay=0,
    )
    
    # load optimizer checkpoint
    if args.resume:
        if checkpoint is not None:
            print('Loading Optim CheckPoint\n')
            osd = checkpoint["optimizer_state_dict"]
            if not args.cpu_offload:
                osd = FSDP.optim_state_dict_to_load(osd, model, optimizer)
            optim.load_state_dict(osd)
            del osd
    
    #Load data based on batch size
    if args.rank == 0:
        print("Loading Data with batch: "+str(args.batch)+"\n")
    
    #For quick validation use generated data.
    #data = RandomVideos(length=args.batch,frames=args.max_frames)

    trainData = WebVidDataset("train_nw.csv","data",args.max_frames,tokenizer,args.max_tokens,samples=16800) #100 videos
    testData = WebVidDataset("test_nw.csv","data",args.max_frames,tokenizer,args.max_tokens,test=True,samples=2000) #20 videos

    sampler1 = DistributedSampler(trainData, rank=args.rank, num_replicas=args.world_size, shuffle=True)
    sampler2 = DistributedSampler(testData, rank=args.rank, num_replicas=args.world_size, shuffle=True)
    train_dataloader = DataLoader(trainData, batch_size=args.batch, sampler=sampler1)
    test_dataloader = DataLoader(testData, batch_size=args.batch, sampler=sampler2)
    scheduler = StepLR(optim, step_size=20, gamma=0.1)
    
    if args.resume:
        if checkpoint is not None:
            print('Loading Schedular CheckPoint\n')
            scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            
    #Start training
    if args.rank == 0:
        print("Starting Training\n")
        
    if checkpoint is not None:
        del checkpoint
    
    #Set Pad Token
    pad_token = tokenizer.encode("<PAD>")[-1]
    
    
    #Init Logger
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    log.addHandler(TqdmLoggingHandler())
    
    args.last_saved_epoch = 0
    for epoch in tqdm(range(resume_from_epoch, args.epochs),position=0, desc="Epochs", leave=False, colour='green', ncols=40):
        train(args, model, args.rank, args.world_size, train_dataloader, optim, epoch, to_logits, pad_token, log, tokenizer, sampler=sampler1)
        scheduler.step()
        dist.barrier()
        if (epoch) % 1 == 0:
            test(args, model, args.rank, args.world_size, test_dataloader, optim, epoch, to_logits, pad_token, log, tokenizer, sampler=sampler2)
            save_checkpoint(model, optim, scheduler, epoch, args, log)
            args.last_saved_epoch = (epoch)
    save_checkpoint(model, optim, scheduler, epoch, args, log)
if __name__ == "__main__":
    main()
