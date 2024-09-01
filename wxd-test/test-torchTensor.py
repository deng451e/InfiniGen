import torch 
import os 
import time 
import argparse
import numpy as np
import psutil 


 
path= os. getcwd() + "/../speedup/flexgen"
original=f"{path}/original"
flexgen=f"{path}/flexgen"
cmd = f"ln -sf {original}/flex_opt.py {path}/flexgen/flex_opt.py"
os.system(cmd) 
cmd = f"ln -sf {original}/pytorch_backend.py {path}/flexgen/pytorch_backend.py"
os.system(cmd)  

from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import, TorchTensor)
from flexgen.compression import CompressionConfig
from utils import add_parser_arguments, weight_init




 
 


##########################
# hyper parameter setting 
 

parser = argparse.ArgumentParser()
add_parser_arguments(parser)
args = parser.parse_args()

arch_name = args.arch_name 
device = args.device
prefill = args.prefill
warmup = args.warmup
repeat = args.repeat
attn_sparsity=args.attn_sparsity
donate = [False] * 14
batch_size=args.batch_size
seq_len=args.seq_len
##########################

if arch_name == "opt-1.3b":
 
    max_seq_len=2048;num_hidden_layers=24; n_head=32
    hidden_size=2048; input_dim=2048; ffn_embed_dim=2048 * 4

elif arch_name == "opt-2.7b":
  
    max_seq_len=2048; num_hidden_layers=32; n_head=32
    hidden_size=2560; input_dim=2560; ffn_embed_dim=2560 * 4
    
elif arch_name == "opt-6.7b":

    max_seq_len=2048; num_hidden_layers=32; n_head=32
    hidden_size=4096; input_dim=4096; ffn_embed_dim=4096 * 4

elif arch_name == "opt-13b":

    max_seq_len=2048; num_hidden_layers=40; n_head=40
    hidden_size=5120; input_dim=5120; ffn_embed_dim=5120 * 4

fix_recursive_import()
compute_device = None 
if device=="cpu":
    compute_device = TorchDevice("cpu")
    warmup = warmup//10 if warmup>=10 else warmup 
    repeat = repeat//10 if repeat>=10 else repeat 
elif device=="gpu":
    device = "cuda:0"
    compute_device = TorchDevice("cuda:0")


##########################
# set up the weight, input, kv cache 
 
 
  
 
mem_usages = list()
if prefill: # prefill 
 
    def test():
        if "cuda" in device:
            #mem_usage = torch.cuda.memory_allocated(0)/1024**3
            torch.cuda.reset_peak_memory_stats( ) 
            
        else:
            mem_usage = psutil.virtual_memory().used/1024**3
        input     = torch.randn((batch_size,seq_len,hidden_size),dtype=torch.float32,device=device)
        h         = TorchTensor((batch_size,seq_len,hidden_size),torch.float32, input, compute_device)
            
        mask_ = torch.randn((batch_size,seq_len))
        mask  = torch.where(mask_>0.5,True,False  ).to(device)
        
        mask = TorchTensor((batch_size,seq_len),torch.bool, mask, compute_device)
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_init(hidden_size,device,compute_device)
        output , k, v =  compute_device.mha(h, mask, w_q, b_q,
            w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
            False,CompressionConfig)
        if "cuda" in device:
            torch.cuda.synchronize()
            #mem_usages.append(torch.cuda.memory_allocated(0)/1024**3 - mem_usage)
            mem_usages.append(torch.cuda.max_memory_allocated( )/1024**3)
            
        else:
            
            mem_usages.append(psutil.virtual_memory().used/1024**3 - mem_usage)


        h.delete(); mask.delete(); w_q.delete(); b_q.delete(); w_k.delete(); b_k.delete(); w_v.delete(); b_v.delete(); w_out.delete(); b_out.delete(); w_ln.delete(); b_ln.delete()
        output.delete();k.delete;v.delete
else: # decode
    
    def test():
        if "cuda" in device:
            torch.cuda. reset_peak_memory_stats( )
           
        else:
            mem_usage = psutil.virtual_memory().used/1024**3
        input     = torch.randn((batch_size,1,hidden_size),dtype=torch.float32,device=device)
        h         = TorchTensor((batch_size,1,hidden_size),torch.float32, input, compute_device)
        # shape: (s, b * n_head, head_dim)
        cache_k     = torch.randn((seq_len,batch_size*n_head,hidden_size//n_head),dtype=torch.float32,device=device)
        cache_v     = torch.randn((seq_len,batch_size*n_head,hidden_size//n_head),dtype=torch.float32,device=device)
        k_cache   = TorchTensor((seq_len,batch_size*n_head,hidden_size//n_head),torch.float32, cache_k, compute_device)
        v_cache   = TorchTensor((seq_len,batch_size*n_head,hidden_size//n_head),torch.float32, cache_v, compute_device)

        mask_ = torch.randn((batch_size,seq_len))
        mask  = torch.where(mask_>0.5,True,False ).to(device)
        mask = TorchTensor((batch_size,seq_len),torch.bool, mask, compute_device)
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_init(hidden_size,device,compute_device)
        output , k, v =  compute_device.mha_gen(h, mask, w_q,
            b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
            k_cache, v_cache, donate,attn_sparsity,
            False,CompressionConfig)
        if "cuda" in device:
            torch.cuda.synchronize()
            mem_usages.append(torch.cuda.max_memory_allocated( )/1024**3 )
        else:
            
            mem_usages.append(psutil.virtual_memory().used/1024**3 - mem_usage)
        
        h.delete(); mask.delete(); w_q.delete(); b_q.delete(); w_k.delete(); b_k.delete(); w_v.delete(); b_v.delete(); w_out.delete(); b_out.delete(); w_ln.delete(); b_ln.delete()
        output.delete();k.delete;v.delete
    
 

##########################

 
        
 
 

for _ in range(warmup):
    test()
        
      
start_time = time.perf_counter()
 
for _ in range(repeat):
    test()
        
end_time = time.perf_counter()

if prefill:
    print(f"memory usage: {(np.mean(mem_usages)):.3} GB")

else:
    print(f"memory usage: {(np.mean(mem_usages)):.3} GB ")

print(f"self attentin time: {(end_time-start_time)/float(repeat):.3}")