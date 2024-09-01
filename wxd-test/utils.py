import argparse
import torch 
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import, TorchTensor)
def add_parser_arguments(parser):
    parser.add_argument("--arch_name", type=str, default="opt-1.3b") 
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--prefill", action="store_true") 
     
 
def weight_init(hidden_size,device,compute_device):

    w_q_ = torch.randn((hidden_size,hidden_size),dtype=torch.float32,device=device)
    w_q  = TorchTensor((hidden_size,hidden_size),torch.float32 ,w_q_ ,compute_device)
    b_q_ = torch.randn((hidden_size),dtype=torch.float32,device=device)
    b_q  = TorchTensor((hidden_size),torch.float32, b_q_, compute_device)


    
    w_k_ = torch.randn((hidden_size,hidden_size),dtype=torch.float32,device=device)
    w_k  = TorchTensor((hidden_size,hidden_size),torch.float32 ,w_k_ ,compute_device)
    b_k_ = torch.randn((hidden_size),dtype=torch.float32,device=device)
    b_k  = TorchTensor((hidden_size),torch.float32, b_k_, compute_device)


    w_v_ = torch.randn((hidden_size,hidden_size),dtype=torch.float32,device=device)
    w_v  = TorchTensor((hidden_size,hidden_size),torch.float32 ,w_v_ ,compute_device)
    b_v_ = torch.randn((hidden_size),dtype=torch.float32,device=device)
    b_v  = TorchTensor((hidden_size),torch.float32, b_v_, compute_device)

    w_out_ = torch.randn((hidden_size,hidden_size),dtype=torch.float32,device=device)
    w_out  = TorchTensor((hidden_size,hidden_size),torch.float32 ,w_out_ ,compute_device)
    b_out_ = torch.randn((hidden_size),dtype=torch.float32,device=device)
    b_out  = TorchTensor((hidden_size),torch.float32, b_out_, compute_device)
    
    
    w_ln_ = torch.randn((hidden_size),dtype=torch.float32,device=device)
    w_ln  = TorchTensor((hidden_size ),torch.float32 ,w_ln_ , compute_device)
    b_ln_ = torch.randn((hidden_size),dtype=torch.float32,device=device)
    b_ln = TorchTensor((hidden_size ),torch.float32, b_ln_, compute_device)
    return  w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

     