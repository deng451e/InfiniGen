import os 
import sys 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import re
from collections import defaultdict
from IPython.display import display
import time 

 

def read(filaname):
    start_flag =  "=========================================== "
    
    
    df =  pd.DataFrame({'arch_name':[],'batch size':[],' seq_len':[],'memory usage':[],'self attentin time':[] })
    
    new_row = {}
     
    with open(filaname, 'r') as lines:
        
        for line in lines:
            if start_flag in line:
                if new_row: 
               
                    df.loc[len(df)] = new_row
                    
                    
                new_row = {}
                continue 
            if "CUDA out of memory" in line:
                new_row['memory usage'] = "OOM"
                new_row['self attentin time'] = "OOM"
            if ":" in line:
                line = line.strip()
                
                

                pairs = line.split(',')
                for pair in pairs:
                    
                    key,value = pair.split(":")
                     
                     
                    if "GB" in value: 
                       
                        value = value[:-3]
                   
                    try:
                        value = float(value[1:] )
                    except:
                        1
                    new_row[key] = value 
            
    
    return df 
path=f"{os.getcwd()}/results"

logs = {}
for device_type in "cpu","gpu":
    for phase_type in "prefill","decode":
        logs[f"{device_type}-{phase_type}"] = read(f"{path}/{device_type}-{phase_type}-time.log")

logs["gpu-prefill"]
host_name = os.getcwd().split("/")[2]
   
for phase_type in "prefill","decode":
    
    fig, ax = plt.subplots(figsize=(15, 6))
    x = [256,512,1024]
    for arch_name in [' opt-1.3b',' opt-2.7b',' opt-6.7b']:
        for batch_size in [32,64]:
            y = []
            for seq_len in [256,512,1024]:
                
            
                df_cpu = logs[f"cpu-{phase_type}"]
                v_cpu  = df_cpu[(df_cpu['arch_name']==arch_name) & (df_cpu['batch size']==batch_size)  & (df_cpu[' seq_len']==seq_len)]['self attentin time']

                df_gpu = logs[f"gpu-{phase_type}"]
                v_gpu  = df_gpu[(df_gpu['arch_name']==arch_name) & (df_gpu['batch size']==batch_size)  & (df_gpu[' seq_len']==seq_len)]['self attentin time']
                
                y.append(float(v_cpu.iloc[0]/v_gpu.iloc[0]))
                # print(x)
            
            
            plt.plot(x,y,label=f"arch_name:{arch_name}, batch_size:{batch_size}")
    plt.title(f"{phase_type}-latency-ratio(cpu/gpu)")
    plt.xticks(x)
    plt.legend(loc='upper left')
    plt.ylabel("lantency ratio")
    plt.xlabel("sequence length")
    plt.savefig(f"figures/{host_name}-{phase_type}-latency-ratio(cpu-gpu).pdf")
    plt.show()

   
for phase_type in "prefill","decode":
    
    fig, ax = plt.subplots(figsize=(15, 6))
    x = [256,512,1024]
    for arch_name in [' opt-1.3b',' opt-2.7b',' opt-6.7b']:
        for batch_size in [32,64]:
            y = []
            for seq_len in [256,512,1024]:
                
            
                df_cpu = logs[f"cpu-{phase_type}"]
                v_cpu  = df_cpu[(df_cpu['arch_name']==arch_name) & (df_cpu['batch size']==batch_size)  & (df_cpu[' seq_len']==seq_len)]['memory usage']

                df_gpu = logs[f"gpu-{phase_type}"]
                v_gpu  = df_gpu[(df_gpu['arch_name']==arch_name) & (df_gpu['batch size']==batch_size)  & (df_gpu[' seq_len']==seq_len)]['memory usage']
                
                y.append(float(v_gpu.iloc[0]/v_cpu.iloc[0]))
                # print(x)
            
            plt.plot(x,y,label=f"arch_name:{arch_name}, batch_size:{batch_size}")
           
    plt.title(f"{phase_type}-memory-ratio (gpu/cpu)")
    plt.xticks(x)
    plt.legend(loc='upper left')
    plt.ylabel("memory ratio")
    plt.xlabel("sequence length")
    plt.savefig(f"figures/{host_name}-{phase_type}-memory-ratio(cpu-gpu).pdf")
    plt.show()


start_flag =  "=========================================== "
df =  pd.DataFrame({'prompt len':[],' gen_len':[],' gpu ratio':[],' cpu ratio':[],'Total':[],' Prefill':[],' Decode':[] })

new_row = {}
    
with open(f"{path}/flexgen-offload.log", 'r') as lines:
    
    for line in lines:
        if start_flag in line:
            if new_row: df.loc[len(df)] = new_row                
            new_row = {}
            continue 
        if "CUDA out of memory" in line:
            new_row['Total'] = "OOM"
            new_row['Prefill'] = "OOM"
            new_row['Decode'] = "OOM"
             
         
        pairs = line.split(',')

        for pair in pairs:
            if ":" in pair:
                key,value = pair.split(":")
                new_row[key] = float(value)  
display(df)


fig, ax = plt.subplots(figsize=(10, 6))
x = [20,40,60,80,100]
for prompt_len in [512,1024]:
    for gen_len in [128,256,512,1024]:
        y = []
       
        for ratio in [20,40,60,80,100]:
            v = df[(df[' gpu ratio']==ratio) & (df['prompt len']==prompt_len)  & (df[' gen_len']==gen_len)]['Total']
             
            y.append(float(v))
        # print(x)
        plt.plot(x,y,label=f" prompt_len:{prompt_len}, gen_len:{gen_len}")
plt.xticks(x)
plt.legend(loc='upper right')
plt.ylabel("time (sec)")
plt.xlabel("GPU Ratio")
plt.savefig(f"figures/{host_name}-flexgen-offload-total-time.pdf")

plt.show()



fig, ax = plt.subplots(figsize=(10, 6))
x = [20,40,60,80,100]
for prompt_len in [512,1024]:
    for gen_len in [128,256,512,1024]:
        y = []
       
        for ratio in [20,40,60,80,100]:
            v = df[(df[' gpu ratio']==ratio) & (df['prompt len']==prompt_len)  & (df[' gen_len']==gen_len)][' Prefill']
             
            y.append(float(v))
        # print(x)
        plt.plot(x,y,label=f" prompt_len:{prompt_len}, gen_len:{gen_len}")
plt.xticks(x)
plt.legend(loc='upper right')
plt.ylabel("time (sec)")
plt.xlabel("GPU Ratio")
plt.savefig(f"figures/{host_name}-flexgen-offload-prefill-time.pdf")

plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
x = [20,40,60,80,100]
for prompt_len in [512,1024]:
    for gen_len in [128,256,512,1024]:
        y = []
       
        for ratio in [20,40,60,80,100]:
            v = df[(df[' gpu ratio']==ratio) & (df['prompt len']==prompt_len)  & (df[' gen_len']==gen_len)][' Decode']
             
            y.append(float(v))
        plt.plot(x,y,label=f" prompt_len:{prompt_len}, gen_len:{gen_len}")
plt.xticks(x)
plt.legend(loc='upper right')
plt.ylabel("time (sec)")
plt.xlabel("GPU Ratio")
plt.savefig(f"figures/{host_name}-flexgen-offload-decode-time.pdf")

plt.show()