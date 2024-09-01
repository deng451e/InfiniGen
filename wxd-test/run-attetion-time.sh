 
 
 
rm cpu-prefill-time.log
rm gpu-prefill-time.log
rm  cpu-decode-time.log
rm  gpu-decode-time.log
for arch_name in "opt-1.3b" "opt-2.7b" "opt-6.7b"
do 
    for device in   "cpu" "gpu"
    do
        for phase  in "prefill" "decode"
        do 
            for batch_size in 32 64  
            do
                
                for seq_len in 256 512 1024   2048 
                do
                    
                    outpt="=========================================== "$'\n'
                    outpt+="arch_name: ${arch_name},batch size: ${batch_size}, seq_len: ${seq_len}"$'\n'
                    
                    if [ "$phase" == "prefill" ];then 
                        out=$(python test-torchTensor.py --arch_name $arch_name --device $device --prefill --seq-len $seq_len --batch-size $batch_size  2>&1 )
                    fi

                    if [ "$phase" == "decode" ];then 
                        out=$(python test-torchTensor.py  --arch_name $arch_name --device $device  --seq-len $seq_len --batch-size $batch_size  2>&1 )
                    fi


                    outpt+="${out}"$'\n'
                    echo  "$outpt" | tee -a ${device}-${phase}-time.log
                    
                done
            
            done
        done     
    done 
done 