for phase  in "prefill" "decode"
do 
    for batch_size in 32 64 128
    do
        
        for seq_len in 256 512 1024 2048 
        do
            if [ "$phase" == "prefill" ];then 
            
                echo "===========================================" | tee -a cpu-prefill-time.log
                python test-torchTensor.py --device cpu --prefill --seq-len $seq_len --batch-size $batch_size  2>&1   | tee -a cpu-prefill-time.log

                echo "===========================================" | tee -a gpu-prefill-time.log
                python test-torchTensor.py --device gpu --prefill --seq-len $seq_len --batch-size $batch_size  2>&1   | tee -a gpu-prefill-time.log
            fi

            if [ "$phase" == "decode" ];then
            
                echo "===========================================" | tee -a cpu-decode-time.log
                python test-torchTensor.py --device cpu   --seq-len $seq_len --batch-size $batch_size  2>&1   | tee -a cpu-decode-time.log

                echo "===========================================" | tee -a gpu-decode-time.log
                python test-torchTensor.py --device gpu   --seq-len $seq_len --batch-size $batch_size  2>&1   | tee -a gpu-decode-time.log
            fi
        done
    
    done
done     
