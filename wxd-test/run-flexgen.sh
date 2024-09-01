# "the percentage of weight on GPU, "
# "the percentage of weight on CPU, "
# "the percentage of attention cache on GPU, "
# "the percentage of attention cache on CPU, "
# "the percentage of activations on GPU, "
# "the percentage of activations on CPU"

 
FLEXGEN_PATH=$PWD/../speedup/flexgen
rm flexgen-offload.log
rm $FLEXGEN_PATH/flexgen/flex_opt.py
rm $FLEXGEN_PATH/flexgen/pytorch_backend.py
ln -sf  $FLEXGEN_PATH/original/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
ln -sf  $FLEXGEN_PATH/original/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py

for prompt_len in 512 1024  
do 
  for gen_len in 128 256 512 1024
  do
       
      for ratio in 0  20  40 60  80 100  
        do
          
             
            outpt="======================================================== "$'\n'
            outpt+="prompt len: ${prompt_len}, gen_len: ${gen_len}, gpu ratio:  $(expr 100 - $ratio), cpu ratio: ${ratio}"$'\n'
            
            CMD="--model huggingface/opt-1.3b --percent 100 0  $(expr 100 - $ratio) $ratio 100 0 --overlap false --gpu-batch-size 20 --num-gpu-batches 1 --prompt-len ${prompt_len} --gen-len ${gen_len} --warmup-input-path pg19_firstbook.txt --test-input-path   pg19_firstbook.txt"
          
            
            outpt+=$(python -m flexgen.flex_opt $CMD   2>&1 | grep "Total:")  
            echo "$outpt" | tee -a flexgen-offload.log
      done
  done
done


 