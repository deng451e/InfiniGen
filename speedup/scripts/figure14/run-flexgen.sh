UVM_PATH=$PWD/../../uvm
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
 

# default=[100, 0, 100, 0, 100, 0],
#     help="Six numbers. They are "
#       "the percentage of weight on GPU, "
#       "the percentage of weight on CPU, "
#       "the percentage of attention cache on GPU, "
#       "the percentage of attention cache on CPU, "
#       "the percentage of activations on GPU, "
#       "the percentage of activations on CPU"


FLEXGEN_PATH=/home/c3/code/infinigen/speedup/flexgen

rm $FLEXGEN_PATH/flexgen/flex_opt.py
rm $FLEXGEN_PATH/flexgen/pytorch_backend.py
 
rm flexgen-offload.log
SCHEME="original" 
for prompt_len in 512 1024  
do 
  echo "=========================${SCHEME}==============================" | tee -a flexgen-offload.log
  
   
  for gen_len in 128 256 512 1024
  do
       
      for ratio in 0  20  40 60  80 100  
        do
            
            ln -sf ../$SCHEME/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
            ln -sf ../$SCHEME/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py
             

            CMD="--model huggingface/opt-1.3b --percent 100 0  $(expr 100 - $ratio) $ratio 100 0 --overlap false --gpu-batch-size 20 --num-gpu-batches 1 --prompt-len ${prompt_len} --gen-len ${gen_len} --warmup-input-path pg19_firstbook.txt --test-input-path pg19_firstbook.txt"
            if [ "$SCHEME" = "int4" ]
            then
            CMD=$CMD" --compress-cache"
            elif [ "$SCHEME" = "h2o" ]
            then
            CMD=$CMD" --max-num-kv 415 --hh-ratio 0.1 --hh-all"
            elif [ "$SCHEME" = "infinigen" ]
            then
            CMD=$CMD" --alpha 4 --partial-weight-ratio 0.2 --max-num-kv 400"
            fi
        
            echo  "prompt len: ${prompt_len} gen_len: ${gen_len} gpu ratio:  $(expr 100 - $ratio) cpu ratio: ${ratio}" | tee -a flexgen-offload.log
            python -m flexgen.flex_opt $CMD   2>&1 | grep "Total:"  | tee -a flexgen-offload.log
      done
  done
done


 