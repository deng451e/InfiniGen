UVM_PATH=$PWD/../../uvm
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
# for SCHEME in "uvm" "uvm_h2o"
# do
#   g++ $UVM_PATH/allocate.cpp -o allocate.so --shared -fPIC -I$CUDA_HOME/include
#   CMD="--embed_dim 5120 --ffn_dim 20480 --enable_bias --n_head 40 --do_layer_norm_before --n_layer 40 --bsz 20 --prompt_len 1920 --gen_len 128 --runs 1"
  
#   if [ "$SCHEME" = "uvm_h2o" ]
#   then 
#     CMD=$CMD" --is_h2o --h2o_ratio 0.2"
#   fi
#   python $UVM_PATH/transformer.py $CMD
#   rm allocate.so
# done


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

###########
#kv cache on cpu
 
# rm cpu-offload.log
# for SCHEME in "original"   "infinigen"  "h2o" "int4" 
# do 
#   echo "=========================${SCHEME}==============================" | tee -a cpu-offload.log
#   for prompt_len in 512 1024  
#   do
#       for gen_len in 128 256 512 1024
#         do
#         if [ "$SCHEME" = "int4" ]
#         then

#           ln -sf ../original/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
#           ln -sf ../original/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py
#         else
#           ln -sf ../$SCHEME/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
#           ln -sf ../$SCHEME/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py
#         fi

#         CMD="--model huggingface/opt-1.3b --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 20 --num-gpu-batches 1 --prompt-len ${prompt_len} --gen-len ${gen_len} --warmup-input-path pg19_firstbook.txt --test-input-path pg19_firstbook.txt"
#         if [ "$SCHEME" = "int4" ]
#         then
#           CMD=$CMD" --compress-cache"
#         elif [ "$SCHEME" = "h2o" ]
#         then
#           CMD=$CMD" --max-num-kv 415 --hh-ratio 0.1 --hh-all"
#         elif [ "$SCHEME" = "infinigen" ]
#         then
#           CMD=$CMD" --alpha 4 --partial-weight-ratio 0.2 --max-num-kv 400"
#         fi
        
        
#         echo  "prompt len: ${prompt_len} gen_len: ${gen_len}" | tee -a cpu-offload.log
#         python -m flexgen.flex_opt $CMD   2>&1 | grep "Total:"  | tee -a cpu-offload.log
#       done
#   done
# done



 
rm gpu-offload.log
 
for SCHEME in "original"   "infinigen"  "h2o" "int4" 
do 
  echo "=========================${SCHEME}==============================" | tee -a gpu-offload.log
  for prompt_len in 512 1024  
  do
      for gen_len in 128 256 512 1024
      do
        if [ "$SCHEME" = "int4" ]
        then

          ln -sf ../original/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
          ln -sf ../original/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py
        else
          ln -sf ../$SCHEME/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
          ln -sf ../$SCHEME/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py
        fi

        CMD="--model huggingface/opt-1.3b --percent 100 0 100 0 100 0 --overlap false --gpu-batch-size 20 --num-gpu-batches 1 --prompt-len ${prompt_len} --gen-len 256 --warmup-input-path pg19_firstbook.txt --test-input-path pg19_firstbook.txt"
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
      
        
        echo   "prompt len: ${prompt_len} gen_len: ${gen_len}" | tee -a gpu-offload.log
        python -m flexgen.flex_opt $CMD   2>&1 | grep "Total:"  | tee -a gpu-offload.log
      done
  done
done

 