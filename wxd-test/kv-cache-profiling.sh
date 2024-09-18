 
FLEXGEN_PATH=$PWD/../speedup/flexgen
rm $FLEXGEN_PATH/flexgen/flex_opt.py
rm $FLEXGEN_PATH/flexgen/pytorch_backend.py
ln -sf  $PWD/flexgen.profile/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
ln -sf  $PWD/flexgen.profile/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py

# "the percentage of weight on GPU, "
# "the percentage of weight on CPU, "
# "the percentage of attention cache on GPU, "
# "the percentage of attention cache on CPU, "
# "the percentage of activations on GPU, "
# "the percentage of activations on CPU"
for dataset_name in   "wikitext-103-raw-v1" "wikitext-103-v1"  "wikitext-2-raw-v1"
do
    for  arch_name in   "opt-1.3b" "opt-2.7b"  "opt-6.7b"
    do 
        for prompt_len in  128 256 512 1024  
        do 
            for gen_len in    128 256 512 1024  
            do
                
                 
                    outpt="======================================================== "$'\n'
                    outpt+="prompt len: ${prompt_len}, gen_len: ${gen_len}, "$'\n'
                    
                    CMD="--model huggingface/${arch_name} --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 1 --num-gpu-batches 1 --prompt-len ${prompt_len} --gen-len ${gen_len} --warmup-input-path pg19_firstbook.txt --test-input-path $HOME/input/wikitext.txt/${dataset_name}/test-00000-of-00001.txt"
                    
                    python -m flexgen.flex_opt  $CMD 
            
                    echo "$outpt"  
                
            done
        done
    done
done 
    


 