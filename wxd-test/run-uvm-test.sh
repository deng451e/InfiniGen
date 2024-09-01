UVM_PATH=$PWD/../speedup/uvm
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
g++ $UVM_PATH/allocate.cpp -o allocate.so --shared -fPIC -I$CUDA_HOME/include
for  arch_name in  "opt-1.3b" "opt-2.7b" "opt-6.7b"
do
    for SCHEME in "uvm" "uvm_h2o"
    do

         

        if [ "$arch_name" == "opt-1.3b" ];then
            max_seq_len=2048;num_hidden_layers=24; n_head=32
            hidden_size=2048; input_dim=2048; ffn_embed_dim=8192
        fi

        if [ "$arch_name" == "opt-2.7b" ];then
            max_seq_len=2048; num_hidden_layers=32; n_head=32
            hidden_size=2560; input_dim=2560; ffn_embed_dim=10240
        fi 


        if [ "$arch_name" == "opt-6.7b" ];then
            max_seq_len=2048; num_hidden_layers=32; n_head=32
            hidden_size=4096; input_dim=4096; ffn_embed_dim=16360
        fi 

        CMD="--embed_dim $hidden_size --ffn_dim $ffn_embed_dim --enable_bias --n_head $n_head --do_layer_norm_before --n_layer $num_hidden_layers --bsz 20 --prompt_len 1024 --gen_len 128 --runs 1"
        
        if [ "$SCHEME" == "uvm_h2o" ]
        then 
            CMD=$CMD" --is_h2o --h2o_ratio 0.2"
        fi
        
        outpt="======================================================== "$'\n'
        outpt+="${CMD}"$'\n'
        out=$(python $UVM_PATH/transformer.py $CMD  2>&1 )
        outpt+="${out}"$'\n'

        rm  ${arch_name}-${SCHEME}-output.log
        echo  "$outpt" | tee -a ${arch_name}-${SCHEME}-output.log
    
    done
done 
rm allocate.so


 