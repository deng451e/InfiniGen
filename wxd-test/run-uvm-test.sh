UVM_PATH=$PWD/../speedup/uvm
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
for SCHEME in "uvm" "uvm_h2o"
do
  g++ $UVM_PATH/allocate.cpp -o allocate.so --shared -fPIC -I$CUDA_HOME/include
  CMD="--embed_dim 5120 --ffn_dim 20480 --enable_bias --n_head 40 --do_layer_norm_before --n_layer 40 --bsz 20 --prompt_len 1920 --gen_len 128 --runs 1"
  
  if [ "$SCHEME" = "uvm_h2o" ]
  then 
    CMD=$CMD" --is_h2o --h2o_ratio 0.2"
  fi
  python $UVM_PATH/transformer.py $CMD
  rm allocate.so
done
