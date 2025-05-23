CUDA_VISIBLE_DEVICES=0 python main.py \
    --model ../cache/llm_weights/models--facebook--opt-125m/snapshots/opt-125m \
    --sparsity_ratio 0.5 \
    --nsamples 128 \
    --device cuda:0 \
    --cali_dataset c4 \
    --cali_data_path ../cache/data/c4 \
    --eval_dataset wikitext2 \
    --eval_data_path ../cache/data/wikitext \
    --output_dir out/opt-125m-d2prune-sp0.5/ \
    --s 1500 \
    --r1 1 \
    --r2 0 \
    --d2_wanda \
    --d2_sparsegpt \
    --target_layer_names "['self_attn.k_proj']"