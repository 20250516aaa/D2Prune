CUDA_VISIBLE_DEVICES=0 python main.py \
    --model ../cache/llm_weights/models--meta-llama--Llama-2-13b-hf/snapshots/llama-2-13b \
    --sparsity_ratio 0.5 \
    --prune_method d2prune \
    --sparsity_type unstructured \
    --cali_dataset c4 \
    --cali_data_path ../cache/data/c4 \
    --eval_dataset wikitext2 \
    --eval_data_path ../cache/data/wikitext \
    --output_dir out/llama-2-13b-d2prune-sp0.5/ \
    --s 1500 \
    --r1 1 \
    --r2 0 \
    --d2_wanda \
    --d2_sparsegpt \
    --target_layer_names "['self_attn.k_proj']"