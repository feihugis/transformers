export CUDA_VISIBLE_DEVICES=GPU-b930159a-a609-47c4-a508-a534e7da9016
export DATA_DIR=cnn_dm

./run_eval.py sshleifer/distilbart-cnn-12-6 $DATA_DIR/val.source dbart_val_generations.txt \
    --reference_path $DATA_DIR/val.target \
    --score_path cnn_rouge.json \
    --task summarization \
    --n_obs 100 \
    --device cuda \
    --fp16 \
    --bs 32

# {'rouge1': 35.8363, 'rouge2': 15.1514, 'rougeL': 25.6202, 'rougeLsum': 32.4937, 'n_obs': 100, 'runtime': 9, 'seconds_per_sample': 0.09}

./run_eval.py t5-base $DATA_DIR/val.source t5_val_generations.txt \
    --reference_path $DATA_DIR/val.target \
    --score_path cnn_rouge.json \
    --task summarization \
    --n_obs 100 \
    --device cuda \
    --fp16 \
    --bs 32
# {'rouge1': 34.5757, 'rouge2': 13.5853, 'rougeL': 24.5685, 'rougeLsum': 31.0061, 'n_obs': 100, 'runtime': 16, 'seconds_per_sample': 0.16}
