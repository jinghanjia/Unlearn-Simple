CORPUS="books"

FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"

TARGET_DIR="muse-bench/MUSE-Books_target"
LLAMA_DIR="meta-llama/Llama-2-7b-hf"

MAX_LEN=2048
EPOCHS=10
LR='5e-6'
PER_DEVICE_BATCH_SIZE=4 # 8 GPUs
FT_EPOCHS=10
FT_LR='1e-5'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7




for algo in 'npo_gdr' 'ga_gdr'; do
    python unlearn.py \
        --algo $algo \
        --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
        --data_file $FORGET --retain_data_file $RETAIN \
        --out_dir "./ckpt/$CORPUS/$algo$LR" \
        --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE \

done


# for target_dir in '/egr/research-optml/jiajingh/muse_bench/baselines/ckpt/books/pretrained/tv_0.2' '/egr/research-optml/jiajingh/muse_bench/baselines/ckpt/books/pretrained/tv_0.4' '/egr/research-optml/jiajingh/muse_bench/baselines/ckpt/books/pretrained/tv_0.6' '/egr/research-optml/jiajingh/muse_bench/baselines/ckpt/books/pretrained/tv_0.8' '/egr/research-optml/jiajingh/muse_bench/baselines/ckpt/books/pretrained/tv_1.0'; do
#     basename=$(basename $target_dir)
#     python unlearn.py \
#         --algo npo_gdr \
#         --model_dir $target_dir --tokenizer_dir $LLAMA_DIR \
#         --data_file $FORGET --retain_data_file $RETAIN \
#         --out_dir "./ckpt/$CORPUS/$algo_$basename" \
#         --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
#         --per_device_batch_size $PER_DEVICE_BATCH_SIZE
# done


# for lr in '1e-4' '1e-5'; do
# for epoch in '30'; do
# for alpha in '1.0' '5.0' '10.0'; do
#     python unlearn.py \
#         --algo 'tv' \
#         --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
#         --data_file $FORGET --retain_data_file $RETAIN \
#         --out_dir "./ckpt/$CORPUS/tv$lr$epoch" \
#         --max_len $MAX_LEN --epochs $epoch --lr $lr \
#         --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
#         --alpha 5.0
#         done
#     done
# done
# python unlearn.py \
#     --algo 'tv' \
#     --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
#     --data_file $FORGET --retain_data_file $RETAIN \
#     --out_dir "./ckpt/$CORPUS/tv" \
#     --max_len $MAX_LEN --epochs $FT_EPOCHS --lr $FT_LR \
#     --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
#     --alpha 5.0

# for alpha in '0.1'; do
# python unlearn.py \
#     --algo 'tv' \
#     --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
#     --data_file $FORGET --retain_data_file $RETAIN \
#     --out_dir "./ckpt/$CORPUS/tv_${alpha}" \
#     --max_len $MAX_LEN --epochs $FT_EPOCHS --lr $FT_LR \
#     --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
#     --alpha $alpha \
#     --ft_model_dir /egr/research-optml/jiajingh/Unlearn-Simple/MUSE/baselines/ckpt/books/tv_0.1_ft
# done

# for target_dir in '/egr/research-optml/jiajingh/Unlearn-Simple/MUSE/baselines/ckpt/books/tv_0.1' '/egr/research-optml/jiajingh/Unlearn-Simple/MUSE/baselines/ckpt/books/tv_1.0' '/egr/research-optml/jiajingh/Unlearn-Simple/MUSE/baselines/ckpt/books/tv_10.0' '/egr/research-optml/jiajingh/Unlearn-Simple/MUSE/baselines/ckpt/books/tv_100'; do
#     basename=$(basename $target_dir)
#     for lr in '5e-6' '1e-5'; do
#     for algo in 'npo_gdr'; do
#         python unlearn.py \
#             --algo $algo \
#             --model_dir $target_dir --tokenizer_dir $LLAMA_DIR \
#             --data_file $FORGET --retain_data_file $RETAIN \
#             --out_dir "./ckpt/$CORPUS/$algo$basename$lr" \
#             --max_len $MAX_LEN --epochs $EPOCHS --lr $lr \
#             --per_device_batch_size $PER_DEVICE_BATCH_SIZE
#             done
#     done
# done