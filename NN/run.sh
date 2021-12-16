CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model_type lr \
    --loss_fn softmax \
    --input_file ../data/5year.arff \
    --max_epoches 30 \
    --batch_size_per_gpu 64 \
    --optim adam \
    --learning_rate 1e-3 \
    --weight_decay 1e-5
