CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model_type svm \
    --sample downsample \
    --sample_rate 0.5 \
    --imputer none \
    --loss_fn sigmoid \
    --input_file ../data/5year.arff \
    --max_epoches 30 \
    --batch_size_per_gpu 16 \
    --optim adam \
    --learning_rate 1e-3 \
    --weight_decay 1e-5
