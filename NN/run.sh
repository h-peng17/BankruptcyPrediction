# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --model_type mlp \
#     --sample none \
#     --sample_rate 0.2 \
#     --imputer simple \
#     --loss_fn sigmoid \
#     --input_file ../data/5year.arff \
#     --max_epoches 30 \
#     --batch_size_per_gpu 16 \
#     --optim adam \
#     --learning_rate 1e-3 \
#     --weight_decay 1e-5

# for year in 1 2 3 4 5; do
#     # for max_depth in None; do
#         # for min_samples_split in 2 4 8; do
#             for max_features in None; do
#                 for imputer in simple knn; do
#                     for sample in none upsample downsample smote; do
#                         if [[ $sample = "upsample" ]]; then
#                             sample_rate=10
#                         elif [[ $sample = "downsample" ]]; then
#                             sample_rate=0.1
#                         elif [[ $sample = "smotesample" ]]; then
#                             sample_rate=0.1
#                         else
#                             sample_rate=1
#                         fi
#                         CUDA_VISIBLE_DEVICES=$1 python main.py \
#                             --model_type svm \
#                             --sample $sample \
#                             --sample_rate $sample_rate \
#                             --imputer $imputer \
#                             --loss_fn sigmoid \
#                             --input_file ../data/${year}year.arff \
#                             --max_epoches 30 \
#                             --batch_size_per_gpu 16 \
#                             --optim adam \
#                             --learning_rate 1e-3 \
#                             --weight_decay 1e-5
#                     done
#                 done
#             done
#         # done
#     # done
# done


for year in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --model_type lr \
        --imputer simple \
        --sample none \
        --sample_rate 0.1 \
        --loss_fn sigmoid \
        --input_file ../data/${year}year.arff \
        --max_epoches 30 \
        --batch_size_per_gpu 16 \
        --optim adam \
        --learning_rate 1e-3 \
        --weight_decay 1e-5
done
