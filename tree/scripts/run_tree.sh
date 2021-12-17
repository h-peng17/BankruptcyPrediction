for year in 1 2 3 4 5; do
    criterion=entropy
    max_depth=None
    min_samples_split=2
    max_features=None
    python main.py \
        --data_path ./data/${year}year.arff \
        --model tree \
        --criterion $criterion \
        --do_test \
        --max_depth $max_depth \
        --min_samples_split $min_samples_split \
        --max_features $max_features \
        --output_path log/year${year}/tree/${criterion}_depth${max_depth}_split${min_samples_split}_features${max_features}
done