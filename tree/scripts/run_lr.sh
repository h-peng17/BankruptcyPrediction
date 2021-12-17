for year in 1 2 3 4 5; do
    for imputer in simple knn; do
        for sample in none upsample downsample smote; do
            if [[ $sample == "upsample" ]]; then
                sample_rate=10
            elif [[ $sample == "downsample" || $sample == "smote" ]]; then
                sample_rate=0.1
            else
                sample_rate=1
            fi
            python main.py \
                --data_path ./data/${year}year.arff \
                --model lr \
                --do_cross_val \
                --do_test \
                --output_path log/year${year}/lr/imputer=${imputer}_sample=${sample}_samplerate=${sample_rate} \
                --imputer $imputer \
                --sample $sample \
                --sample_rate $sample_rate \
                --n_jobs 16
        done
    done
done