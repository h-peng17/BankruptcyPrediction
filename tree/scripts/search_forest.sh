for year in 1 2 3 4 5; do
    for criterion in gini entropy; do
        for max_depth in None; do
            for min_samples_split in 2 4 8; do
                for max_features in None; do
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
                                --model forest \
                                --criterion $criterion \
                                --do_cross_val \
                                --do_test \
                                --max_depth $max_depth \
                                --min_samples_split $min_samples_split \
                                --max_features $max_features \
                                --output_path log/year${year}/forest/${criterion}_depth=${max_depth}_split=${min_samples_split}_features=${max_features}_imputer=${imputer}_sample=${sample}_samplerate=${sample_rate} \
                                --imputer $imputer \
                                --sample $sample \
                                --sample_rate $sample_rate \
                                --n_jobs 32
                        done
                    done
                done
            done
        done
    done
done