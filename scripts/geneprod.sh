small_model_list=("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
large_model_list=("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")

masking_probability=(0.00 0.01 0.02 0.03 0.04 0.05 0.10 0.15 0.20 0.30 0.50 1.00)

for i in ${!small_model_list[@]}; do
    for j in ${!masking_probability[@]}; do
        python -m soda_model.token_classification.trainer \
            --dataset_id "EMBO/SourceData" \
            --task ROLES_GP \
            --version 1.0.0 \
            --from_pretrained ${small_model_list[$i]} \
            --masking_probability ${masking_probability[$j]} \
            --replacement_probability 0.0 \
            --per_device_train_batch_size 16 \
            --add_prefix_space \
            --num_train_epochs 2.0 \
            --learning_rate 0.0001 \
            --disable_tqdm False \
            --report_to none \
            --classifier_dropout 0.2 \
            --do_train \
            --do_eval \
            --do_predict \
            --truncation \
            --padding "longest" \
            --ner_labels all \
            --results_file "geneprod_roles_for_gen_vs_memo_base_pmb_"
    done
done

for i in ${!large_model_list[@]}; do
    for j in ${!masking_probability[@]}; do
        python -m soda_model.token_classification.trainer \
            --dataset_id "EMBO/SourceData" \
            --task ROLES_GP \
            --version 1.0.0 \
            --from_pretrained ${large_model_list[$i]} \
            --masking_probability ${masking_probability[$j]} \
            --replacement_probability 0.0 \
            --per_device_train_batch_size 8 \
            --add_prefix_space \
            --num_train_epochs 1.0 \
            --learning_rate 0.00005 \
            --disable_tqdm False \
            --report_to none \
            --classifier_dropout 0.2 \
            --do_train \
            --do_eval \
            --do_predict \
            --truncation \
            --padding "longest" \
            --ner_labels all \
            --results_file "geneprod_roles_for_gen_vs_memo_large_pmb_"
    done
done
