small_model_list=("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" "michiyasunaga/BioLinkBERT-base")
large_model_list=("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract" "michiyasunaga/BioLinkBERT-large")

model_name_list=("PubMedBERT" "BioLinkBERT")

small_model_list=("michiyasunaga/BioLinkBERT-base")
large_model_list=()

train_size=(0.05 0.10 0.25 0.35 0.50 0.75 1.00)

model_name_list=("BioLinkBERT")

# masking_probability=(0.0 0.025 0.05 0.125 0.25 0.375 0.5)
masking_probability=(0.0)

for i in ${!small_model_list[@]}; do
    for j in ${!masking_probability[@]}; do
        for k in ${!train_size[@]}; do
        python -m soda_model.token_classification.trainer \
            --dataset_id "EMBO/SourceData" \
            --task NER \
            --version 2.0.3 \
            --from_pretrained ${small_model_list[$i]} \
            --masking_probability ${masking_probability[$j]} \
            --replacement_probability 0.0 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 64 \
            --add_prefix_space \
            --num_train_epochs 2.0 \
            --learning_rate 0.0001 \
            --disable_tqdm False \
            --report_to none \
            --classifier_dropout 0.2 \
            --use_crf \
            --do_train \
            --do_eval \
            --do_predict \
            --truncation \
            --padding "longest" \
            --ner_labels all \
            --results_file "v2.0.3\ner_with_crf_${model_name_list[$i]}_base_maskingprob_${masking_probability[$j]}_training_size_${train_size[$k]}_"  \
            --train_set_perc ${train_size[$k]}
    done
done

for i in ${!large_model_list[@]}; do
    for j in ${!masking_probability[@]}; do
        for k in ${!train_size[@]}; do
            python -m soda_model.token_classification.trainer \
                --dataset_id "EMBO/SourceData" \
                --task NER \
                --version 2.0.3 \
                --from_pretrained ${large_model_list[$i]} \
                --masking_probability ${masking_probability[$j]} \
                --replacement_probability 0.0 \
                --gradient_accumulation_steps 2 \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 32 \
                --add_prefix_space \
                --num_train_epochs 2.0 \
                --learning_rate 0.00005 \
                --disable_tqdm False \
                --report_to none \
                --classifier_dropout 0.2 \
                --use_crf \
                --do_train \
                --do_eval \
                --do_predict \
                --truncation \
                --padding "longest" \
                --ner_labels all \
                --results_file "v2.0.3\ner_with_crf_${model_name_list[$i]}_large_maskingprob_${masking_probability[$j]}_training_size_${train_size[$k]}_" \
                --train_set_perc ${train_size[$k]}
    done
done

