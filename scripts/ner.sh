small_model_list=("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" "michiyasunaga/BioLinkBERT-base")
large_model_list=("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract" "michiyasunaga/BioLinkBERT-large")

model_name_list=("PubMedBERT" "BioLinkBERT")

masking_probability=(0.0)

version="1.0.0"
for i in ${!small_model_list[@]}; do
    for j in ${!masking_probability[@]}; do
        python -m soda_model.token_classification.trainer \
            --dataset_id "EMBO/SourceData" \
            --task NER \
            --version ${version} \
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
            --do_train \
            --do_predict \
            --truncation \
            --padding "longest" \
            --ner_labels all \
            --results_file "ner_${model_name_list[$i]}_base_" \
            --paper_report 
    done
done

for i in ${!large_model_list[@]}; do
    for j in ${!masking_probability[@]}; do
        python -m soda_model.token_classification.trainer \
            --dataset_id "EMBO/SourceData" \
            --task NER \
            --version ${version} \
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
            --do_train \
            --do_predict \
            --truncation \
            --padding "longest" \
            --ner_labels all \
            --results_file "ner_${model_name_list[$i]}_large_" \
            --paper_report 
    done
done
