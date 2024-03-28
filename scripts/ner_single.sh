# small_model_list=("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" "michiyasunaga/BioLinkBERT-base")
small_model_list=("michiyasunaga/BioLinkBERT-base")
# large_model_list=("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract" "michiyasunaga/BioLinkBERT-large")
# large_model_list=("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")

# model_name_list=("PubMedBERT" "BioLinkBERT")
model_name_list=("BioLinkBERT")

masking_probability=(0.0 0.025 0.05 0.125 0.25 0.375 0.5)

# ner_labels=("DISEASE" "CELL_LINE" "CELL_TYPE" "GENEPROD" "SMALL_MOLECULE" "EXP_ASSAY" "TISSUE" "ORGANISM" "SUBCELLULAR")
ner_labels=("CELL_TYPE" "GENEPROD" "EXP_ASSAY" "DISEASE")

for i in ${!masking_probability[@]}; do
    for j in ${!ner_labels[@]}; do
        python -m soda_model.token_classification.trainer \
            --dataset_id "EMBO/SourceData" \
            --task NER \
            --version 2.0.3 \
            --from_pretrained "michiyasunaga/BioLinkBERT-base" \
            --masking_probability ${masking_probability[$i]} \
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
            --ner_labels ${ner_labels[$j]} \
            --results_file "v2.0.3\ner_${ner_labels[$j]}_BioLinkBERT_base_maskingprob_${masking_probability[$i]}_"
    done
done


ner_labels=("CELL_LINE" "SMALL_MOLECULE" "TISSUE" "ORGANISM" "SUBCELLULAR")
masking_probability=(0.0)
for i in ${!masking_probability[@]}; do
    for j in ${!ner_labels[@]}; do
        python -m soda_model.token_classification.trainer \
            --dataset_id "EMBO/SourceData" \
            --task NER \
            --version 2.0.3 \
            --from_pretrained "michiyasunaga/BioLinkBERT-base" \
            --masking_probability ${masking_probability[$i]} \
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
            --ner_labels ${ner_labels[$j]} \
            --results_file "v2.0.3\ner_${ner_labels[$j]}_BioLinkBERT_base_maskingprob_${masking_probability[$i]}_"
    done
done

# for i in ${!large_model_list[@]}; do
#         python -m soda_model.token_classification.trainer \
#             --dataset_id "EMBO/SourceData" \
#             --task NER \
#             --version 1.0.0 \
#             --from_pretrained ${large_model_list[$i]} \
#             --masking_probability 0.25 \
#             --replacement_probability 0.0 \
#             --gradient_accumulation_steps 2 \
#             --per_device_train_batch_size 8 \
#             --per_device_eval_batch_size 32 \
#             --add_prefix_space \
#             --num_train_epochs 2.0 \
#             --learning_rate 0.00005 \
#             --gradient_accumulation_steps 2 \
#             --disable_tqdm False \
#             --report_to none \
#             --classifier_dropout 0.2 \
#             --do_train \
#             --do_predict \
#             --truncation \
#             --padding "longest" \
#             --ner_labels ${ner_labels[$j]} \
#             --filter_empty \
#             --results_file "ner_${ner_labels[$j]}_${model_name_list[$i]}_large_maskingprob_0.0_"
#    done
# done

# for i in ${!large_model_list[@]}; do
#         python -m soda_model.token_classification.trainer \
#             --dataset_id "EMBO/SourceData" \
#             --task NER \
#             --version 1.0.0 \
#             --from_pretrained ${large_model_list[$i]} \
#             --masking_probability 0.0 \
#             --replacement_probability 0.0 \
#             --gradient_accumulation_steps 2 \
#             --per_device_train_batch_size 8 \
#             --per_device_eval_batch_size 32 \
#             --add_prefix_space \
#             --use_crf \
#             --num_train_epochs 2.0 \
#             --learning_rate 0.00005 \
#             --gradient_accumulation_steps 2 \
#             --disable_tqdm False \
#             --report_to none \
#             --classifier_dropout 0.2 \
#             --do_train \
#             --do_predict \
#             --truncation \
#             --padding "longest" \
#             --ner_labels ${ner_labels[$j]} \
#             --filter_empty \
#             --results_file "ner_CRF_${ner_labels[$j]}_${model_name_list[$i]}_large_maskingprob_0.0_"
#    done
# done


python -m soda_model.token_classification.trainer \
    --dataset_id "EMBO/SourceData" \
    --task NER \
    --version 2.0.3 \
    --from_pretrained "michiyasunaga/BioLinkBERT-base" \
    --masking_probability 1.0 \
    --replacement_probability 0.0 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --add_prefix_space \
    --num_train_epochs 0.5 \
    --learning_rate 0.0001 \
    --disable_tqdm False \
    --report_to none \
    --classifier_dropout 0.2 \
    --do_train \
    --do_predict \
    --truncation \
    --padding "longest" \
    --ner_labels GENEPROD \
    --results_file "v2.0.3\ner_GENEPROD_BioLinkBERT_base_maskingprob_1.0_"
