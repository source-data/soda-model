small_model_list=("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
large_model_list=("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")

ner_labels=("DISEASE" "CELL_LINE" "CELL_TYPE" "GENEPROD" "SMALL_MOLECULE" "EXP_ASSAY" "TISSUE" "ORGANISM" "SUBCELLULAR")

for i in ${!small_model_list[@]}; do
    for j in ${!ner_labels[@]}; do
        python -m soda_model.token_classification.trainer \
            --dataset_id "EMBO/SourceData" \
            --task NER \
            --version 1.0.0 \
            --from_pretrained ${small_model_list[$i]} \
            --masking_probability 0.0 \
            --replacement_probability 0.0 \
            --per_device_train_batch_size 16 \
            --add_prefix_space \
            --num_train_epochs 2.0 \
            --learning_rate 0.0001 \
            --disable_tqdm False \
            --report_to none \
            --classifier_dropout 0.2 \
            --do_train \
            --do_predict \
            --truncation \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --padding "longest" \
            --ner_labels ${ner_labels[$j]} \
            --filter_empty \
            --results_file "ner_select_for_gen_vs_memo_base_pmb_only_${ner_labels[$j]}"
    done
done

for i in ${!large_model_list[@]}; do
        python -m soda_model.token_classification.trainer \
            --dataset_id "EMBO/SourceData" \
            --task NER \
            --version 1.0.0 \
            --from_pretrained ${large_model_list[$i]} \
            --masking_probability 0.0 \
            --replacement_probability 0.0 \
            --per_device_train_batch_size 8 \
            --add_prefix_space \
            --num_train_epochs 1.0 \
            --learning_rate 0.00005 \
            --disable_tqdm False \
            --report_to none \
            --classifier_dropout 0.2 \
            --do_train \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --do_predict \
            --truncation \
            --padding "longest" \
            --ner_labels ${ner_labels[$j]} \
            --filter_empty \
            --results_file "ner_select_for_gen_vs_memo_large_pmb_only_${ner_labels[$j]}"
   done
done
