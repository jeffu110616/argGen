python main.py --mode=predict \
    --exp_name=arggen_mt_trans_100k \
    --encode_passage \
    --replace_unk \
    --type_conditional_lm \
    --infer_fold=4 \
    --infer_fold_selected=4 \
    --task=arggen \
    --batch_size=48 \
    --load_model_path=epoch_32_train_170.9898_val_197.8458_ppl_8.1034.tar
