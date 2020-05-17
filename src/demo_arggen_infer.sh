python main.py --mode=predict \
    --exp_name=arggen_mt_20_test \
    --encode_passage \
    --replace_unk \
    --type_conditional_lm \
    --infer_fold=4 \
    --infer_fold_selected=4 \
    --task=arggen \
    --batch_size=20 \
    --load_model_path=epoch_21_train_166.7865_val_181.5731_ppl_8.0979.tar
