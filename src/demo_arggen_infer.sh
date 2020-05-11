python main.py --mode=predict \
    --exp_name=arggen_mt_20_noEncMulti \
    --encode_passage \
    --replace_unk \
    --type_conditional_lm \
    --infer_fold=4 \
    --infer_fold_selected=4 \
    --task=arggen \
    --batch_size=32 \
    --load_model_path=epoch_20_train_168.7582_val_182.5564_ppl_7.3618.tar
