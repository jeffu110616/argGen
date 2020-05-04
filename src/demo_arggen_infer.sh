python main.py --mode=predict \
    --exp_name=arggen_mt_trans \
    --encode_passage \
    --replace_unk \
    --type_conditional_lm \
    --infer_fold=4 \
    --infer_fold_selected=3 \
    --task=arggen \
    --batch_size=48 \
    --load_model_path=epoch_49_train_198.8612_val_257.5234_ppl_15.7782.tar
