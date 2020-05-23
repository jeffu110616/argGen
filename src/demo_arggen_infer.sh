python main.py --mode=predict \
    --exp_name=arggen_20_noEncPsgSingle \
    --infer_fold=6 \
    --infer_fold_selected=6 \
    --replace_unk \
    --type_conditional_lm \
    --task=arggen \
    --batch_size=32 \
    --max_tgt_words=150 \
    --load_model_path=epoch_15_train_172.7537_val_183.8388_ppl_7.4646.tar
