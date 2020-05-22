python main.py --mode=predict \
    --exp_name=arggen_20_noEncPsgSingle \
    --infer_fold=1 \
    --infer_fold_selected=1 \
    --replace_unk \
    --type_conditional_lm \
    --task=arggen \
    --batch_size=32 \
    --max_tgt_words=150 \
    --load_model_path=epoch_27_train_166.6683_val_183.5460_ppl_7.4430.tar
