python main.py --mode=predict \
    --exp_name=arggen_20_noEncPsgSingle \
    --infer_fold=4 \
    --infer_fold_selected=4 \
    --replace_unk \
    --type_conditional_lm \
    --task=arggen \
    --batch_size=32 \
    --load_model_path=epoch_21_train_169.2129_val_183.3461_ppl_7.4259.tar
