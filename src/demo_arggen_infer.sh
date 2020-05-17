python main.py --mode=predict \
    --exp_name=arggen_mt_20_speakerEmbed \
    --encode_passage \
    --replace_unk \
    --type_conditional_lm \
    --infer_fold=4 \
    --infer_fold_selected=4 \
    --task=arggen \
    --batch_size=32 \
    --load_model_path=epoch_20_train_170.5814_val_183.0892_ppl_7.4044.tar
