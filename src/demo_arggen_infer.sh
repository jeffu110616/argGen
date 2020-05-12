python main.py --mode=predict \
    --exp_name=arggen_mt_20_speakerEmbed \
    --encode_passage \
    --replace_unk \
    --type_conditional_lm \
    --infer_fold=4 \
    --infer_fold_selected=4 \
    --task=arggen \
    --batch_size=32 \
    --load_model_path=epoch_17_train_172.4395_val_183.1406_ppl_7.4081.tar
