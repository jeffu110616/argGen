python main.py --mode=predict \
    --exp_name=arggen_20 \
    --infer_fold=4 \
    --infer_fold_selected=2 \
    --encode_passage \
    --replace_unk \
    --type_conditional_lm \
    --task=arggen \
    --batch_size=48 \
    --load_model_path=epoch_26_train_167.6045_val_181.4993_ppl_6.8601.tar
