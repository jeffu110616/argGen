python3 main.py --mode=train \
    --exp_name=arggen_20 \
    --encode_passage \
    --type_conditional_lm \
    --debug \
    --task=arggen \
    --batch_size=20 \
    --num_train_epochs=50 \
    --logging_freq=2
