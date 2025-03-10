python main.py --mode=train \
		--exp_name=arggen_mt_20_test \
	    --encode_passage \
		--type_conditional_lm \
		--task=arggen \
		--batch_size=20 \
		--num_train_epochs=50 \
		--logging_freq=2 \
    	--max_src_words=500 \
	    --max_passage_words=200 \
		--max_sent_num=10 \
		--max_bank_size=70 \
		--load_model_path=epoch_15_train_169.8734_val_181.8314_ppl_8.1217.tar
