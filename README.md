
python main.py --do_train --load_local_model --train_file=data/ --train_batch_size=4 --num_train_epochs=100

python main.py --do_predict --load_local_model --local_model_folder=check_points/ --local_tokenizer_folder=check_points/ --predict_file=data/ --predict_batch_size=1 --max_answer_length=30
