translate:
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 1; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 3; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 5; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 10; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 20; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 25; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 30; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 35; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 50; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 100; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 200; \

train:
	python3 preprocess.py -train_src data/wnt15de/train.en -train_tgt data/wnt15de/train.de -valid_src data/wnt15de/newstest2015.en -valid_tgt data/wnt15de/newstest2015.de -save_data data/datade -src_vocab_size 50000 -tgt_vocab_size 50000 ; \
	python3 train.py -data data/datade -save_model dedata_model_1_ -gpuid 0 -rnn_size 100 -layers 1 -optim sgd -learning_rate 0.001 -batch_size 1 -optim sgd -report_every 5000; \
	python3 train.py -data data/datade -save_model dedata-model_3_ -gpuid 0 -rnn_size 100 -layers 1 -optim sgd -learning_rate 0.001 -batch_size 3 -optim sgd -report_every 5000; \
	python3 train.py -data data/datade -save_model dedata-model_5_ -gpuid 0 -rnn_size 100 -layers 1 -optim sgd -learning_rate 0.001 -batch_size 5 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 train.py -data data/datade -save_model dedata-model_10_ -gpuid 0 -rnn_size 100 -layers 1 -optim sgd -learning_rate 0.001 -batch_size 10 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 train.py -data data/datade -save_model dedata-model_20_ -gpuid 0 -rnn_size 100 -layers 1 -optim sgd -learning_rate 0.001 -batch_size 20 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 train.py -data data/datade -save_model dedata-model_25_ -gpuid 0 -rnn_size 100 -layers 1 -optim sgd -learning_rate 0.001 -batch_size 25 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 train.py -data data/datade -save_model dedata-model_30_ -gpuid 0 -rnn_size 100 -layers 1 -optim sgd -learning_rate 0.001 -batch_size 30 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 train.py -data data/datade -save_model dedata-model_35_ -gpuid 0 -rnn_size 100 -layers 1 -optim sgd -learning_rate 0.001 -batch_size 35 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 train.py -data data/datade -save_model dedata-model_50_ -gpuid 0 -rnn_size 100 -layers 1 -optim sgd -learning_rate 0.001 -batch_size 50 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 train.py -data data/datade -save_model dedata-model_100_ -gpuid 0 -rnn_size 100 -layers 1 -optim sgd -learning_rate 0.001 -batch_size 100 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 train.py -data data/datade -save_model dedata-model_200_ -gpuid 0 -rnn_size 100 -layers 1 -optim sgd -learning_rate 0.001 -batch_size 200 -optim sgd -report_every 5000 -train_steps 100000; \

test:
	#python3 preprocess.py -train_src data/wnt15vi/train.en -train_tgt data/wnt15vi/train.vi -valid_src data/wnt15vi/tst2013.en -valid_tgt data/wnt15vi/tst2013.vi -save_data data/datavi -src_vocab_size 50000 -tgt_vocab_size 50000
	python3 train.py -data data/datavi -save_model vidata-model_1 -gpuid 0 -rnn_size 500 -layers 1 -word_vec_size 300 -optim sgd -learning_rate 0.2 -learning_rate_decay 1 -batch_size 1 -optim sgd -report_every 2000 -train_steps 200000 -save_checkpoint_steps 100000
	python3 train.py -data data/datavi -save_model vidata-model_1 -gpuid 0 -rnn_size 500 -layers 1 -word_vec_size 300 -optim sgd -learning_rate 0.2 -learning_rate_decay 1 -batch_size 5 -optim sgd -report_every 2000 -train_steps 200000 -save_checkpoint_steps 50000
	python3 train.py -data data/datavi -save_model vidata-model_1 -gpuid 0 -rnn_size 500 -layers 1 -word_vec_size 300 -optim sgd -learning_rate 0.2 -learning_rate_decay 1 -batch_size 10 -optim sgd -report_every 2000 -train_steps 200000 -save_checkpoint_steps 50000
	python3 train.py -data data/datavi -save_model vidata-model_1 -gpuid 0 -rnn_size 500 -layers 1 -word_vec_size 300 -optim sgd -learning_rate 0.2 -learning_rate_decay 1 -batch_size 15 -optim sgd -report_every 2000 -train_steps 200000 -save_checkpoint_steps 50000
	python3 train.py -data data/datavi -save_model vidata-model_1 -gpuid 0 -rnn_size 500 -layers 1 -word_vec_size 300 -optim sgd -learning_rate 0.2 -learning_rate_decay 1 -batch_size 20 -optim sgd -report_every 2000 -train_steps 200000 -save_checkpoint_steps 50000
