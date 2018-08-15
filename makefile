train:
	python3 preprocess.py -train_src data/wnt15de/train.en -train_tgt data/wnt15de/train.de -valid_src data/wnt15de/newstest2015.en -valid_tgt data/wnt15de/newstest2015.de -save_data data/datade -src_vocab_size 50000 -tgt_vocab_size 50000 ; \
	python3 train.py -data data/datade -save_model dedata-model -gpuid 0 -rnn_size 100 -layers 1 -shuffle -optim sgd -learning_rate 0.001 -batch_size 1 -optim sgd -report_every 5000; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 1; \
	python3 train.py -data data/datade -save_model dedata-model -gpuid 0 -rnn_size 100 -layers 1 -shuffle -optim sgd -learning_rate 0.001 -batch_size 3 -optim sgd -report_every 5000; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 3; \
	python3 train.py -data data/datade -save_model dedata-model -gpuid 0 -rnn_size 100 -layers 1 -shuffle -optim sgd -learning_rate 0.001 -batch_size 5 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 5; \
	python3 train.py -data data/datade -save_model dedata-model -gpuid 0 -rnn_size 100 -layers 1 -shuffle -optim sgd -learning_rate 0.001 -batch_size 10 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 10; \
	python3 train.py -data data/datade -save_model dedata-model -gpuid 0 -rnn_size 100 -layers 1 -shuffle -optim sgd -learning_rate 0.001 -batch_size 20 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 20; \
	python3 train.py -data data/datade -save_model dedata-model -gpuid 0 -rnn_size 100 -layers 1 -shuffle -optim sgd -learning_rate 0.001 -batch_size 25 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 25; \
	python3 train.py -data data/datade -save_model dedata-model -gpuid 0 -rnn_size 100 -layers 1 -shuffle -optim sgd -learning_rate 0.001 -batch_size 30 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 30; \
	python3 train.py -data data/datade -save_model dedata-model -gpuid 0 -rnn_size 100 -layers 1 -shuffle -optim sgd -learning_rate 0.001 -batch_size 35 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 35; \
	python3 train.py -data data/datade -save_model dedata-model -gpuid 0 -rnn_size 100 -layers 1 -shuffle -optim sgd -learning_rate 0.001 -batch_size 50 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 50; \
	python3 train.py -data data/datade -save_model dedata-model -gpuid 0 -rnn_size 100 -layers 1 -shuffle -optim sgd -learning_rate 0.001 -batch_size 100 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 100; \
	python3 train.py -data data/datade -save_model dedata-model -gpuid 0 -rnn_size 100 -layers 1 -shuffle -optim sgd -learning_rate 0.001 -batch_size 200 -optim sgd -report_every 5000 -train_steps 100000; \
	python3 translate.py -model dedata-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge -batch_size 200; \
