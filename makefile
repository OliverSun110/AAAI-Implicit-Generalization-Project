preprocess:
	python2 preprocess.py -train_src data/wnt15de/train.en -train_tgt data/wnt15de/train.de -valid_src data/wnt15de/newtest2015.en -valid_tgt data/wnt15de/newstest2015.de -save_data data/datade -src_vocab_size 50000 -tgt_vocab_size 50000

train:
	python2 train.py -data data/datade -save_model dedata-model -gpuid 0 -rnn_size 100 -layers 1 -optim sgd -learning_rate 0.001

translate:
	python2 translate.py -model data-model_step_50000.pt -src data/wnt15de/newtest2015.en -tgt data/wnt15de/newstest2015.de -output pred.txt -replace_unk -verbose -report_bleu -report_rouge
