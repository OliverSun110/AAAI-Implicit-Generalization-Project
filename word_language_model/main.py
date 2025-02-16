# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import random

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--upperweight', type=float, default=0.1,
                    help='initial weight upper bound')
parser.add_argument('--lowerweight', type=float, default=-0.1,
                    help='iniital weight lower bound')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    #random.shuffle(data)
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied, upperweight=args.upperweight, lowerweight=args.lowerweight).to(device)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    '''
    noise_rate = 0.5
    num = float(target.size(0))*0.5
    for i in range (int(num)):
        target[random.randint(0,int(num))] = '<unk>'
    '''
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    train_loss=0
    train_ppl=0
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        j = random.randint(0,len(train_data)-2)
        data, targets = get_batch(train_data, j)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            train_loss = cur_loss
            train_ppl = math.exp(cur_loss)
            start_time = time.time()
    return (train_ppl, train_loss)


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None
#print("batch size: ", args.batch_size)
# At any point you can hit Ctrl + C to break out of training early.
try:
    with open('output_batch' + str(args.batch_size) + '.txt','a+') as f:
        f.write('\n' + '+' * 89 + '\n' )
        f.write('Weight initialization range ('+str(args.upperweight) + ', ' + str(args.lowerweight) + ') \n')
        f.write('Random Seed: ' + str(args.seed)+'\n')
    count = 0
    early_stop = 0
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        #shuffle each epoch 
        #traindata = batchify(corpus.train, args.batch_size)
        (train_ppl, train_loss) = train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        '''
        with open('output_batch' + str(args.batch_size)  + '.txt','w+') as file:
            file.write('-' * 89 + '\n')
            file.write('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                             'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                        val_loss, math.exp(val_loss)))
            file.write('\n' + '-' * 89 + '-' + '\n')
        '''
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            count = 0
        else:
            print('Valid Loss stop decreasing the ' + str(count) + 'time in epoch ' + str(epoch))
            if early_stop == 1:
                if count < 2 * args.batch_size:
                    if count > 0:
                        lr /= 4.0
                    count += 1
                    continue
                break
            if early_stop == 0:
                if count < 2 * args.batch_size:
                     count += 1
                     continue
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            print("Early stopping in epoch " + str(epoch))
            # train
            with open('output_batch' + str(args.batch_size)  + '.txt','a+') as file:
                file.write('=' * 89 + '\n')
                file.write('Early Train: | end of epoch' + str(epoch) + '| train ppl ' + str(train_ppl) + '| train loss ' + str(train_loss) + '.' )
                file.write('\n' + '=' * 89 + '\n')            
            # valid
            with open('output_batch' + str(args.batch_size)  + '.txt','a+') as file:
                file.write('=' * 89 + '\n')
                file.write('Early Valid: | end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                           'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),val_loss, math.exp(val_loss)))
                file.write('\n' + '=' * 89 + '\n')
            # Run on test data.
            test_loss = evaluate(test_data)
            print('=' * 89)
            print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
                test_loss, math.exp(test_loss)))
            print('=' * 89)
            with open('output_batch' + str(args.batch_size) + '.txt','a+') as f:
                       f.write('=' * 89 + '\n')
                       f.write('Early Test: | End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
                           test_loss, math.exp(test_loss)))
                       f.write('\n' + '=' * 89 + '\n')
            lr /= 4.0
            early_stop = 1
            count = 0
                           
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()
'''
# train
train_loss = evaluate(train_data)
with open('output_batch' + str(args.batch_size)  + '.txt','a+') as file:
    file.write('=' * 89 + '\n')
    file.write('Final Train | end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
               'train ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),train_loss, math.exp(train_loss)))
    file.write('\n' + '=' * 89 + '\n')
'''
# valid 
val_loss = evaluate(val_data)
with open('output_batch' + str(args.batch_size)  + '.txt','a+') as file:
    file.write('=' * 89 + '\n')
    file.write('Final Valid | end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
               'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),val_loss, math.exp(val_loss)))
    file.write('\n' + '=' * 89 + '\n')
# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
with open('output_batch' + str(args.batch_size)  +  '.txt','a+') as f:
    f.write('=' * 89 + '\n')
    f.write('Final Test: | End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    f.write('\n' + '=' * 89 + '\n')

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
