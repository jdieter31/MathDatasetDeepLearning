import torchtext
from torchtext.data import Field, Iterator, TabularDataset, Dataset
from torchtext.data.utils import get_tokenizer
import torch
import torch.nn.functional as F
import math
from model.transformer_model import TransformerModel
import numpy as np
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

batch_size = 400
eval_batch_size = 10

"""
def tokenize(sentence):
    return list(sentence)
"""
# non_letters = ["0","1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "(", ")", "=", "/", ".", "?"]
punctuation = set([",", ".", "?", ":", ";", "!"])
from nltk.corpus import words
en_words = set(words.words())

def tokenize(text):
    text_split = text.split(" ")
    split_by_spaces = []

    # Add in spaces to text_split
    for i in range(2 * len(text_split) - 1):
        if i % 2 == 0:
            # Only add token if not empty
            if text_split[i // 2]:
                # Break off punctuation
                if text_split[i // 2][-1] in punctuation:
                    if text_split[i // 2][:-1]:
                        split_by_spaces.append(text_split[i // 2][:-1])
                    split_by_spaces.append(text_split[i // 2][-1])
                else:
                    split_by_spaces.append(text_split[i // 2])
        else:
            split_by_spaces.append(" ")

    tokens = []
    for word in split_by_spaces: 
        # If english word don't split by character
        if len(word) > 3 and word.lower() in en_words:
            tokens.append(word)
        else:
            tokens += word

    return tokens;


def combine_datasets(datasets, fields):
    list_of_ex = sum([[x for x in d] for d in datasets], [])
    return Dataset(list_of_ex, fields)

def reduce_dataset(dataset, fields, fraction=0.3):
    list_of_ex = [x for x in dataset]
    list_of_ex = list_of_ex[:int(len(list_of_ex) * fraction)]
    return Dataset(list_of_ex, fields)

def get_datasets_from_dir(path, fields, reduce_data=False):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    datasets = {}
    for file_name in tqdm(files):
        file_loc = path + file_name
        if os.path.splitext(file_loc)[1] == ".tsv":
            datasets[os.path.splitext(file_name)[0]] = TabularDataset(path=file_loc, format='tsv', fields=data_fields, skip_header=True)
            if reduce_data:
                datasets[os.path.splitext(file_name)[0]] = reduce_dataset(datasets[os.path.splitext(file_name)[0]], fields)
    return datasets

IN_TEXT = Field(tokenize=tokenize)
OUT_TEXT = Field(tokenize=tokenize, init_token = "<sos>", eos_token = "<eos>")

data_fields = [('question', IN_TEXT), ('answer', OUT_TEXT)]
# train,val = TabularDataset.splits(path='./', train='mathematics_dataset-v1.0/train-easy/arithmetic__add_or_sub.tsv', validation='mathematics_dataset-v1.0/interpolate/arithmetic__add_or_sub.tsv', format='tsv', fields=data_fields)

interp = combine_datasets(list(get_datasets_from_dir("./mathematics_dataset-v1.0/interpolate/", data_fields).values()), data_fields)
extrap = combine_datasets(list(get_datasets_from_dir("./mathematics_dataset-v1.0/extrapolate/", data_fields).values()), data_fields)
easy_train = combine_datasets(list(get_datasets_from_dir("./mathematics_dataset-v1.0/train-easy/", data_fields, True).values()), data_fields)
med_train = combine_datasets(list(get_datasets_from_dir("./mathematics_dataset-v1.0/train-medium/", data_fields, True).values()), data_fields)
hard_train = combine_datasets(list(get_datasets_from_dir("./mathematics_dataset-v1.0/train-hard/", data_fields, True).values()), data_fields)
total_train = combine_datasets([easy_train, med_train, hard_train], data_fields)

IN_TEXT.build_vocab(interp, extrap, total_train)
OUT_TEXT.build_vocab(interp, extrap, total_train)

# train_iter = BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.answer), shuffle=True)
# val_iter = BucketIterator(val, batch_size=batch_size, sort_key=lambda x: len(x.answer), shuffle=True)
interp_iter = Iterator(interp, batch_size=batch_size, shuffle=True, sort=False)
extrap_iter = Iterator(extrap, batch_size=batch_size, shuffle=True, sort=False)
train_iter = Iterator(total_train, batch_size=batch_size, shuffle=True, sort=False)

ntokens = len(IN_TEXT.vocab.stoi) # the size of vocabulary
ntokens_dec = len(OUT_TEXT.vocab.stoi)
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0 # the dropout value
model = TransformerModel(emsize, ntokens, ntokens_dec, dropout=dropout)
model_name = "default_transformer"
model.to(device)

lr = 0.0001 # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(.9, .995), eps=1e-9)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

best_extrap_loss = float("inf")

import time

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg):
    src_mask = (src != 1)
    trg_mask = (trg != 1)
    '''
    if trg is not None:
        trg_mask = (trg != 1).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    '''

    return src_mask, trg_mask


def train():
    model.train() # Turn on the train mode
    global best_extrap_loss
    total_loss = 0.
    start_time = time.time()
    ntokens = len(IN_TEXT.vocab.stoi)
    from tqdm import tqdm
    for i, batch in tqdm(enumerate(train_iter)):
        optimizer.zero_grad()
        src = batch.question.transpose(0,1).to(device)
        trg = batch.answer.transpose(0,1).to(device)
        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input)
        preds = model(src, trg_input, src_mask=src_mask, tgt_mask=trg_mask)
        ys = trg[:, 1:].transpose(0,1).contiguous().view(-1)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        writer.add_scalar('loss/train', loss.item(), i + (epoch - 1) * len(train_iter))
        log_interval = 500
        generate_interval = 500
        if (i + (epoch - 1) * len(train_iter)) % log_interval == 0 and (i + (epoch - 1) * len(train_iter)) > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i, len(train_iter), 0.01, #scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            generate = i % generate_interval == 0
            total_loss, (total_right, total_tested) = evaluate(model, interp_iter, generate)
            writer.add_scalar('loss/interp', total_loss, i + (epoch - 1) * len(train_iter))
            if generate:
                writer.add_scalar('acc/interp', total_right.float() / total_tested, i + (epoch - 1) * len(train_iter))

            total_loss, (total_right, total_tested) = evaluate(model, extrap_iter, generate)
            writer.add_scalar('loss/extrap', total_loss, i + (epoch - 1) * len(train_iter))
            if generate:
                writer.add_scalar('acc/extrap', total_right.float() / total_tested, i + (epoch - 1) * len(train_iter))
            if total_loss < best_extrap_loss:
                best_extrap_loss = total_loss
                model.save(f"models/{model_name}.pkl")
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_iter, generate=True, num_batches=50):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0
    total_right = 0
    total_tested = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_iter)):
            if i >= num_batches:
                break
            src = batch.question.transpose(0,1).to(device)
            trg = batch.answer.transpose(0,1).to(device)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input)
            preds = eval_model(src, trg_input, src_mask=src_mask, tgt_mask=trg_mask)
            ys = trg[:, 1:].transpose(0,1).contiguous().view(-1)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=1)
            if generate:
                generated = eval_model.greedy_decode(src, src_mask, OUT_TEXT.vocab.stoi["<sos>"], max_length=trg.size(1))
                num_right = (((generated * (trg != 1).long()) - (trg * (trg != 1).long())).abs().sum(dim=-1) == 0).sum()
                total_tested += trg.size(0)
                total_right += num_right

            total_loss += loss.item()
    return total_loss / min(len(data_iter), num_batches), (total_right, total_tested)

######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.

epochs = 100 # The number of epochs

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    print('-' * 89)
    val_loss = 0
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    #scheduler.step()


######################################################################
# Evaluate the model with the test dataset
# -------------------------------------
#
# Apply the best model to check the result with the test dataset.

'''
test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
'''
