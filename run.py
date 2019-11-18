import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.data.utils import get_tokenizer
import torch
import torch.nn.functional as F
import math
from model.transformer_model import TransformerModel
import numpy as np
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

batch_size = 1000
eval_batch_size = 10

def tokenize(sentence):
    return list(sentence)

IN_TEXT = Field(tokenize=tokenize)
OUT_TEXT = Field(tokenize=tokenize, init_token = "<sos>", eos_token = "<eos>")

data_fields = [('question', IN_TEXT), ('answer', OUT_TEXT)]
train,val = TabularDataset.splits(path='./', train='mathematics_dataset-v1.0/train-easy/arithmetic__add_or_sub.tsv', validation='mathematics_dataset-v1.0/interpolate/arithmetic__add_or_sub.tsv', format='tsv', fields=data_fields)

IN_TEXT.build_vocab(train, val)
OUT_TEXT.build_vocab(train, val)

train_iter = BucketIterator(train, batch_size=batch_size, \
sort_key=lambda x: len(x.answer), shuffle=True)

val_iter = BucketIterator(val, batch_size=batch_size, \
sort_key=lambda x: len(x.answer), shuffle=True)

ntokens = len(IN_TEXT.vocab.stoi) # the size of vocabulary
ntokens_dec = len(OUT_TEXT.vocab.stoi) 
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(emsize, ntokens, ntokens_dec, dropout=dropout)
model.to(device)

lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

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
    total_loss = 0.
    start_time = time.time()
    ntokens = len(IN_TEXT.vocab.stoi)
    from tqdm import tqdm
    for i, batch in tqdm(enumerate(train_iter)):
        src = batch.question.transpose(0,1).to(device)
        trg = batch.answer.transpose(0,1).to(device)
        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input)
        preds = model(src, trg_input, src_mask=src_mask, tgt_mask=trg_mask)
        ys = trg[:, 1:].contiguous().view(-1)
        optimizer.zero_grad()
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        writer.add_scalar('loss/train', loss.item(), i + (epoch - 1) * len(train_iter))
        log_interval = 50
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i, len(train_iter), scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            writer.add_scalar('loss/val', evaluate(model), i + (epoch - 1) * len(train_iter))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        from tqdm import tqdm
        for i, batch in tqdm(enumerate(val_iter)):
            src = batch.question.transpose(0,1).to(device)
            trg = batch.answer.transpose(0,1).to(device)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input)
            preds = eval_model(src, trg_input, src_mask=src_mask, tgt_mask=trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=1)
            total_loss += loss.item()
    return total_loss / (len(val_iter))

######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.

best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = 0 #evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()


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
