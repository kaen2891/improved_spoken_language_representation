import torch
import torch.nn as nn
from transformers import AdamW, BertConfig, BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import torch.backends.cudnn as cudnn
#import tensorflow as tf
import warnings
from glob import glob

import pandas as pd
import numpy as np 
import random
import time
import datetime
#SET PARAMETERS
import argparse
import os

from transformers import BertForSequenceClassification, BertTokenizer # BERT
from transformers import AlbertForSequenceClassification, AlbertTokenizer # ALBERT
from transformers import XLNetForSequenceClassification, XLNetTokenizer # XL-NET
from transformers import ElectraForSequenceClassification, ElectraTokenizer #ELECTRA
from transformers import RobertaForSequenceClassification, RobertaTokenizer #ROBERTa

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4, help='learning_rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--gpus', type=str, default='0', help='gpu numbers')
parser.add_argument('--epochs', type=int, default=5, help='gpu numbers')
parser.add_argument('--ckpt', type=str, default='0', help='')

parser.add_argument('--seed', type=int, default=2891)
parser.add_argument('--warmup', type=int, default=4000)
parser.add_argument('--optim', type=str, default='adamw')

parser.add_argument('--w1', type=float, default='0.55')
parser.add_argument('--w2', type=float, default='0.45')
parser.add_argument('--drop', type=float, default='0.3')

parser.add_argument('--data_dir', default='./dataset/snips_audio/', type=str, help='dataset dir')
parser.add_argument('--dataset_name', default='salli', type=str, help='dataset name')
parser.add_argument('--dataset_name2', default='Audio-Snips', type=str, help='dataset name')
parser.add_argument('--resume', default='None', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--pretrained_model', default='bert', type=str, help='pretrained_model')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
MAX_LEN = 64

import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Source Label')
    plt.xlabel('Predicted Label')
    #plt.show()


if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    warnings.warn('You have chosen to seed training. '
        'This will turn on the CUDNN deterministic setting, '
        'which can slow down your training considerably! '
        'You may see unexpected behavior when restarting '
        'from checkpoints.')


def preprocess_data(tokenizer, dataset, mode, name):
    
    labelled_sentences = list()
    recognized_sentences = list()
    intent_labels = list()
    
    if name == 'all':
        
        labelled_sentence = list()
        recognized_sentence = list()
        label = list()
        for i in range(len(dataset)):
            _, file_names = os.path.split(dataset[i])
            name, _ = file_names.split('_')
            data = pd.read_csv(dataset[i]).dropna()
            
            labelled_sentence_tmp = data['{}_sentence'.format(mode)]
            recognized_sentence_tmp = data['{}_sentence'.format(name)]
            label_tmp = data['{}_label'.format(mode)]
            
            labelled_sentence.extend(labelled_sentence_tmp)
            recognized_sentence.extend(recognized_sentence_tmp)
            label.extend(label_tmp)
    
    elif name is not None:
        data = pd.read_csv(dataset).dropna()
        
        labelled_sentence = data['{}_sentence'.format(mode)]
        recognized_sentence = data['{}_sentence'.format(name)]
        label = data['{}_label'.format(mode)]
                    
    else:
        data = pd.read_csv(dataset).dropna()
        labelled_sentence = data['label_sentence'].values.tolist()        
        recognized_sentence = data['stt_sentence'].values.tolist()
        label = data['label'].values.tolist()
        
    for i in range(len(labelled_sentence)):        
        labelled_sen_tmp = labelled_sentence[i].lower()
        labelled_sentences.append(labelled_sen_tmp)
        
        recognized_sen_tmp = recognized_sentence[i].lower()
        recognized_sentences.append(recognized_sen_tmp)
        
        intent_labels.append(label[i])
    
    label_list = list(set(intent_labels))
    label_list.sort()
    label_map = {label: i for i, label in enumerate(label_list)}
    
    labelled_inputs = []
    labelled_segs = []
    
    recognized_inputs = []
    recognized_segs = []
    
    output_labels = []
    for i in range(len(labelled_sentences)):
        input_dict_labelled = tokenizer(labelled_sentences[i], padding = 'max_length', truncation = True, max_length = MAX_LEN, return_tensors = 'pt', return_attention_mask = False)
        input_dict_recognized = tokenizer(recognized_sentences[i], padding = 'max_length', truncation = True, max_length = MAX_LEN, return_tensors = 'pt', return_attention_mask = False)
        labelled_inputs.append(input_dict_labelled['input_ids'])
        recognized_inputs.append(input_dict_recognized['input_ids'])
        
        
        if args.pretrained_model != 'roberta':
            labelled_segs.append(input_dict_labelled['token_type_ids'])
            recognized_segs.append(input_dict_recognized['token_type_ids'])
                                
        output_labels.append(label_map[intent_labels[i]])
    
    if args.pretrained_model != 'roberta':
        input_tensor_labelled = torch.stack(labelled_inputs, dim=0)    
        seg_tensor_labelled = torch.stack(labelled_segs, dim=0)
        mask_tensor_labelled = ~ (input_tensor_labelled == 0)
        output_tensor_labelled = torch.cat([input_tensor_labelled, seg_tensor_labelled, mask_tensor_labelled], dim=1)
        
        input_tensor_recognized = torch.stack(recognized_inputs, dim=0)    
        seg_tensor_recognized = torch.stack(recognized_segs, dim=0)
        mask_tensor_recognized = ~ (input_tensor_recognized == 0)
        output_tensor_recognized = torch.cat([input_tensor_recognized, seg_tensor_recognized, mask_tensor_recognized], dim=1)
    else:
        input_tensor_labelled = torch.stack(labelled_inputs, dim=0)
        mask_tensor_labelled = ~ (input_tensor_labelled == 0)
        output_tensor_labelled = torch.cat([input_tensor_labelled, mask_tensor_labelled], dim=1)
        
        input_tensor_recognized = torch.stack(recognized_inputs, dim=0)
        mask_tensor_recognized = ~ (input_tensor_recognized == 0)
        output_tensor_recognized = torch.cat([input_tensor_recognized, mask_tensor_recognized], dim=1)
    
    
    return output_tensor_labelled, output_tensor_recognized, torch.tensor(output_labels), label_list, labelled_sentences, recognized_sentences

def preprocess_data_mask(tokenizer, dataset, mode, name):
    
    labelled_sentences = list()
    recognized_sentences = list()
    intent_labels = list()
    
    if name == 'all':
        
        labelled_sentence = list()
        recognized_sentence = list()
        label = list()
        for i in range(len(dataset)):
            _, file_names = os.path.split(dataset[i])
            name, _ = file_names.split('_')
            data = pd.read_csv(dataset[i]).dropna()
            
            labelled_sentence_tmp = data['{}_sentence'.format(mode)]
            recognized_sentence_tmp = data['{}_sentence'.format(name)]
            label_tmp = data['{}_label'.format(mode)]
            
            labelled_sentence.extend(labelled_sentence_tmp)
            recognized_sentence.extend(recognized_sentence_tmp)
            label.extend(label_tmp)
    
    elif name is not None:
        data = pd.read_csv(dataset).dropna()
        
        labelled_sentence = data['{}_sentence'.format(mode)]
        recognized_sentence = data['{}_sentence'.format(name)]
        label = data['{}_label'.format(mode)]
                    
    else:
        data = pd.read_csv(dataset).dropna()
        labelled_sentence = data['label_sentence'].values.tolist()        
        recognized_sentence = data['stt_sentence'].values.tolist()
        label = data['label'].values.tolist()
        
    for i in range(len(labelled_sentence)):        
        labelled_sen_tmp = labelled_sentence[i].lower()
        labelled_sentences.append(labelled_sen_tmp)
        
        recognized_sen_tmp = recognized_sentence[i].lower()
        recognized_sentences.append(recognized_sen_tmp)
        
        intent_labels.append(label[i])
    
    label_list = list(set(intent_labels))
    label_list.sort()
    label_map = {label: i for i, label in enumerate(label_list)}
    
    labelled_inputs = []
    labelled_segs = []
    
    recognized_inputs = []
    recognized_segs = []
    
    output_labels = []
    for i in range(len(labelled_sentences)):
        
        input_sentence_labelled = labelled_sentences[i]
        input_sentence_recognized = recognized_sentences[i]
        
        word_cut_labelled = input_sentence_labelled.split(' ')
        word_cut_recognized = input_sentence_recognized.split(' ')
        
        output_sentence_labelled = ''
        output_sentence_recognized = ''
        
        for word in word_cut_labelled:
            prob = random.random()
            if prob <= args.drop:
                word = '[MASK]'
            output_sentence_labelled += word+' '
        
        for word in word_cut_recognized:
            prob = random.random()
            if prob <= args.drop:
                word = '[MASK]'
            output_sentence_recognized += word+' '
                
        
        input_dict_labelled = tokenizer(output_sentence_labelled.rstrip(), padding = 'max_length', truncation = True, max_length = MAX_LEN, return_tensors = 'pt', return_attention_mask = False)
        input_dict_recognized = tokenizer(output_sentence_recognized.rstrip(), padding = 'max_length', truncation = True, max_length = MAX_LEN, return_tensors = 'pt', return_attention_mask = False)
        labelled_inputs.append(input_dict_labelled['input_ids'])
        recognized_inputs.append(input_dict_recognized['input_ids'])        
        
        if args.pretrained_model != 'roberta':
            labelled_segs.append(input_dict_labelled['token_type_ids'])
            recognized_segs.append(input_dict_recognized['token_type_ids'])
                                
        output_labels.append(label_map[intent_labels[i]])
            
    if args.pretrained_model != 'roberta':
        input_tensor_labelled = torch.stack(labelled_inputs, dim=0)    
        seg_tensor_labelled = torch.stack(labelled_segs, dim=0)
        mask_tensor_labelled = ~ (input_tensor_labelled == 0)
        output_tensor_labelled = torch.cat([input_tensor_labelled, seg_tensor_labelled, mask_tensor_labelled], dim=1)
        
        input_tensor_recognized = torch.stack(recognized_inputs, dim=0)    
        seg_tensor_recognized = torch.stack(recognized_segs, dim=0)
        mask_tensor_recognized = ~ (input_tensor_recognized == 0)
        output_tensor_recognized = torch.cat([input_tensor_recognized, seg_tensor_recognized, mask_tensor_recognized], dim=1)
    else:
        input_tensor_labelled = torch.stack(labelled_inputs, dim=0)
        mask_tensor_labelled = ~ (input_tensor_labelled == 0)
        output_tensor_labelled = torch.cat([input_tensor_labelled, mask_tensor_labelled], dim=1)
        
        input_tensor_recognized = torch.stack(recognized_inputs, dim=0)
        mask_tensor_recognized = ~ (input_tensor_recognized == 0)
        output_tensor_recognized = torch.cat([input_tensor_recognized, mask_tensor_recognized], dim=1)
    
    
    return output_tensor_labelled, output_tensor_recognized, torch.tensor(output_labels), label_list, labelled_sentences, recognized_sentences
    
if args.dataset_name == 'all':
    train_dataset = sorted(glob(os.path.join(args.data_dir, '*_train.csv')))
    valid_dataset = sorted(glob(os.path.join(args.data_dir, '*_valid.csv')))
    test_dataset = sorted(glob(os.path.join(args.data_dir, '*_test.csv')))
elif args.dataset_name == 'FSC_dataset':
    train_dataset = os.path.join(args.data_dir, 'train_stt_result.csv')
    valid_dataset = os.path.join(args.data_dir, 'valid_stt_result.csv')
    test_dataset = os.path.join(args.data_dir, 'test_stt_result.csv')
else:
    train_dataset = os.path.join(args.data_dir, args.dataset_name.lower()+'_train.csv')
    valid_dataset = os.path.join(args.data_dir, args.dataset_name.lower()+'_valid.csv')
    test_dataset = os.path.join(args.data_dir, args.dataset_name.lower()+'_test.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from transformers import BertForSequenceClassification, BertTokenizer # BERT
#from transformers import AlbertForSequenceClassification, AlbertTokenizer # ALBERT
#from transformers import XLNetForSequenceClassification, XLNetTokenizer # XL-NET
#from transformers import ElectraForSequenceClassification, ElectraTokenizer #ELECTRA
#from transformers import RobertaForSequenceClassification, RobertaTokenizer #ROBERTa

if args.pretrained_model == 'bert':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
elif args.pretrained_model == 'albert':
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
elif args.pretrained_model == 'xlnet':
    tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
elif args.pretrained_model == 'electra':
    tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')    
elif args.pretrained_model == 'roberta':
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
else:
    print('unable')

print('selective tokenizer is', tokenizer)    


if args.dataset_name == 'FSC_dataset':
    labelled_train, recognized_train, train_label, label_names, _, _ = preprocess_data_mask(tokenizer, train_dataset, None, None) 
    labelled_valid, recognized_valid, valid_label, _, _, _ = preprocess_data(tokenizer, valid_dataset, None, None)
    labelled_test, recognized_test, test_label, _, labelled_sentences, recognized_sentences = preprocess_data(tokenizer, test_dataset, None, None)
else:
    labelled_train, recognized_train, train_label, label_names, _, _ = preprocess_data_mask(tokenizer, train_dataset, 'train', args.dataset_name.lower()) 
    labelled_valid, recognized_valid, valid_label, _, _, _ = preprocess_data(tokenizer, valid_dataset, 'valid', args.dataset_name.lower())
    labelled_test, recognized_test, test_label, _, labelled_sentences, recognized_sentences = preprocess_data(tokenizer, test_dataset, 'test', args.dataset_name.lower())

print('dataset_name {} labelled_train {} recognized_train {} train_label {}'.format(args.dataset_name, len(labelled_train), len(recognized_train), len(train_label)))
print('labelled_train {} recognized_train {} train_label {}'.format(len(labelled_train), len(recognized_train), len(train_label)))
print('labelled_valid {} recognized_valid {} valid_label {}'.format(len(labelled_valid), len(recognized_valid), len(valid_label)))
print('labelled_test {} recognized_test {} test_label {}'.format(len(labelled_test), len(recognized_test), len(test_label)))

## labelled dataset
train_set = TensorDataset(labelled_train, train_label)
#train_sampler = RandomSampler(train_set)
train_dataloader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size)

test_labelled = TensorDataset(labelled_test, test_label)
test_labelled_dataloader = DataLoader(test_labelled, shuffle=False, batch_size=args.batch_size)


test_recognized = TensorDataset(recognized_test, test_label)
test_recognized_dataloader = DataLoader(test_recognized, sampler=None, batch_size=args.batch_size)

### model
if args.pretrained_model == 'bert':
    llm = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_names)).cuda()
    #rlm = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_names)).cuda()
elif args.pretrained_model == 'albert':
    llm = AutoModelForSequenceClassification.from_pretrained('albert-base-v2', num_labels=len(label_names)).cuda()
    #rlm = AutoModelForSequenceClassification.from_pretrained('albert-base-v2', num_labels=len(label_names)).cuda()
elif args.pretrained_model == 'xlnet':
    llm = AutoModelForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=len(label_names)).cuda()
    #rlm = AutoModelForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=len(label_names)).cuda()
elif args.pretrained_model == 'electra':
    llm = AutoModelForSequenceClassification.from_pretrained('google/electra-small-discriminator', num_labels=len(label_names)).cuda()
    #rlm = AutoModelForSequenceClassification.from_pretrained('google/electra-small-discriminator', num_labels=len(label_names)).cuda()
elif args.pretrained_model == 'roberta':
    llm = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_names)).cuda()
    #rlm = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_names)).cuda()
else:
    print('no enable model')

num_label = len(label_names)
print('number of label', num_label)
model_params = list(llm.parameters())

if args.optim == 'adamw':
    optimizer = AdamW(model_params, lr = args.lr,  eps = 1e-8)
elif args.optim == 'adam':
    optimizer = torch.optim.Adam(model_params, lr=args.lr, betas=(0.9, 0.999))

total_steps = len(train_dataloader) * args.epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = args.warmup,
                                            num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


model_record_dir = './S_trained_analysis/ckpt{}/'.format(args.ckpt)
if not os.path.exists(model_record_dir):
    os.makedirs(model_record_dir)
recognized_test_acc_list =[]
recognized_test_acc_list =[]
all_epochs = []

for epoch_i in range(0, args.epochs):
    save_epoch = epoch_i+1
    save_model = os.path.join(model_record_dir,'epoch={}.hdf5'.format(save_epoch))
    all_epochs.append(save_epoch)
    # ========================================
    #               Training
    # ========================================
    
    #print("")
    print('======== Epoch {:} / {:} ========'.format(save_epoch, args.epochs))
    print('Training...')
        
    total_loss = 0.0
    
    c_correct = 0
    a_correct = 0
    
    llm.train()
    #rlm.train()    
    
    for step, (labelled, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        if step % 100 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        
        labelled_src = labelled[:, 0, :]
        if args.pretrained_model != 'roberta':
            labelled_segs = labelled[:, 1, :]
        labelled_mask = labelled[:, -1, :]
        
        
        labelled_src = labelled_src.cuda()
        if args.pretrained_model != 'roberta':
            labelled_segs = labelled_segs.cuda()
        labelled_mask = labelled_mask.cuda()
 
                
        labels = labels.cuda()
        
        if args.pretrained_model != 'roberta':
            llm_outputs = llm(labelled_src, token_type_ids=labelled_segs, attention_mask=labelled_mask, labels=labels)
        else:
            llm_outputs = llm(labelled_src, token_type_ids=None, attention_mask=labelled_mask, labels=labels)
            #rlm_outputs = rlm(recognized_src, token_type_ids=None, attention_mask=recognized_mask, labels=labels)
        
        #llm_loss = llm_outputs[0]
        #rlm_loss = rlm_outputs[0]
        loss = llm_outputs[0]
        #loss = (llm_loss * args.w1) + (rlm_loss * args.w2)
        
        #sum_loss_val = (llm_loss.item() * args.w1 + rlm_loss.item() * args.w2)
        
        total_loss += loss
        loss.backward()                
        torch.nn.utils.clip_grad_norm_(model_params, 1.0)
        
        optimizer.step()
        scheduler.step() 
        
                
    avg_train_loss = total_loss / len(train_dataloader)

    print(f'  Average training loss: {avg_train_loss:.2f}')
    
    writer.add_scalar('Loss/train', avg_train_loss, save_epoch)
    
    
              
    # ========================================
    #               Evaluation (labelled)
    # ========================================

    ## clean test 
    llm.eval()
    labelled_test_accuracy = 0
    labelled_targets_list = []
    labelled_preds_list = []
    
    with torch.no_grad():
    
        for data, labels in test_labelled_dataloader:
            
            src = data[:, 0, :]
            if args.pretrained_model != 'roberta':
                segs = data[:, 1, :]
            mask = data[:, -1, :]
    
            src = src.cuda()
            if args.pretrained_model != 'roberta':
                segs = segs.cuda()
            mask = mask.cuda()
            labels = labels.cuda()
    
            if args.pretrained_model != 'roberta':
                outputs = llm(src, token_type_ids=segs, attention_mask=mask)
            else:
                outputs = llm(src, token_type_ids=None, attention_mask=mask)
            
            targets = labels.detach().cpu().numpy()            
            preds = np.argmax(outputs.logits.detach().cpu().numpy(), axis = 1)
            acc = np.equal(targets, preds).sum()
            labelled_test_accuracy += acc
            labelled_targets_list.extend(labels.detach().cpu())
            labelled_preds_list.extend((outputs.logits.detach().cpu()).argmax(dim=1))
    
        
    labelled_test_acc = 100.*(labelled_test_accuracy/len(test_labelled_dataloader.dataset))
    print('Epoch %d (labelled Test) Acc %0.4f ' % (save_epoch, labelled_test_acc))
    writer.add_scalar('Accuracy_labelled/test', labelled_test_acc, save_epoch)
    
    labelled_test_stacked = torch.stack((torch.stack(labelled_targets_list, dim=0), torch.stack(labelled_preds_list, dim=0)), dim=1)
    
    labelled_cmt = torch.zeros(num_label,num_label, dtype=torch.int64)
    
    for p in labelled_test_stacked:
        tl, pl = p.tolist()
        labelled_cmt[tl, pl] = labelled_cmt[tl, pl] + 1
    
    
    plt.figure(figsize=(12,10))
    
    if args.dataset_name == 'all':
        plot_confusion_matrix(labelled_cmt.numpy(), label_names, title=r'Confusion Matrix of $S \rightarrow S$ (All of Speakers in {} Dataset)'.format(args.dataset_name2))
        plt.savefig(os.path.join(model_record_dir, 'epoch={},proposed->S,speaker=all'.format(save_epoch)))
    elif args.dataset_name is not None:
        plot_confusion_matrix(labelled_cmt.numpy(), label_names, title=r'Confusion Matrix of $S \rightarrow S$ ({} in {} Dataset)'.format(args.dataset_name, args.dataset_name2))
        plt.savefig(os.path.join(model_record_dir, 'epoch={},proposed->S,speaker={}'.format(save_epoch, args.dataset_name)))   
    else:
        plot_confusion_matrix(labelled_cmt.numpy(), label_names, title=r'Confusion Matrix of $S \rightarrow S$ (FSC Dataset)')
        plt.savefig(os.path.join(model_record_dir, 'epoch={},proposed->S,FSC'.format(save_epoch)))
    

    
    # ========================================
    #               Evaluation (recognized)
    # ========================================
    ## recognized test 
    llm.eval()
    recognized_test_accuracy = 0
    recognized_targets_list = []
    recognized_preds_list = []
        
    with torch.no_grad():
    
        for data, labels in test_recognized_dataloader:
            
            src = data[:, 0, :]
            if args.pretrained_model != 'roberta':
                segs = data[:, 1, :]
            mask = data[:, -1, :]
    
            src = src.cuda()
            if args.pretrained_model != 'roberta':
                segs = segs.cuda()
            mask = mask.cuda()
            labels = labels.cuda()
    
            if args.pretrained_model != 'roberta':
                outputs = llm(src, token_type_ids=segs, attention_mask=mask)
            else:
                outputs = llm(src, token_type_ids=None, attention_mask=mask)
            
            targets = labels.detach().cpu().numpy()
            preds = np.argmax(outputs.logits.detach().cpu().numpy(), axis = 1)
            acc = np.equal(targets, preds).sum()
            recognized_test_accuracy += acc
            recognized_targets_list.extend(labels.detach().cpu())
            recognized_preds_list.extend((outputs.logits.detach().cpu()).argmax(dim=1))
            
        
    recognized_test_acc = 100.*(recognized_test_accuracy/len(test_recognized_dataloader.dataset))
    print('Epoch %d (recognized Test) Acc %0.4f ' % (save_epoch, recognized_test_acc))
    writer.add_scalar('Accuracy_recognized/test', recognized_test_acc, save_epoch)
    
    recognized_test_stacked = torch.stack((torch.stack(recognized_targets_list, dim=0), torch.stack(recognized_preds_list, dim=0)), dim=1)
    
    recognized_cmt = torch.zeros(num_label,num_label, dtype=torch.int64)
    
    for p in recognized_test_stacked:
        tl, pl = p.tolist()
        recognized_cmt[tl, pl] = recognized_cmt[tl, pl] + 1
    
    
    plt.figure(figsize=(12,10))
    
    if args.dataset_name == 'all':
        plot_confusion_matrix(recognized_cmt.numpy(), label_names, title=r'Confusion Matrix of $S \rightarrow \hat{S}$'+' (All of Speakers in {} Dataset)'.format(args.dataset_name2))
        plt.savefig(os.path.join(model_record_dir, 'epoch={},proposed->S_hat,speaker=all'.format(save_epoch)))
    elif args.dataset_name is not None:
        plot_confusion_matrix(recognized_cmt.numpy(), label_names, title=r'Confusion Matrix of $S \rightarrow \hat{S}$'+' ({} in {} Dataset)'.format(args.dataset_name, args.dataset_name2))
        plt.savefig(os.path.join(model_record_dir, 'epoch={},proposed->S_hat,speaker={}'.format(save_epoch, args.dataset_name)))   
    else:
        plot_confusion_matrix(recognized_cmt.numpy(), label_names, title=r'Confusion Matrix of $S \rightarrow \hat{S}$'+' (FSC Dataset)')
        plt.savefig(os.path.join(model_record_dir, 'epoch={},proposed->S_hat,FSC'.format(save_epoch)))
    
    
    with open(os.path.join(model_record_dir, 's_to_hat_test_result_{}.csv'.format(save_epoch)), 'w') as ff:
        for i in range(len(recognized_test)):
            if label_names[recognized_targets_list[i]] != label_names[recognized_preds_list[i]]:
                for_write = '{}th labelled_sentences {} recognized_sentences {} label {} predict {}'.format(i, labelled_sentences[i], recognized_sentences[i], label_names[recognized_targets_list[i]], label_names[recognized_preds_list[i]]) 
                ff.write(for_write+'\n')
    
    
    with open(os.path.join(model_record_dir, 'test_result.txt'), 'a') as f:
        labelled_test_record_files = 'epoch={}, Labelled Test accuracy : {}\n'.format(save_epoch, labelled_test_acc)
        f.write(labelled_test_record_files)
        
        recognized_test_record_files = 'epoch={}, Recognized Test accuracy : {}\n'.format(save_epoch, recognized_test_acc)
        f.write(recognized_test_record_files)
    
    plt.close('all')
