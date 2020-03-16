from datetime import datetime

import os

from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from sklearn.metrics import f1_score
import numpy as np
#from pytorch_pretrained_bert.modeling import BertModel
import torch.nn as nn
import torch
import pickle
from tqdm import tqdm, trange 
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                           TensorDataset)
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMultipleChoice, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn import CrossEntropyLoss


#t1_num_labels = 2
#t2_num_labels = 4
#prepare classification data
def tokenize_sentence(sentence, max_seq_length):
    tokens = tokenizer.tokenize(sentence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = ids[:max_seq_length]
    segment_ids = [1] * len(ids)
    input_mask = [1] * len(ids)
    while len(ids) < max_seq_length:
        ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    return ids, segment_ids, input_mask
#OFF NOT TIN UNT GRP IND
def make_class_label(off, tin, t_type):
    assert off in ["OFF", "NOT"]
    assert tin in ["NULL", "TIN", "UNT"]
    assert t_type in ["NULL", "GRP", "IND", "OTH"]
    if off == "NOT":
        return 0
    else:
        if tin == "UNT":
            return 1
        elif tin == "TIN":
            if t_type == "IND":
                return 2
            elif t_type == "GRP":
                return 3
            elif t_type == "OTH":
                return 4
    raise
def make_class_label_2stage(off, tin, t_type):
    assert off in ["OFF", "NOT"]
    assert tin in ["NULL", "TIN", "UNT"]
    assert t_type in ["NULL", "GRP", "IND", "OTH"]
    if off == "NOT":
        return [0, -1]
    else:
        if tin == "UNT":
            return [1,0]
        elif tin == "TIN":
            if t_type == "IND":
                return [1,1]
            elif t_type == "GRP":
                return [1,2]
            elif t_type == "OTH":
                return [1,3]
    raise    

def make_class_label_3stage(off, tin, t_type):
    assert off in ["OFF", "NOT"]
    assert tin in ["NULL", "TIN", "UNT"]
    assert t_type in ["NULL", "GRP", "IND", "OTH"]
    if off == "NOT":
        return [0, -1, -1]
    else:
        if tin == "UNT":
            return [1, 0, -1]
        elif tin == "TIN":
            if t_type == "IND":
                return [1,1,0]
            elif t_type == "GRP":
                return [1,1,1]
            elif t_type == "OTH":
                return [1,1,2]
    raise  

from pytorch_pretrained_bert.modeling import BertModel
import torch.nn as nn

class BertForClassification(BertPreTrainedModel):
    def __init__(self, config, num_choices=2, output_attentions=False, keep_multihead_output=False):
        super(BertForClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_choices)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, head_mask=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        if self.output_attentions:
            all_attentions, _, pooled_output = outputs
        else:
            _, pooled_output = outputs
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
        elif self.output_attentions:
            return all_attentions, _logits
        return logits
    
#####train
#init classifier model
#cfg = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#t1_bert_classifier = BertForClassification(cfg, 2)
#t2_bert_classifier = BertForClassification(cfg, 4)
def init_bert(BertModel, bert_type, args, train_data, device, epochs=3, batch_size=32,
              BertSampler=RandomSampler, BertDataLoader=DataLoader):
    bert_model = BertModel.from_pretrained(bert_type, **args)
    bert_model.to(device)
    bert_model = torch.nn.DataParallel(bert_model)
    train_sampler = BertSampler(train_data)
    train_dataloader = BertDataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    num_train_optimization_steps = len(train_dataloader) * epochs
    param_optimizer = list(bert_model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, 
                         lr=lr,warmup=warmup_proportion, 
                         t_total=num_train_optimization_steps)
    return bert_model, train_sampler, train_dataloader, num_train_optimization_steps, optimizer

def train_bert(bert_model, train_dataloader, optimizer, tb_writer, loss_fct):
    global_step = 0
    tr_loss = 0
    print("Start training {}".format(datetime.now()))
    for epoch in trange(epochs, desc="Epoch"):
        nb_tr_examples, nb_tr_steps = 0, 0
        print("started epoch {}: {}".format(epoch, datetime.now()))
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=True)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            #input_ids, input_mask, segment_ids, label = batch
            logits = bert_model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            loss = loss_fct(logits, label_ids.view(-1))
            loss = loss.mean() # mean() to average on multi-gpu.

            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
            tb_writer.add_scalar('loss', loss.item(), global_step)
            
def train_bert_multitask(bert_models, train_dataloaders, optimizers, tb_writer, loss_fcts, task_weights=[1.0,1.0]):
    global_step = 0
    tr_loss = 0
    print("Start training {}".format(datetime.now()))
    for epoch in trange(epochs, desc="Epoch"):
        nb_tr_examples, nb_tr_steps = 0, 0
        print("started epoch {}: {}".format(epoch, datetime.now()))
        for bert_model, train_dataloader, optimizer, loss_fct, task_weight in zip(bert_models, train_dataloaders, optimizers, loss_fcts, task_weights):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=True)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                #input_ids, input_mask, segment_ids, label = batch
                #logits = bert_model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                #loss = loss_fct(logits, label_ids.view(-1))
                loss = bert_model(input_ids, segment_ids, input_mask, label_ids)
                loss = loss.mean() # mean() to average on multi-gpu.
                loss = loss * task_weight
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', loss.item(), global_step)
            
            
def dump_bert(output_dir, model):
    # If we save using the predefined names, we can load using `from_pretrained`
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    import os
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir,WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir,CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)
        

def prepare_test_model(BertModel, args, model_dir):
    saved_model = BertForClassification.from_pretrained(model_dir, **args)
    test_tokenizer = BertTokenizer.from_pretrained(model_dir)
    saved_model.to(device)
    return saved_model


'''
t1_saved_model = BertForClassification.from_pretrained(output_dir)
test_tokenizer = BertTokenizer.from_pretrained(bert_type, 'False')
t1_saved_model.to(device)
'''
#####evaluation

def make_test_features(rows):
    all_ids, all_segment_ids, all_input_mask, all_label = [],[],[],[]
    for row in rows:
        r_id, sentence, label = row
        ids, segment_ids, input_mask = tokenize_sentence(sentence, max_seq_length=max_seq_length)
        all_ids.append(ids)
        all_segment_ids.append(segment_ids)
        all_input_mask.append(input_mask)
        all_label.append(label) 
    return torch.tensor(all_ids, dtype=torch.long),torch.tensor(all_segment_ids, dtype=torch.long), torch.tensor(all_input_mask, dtype=torch.long), torch.tensor(all_label, dtype=torch.long)
def prepare_test_data(test_features, batch_size=32):
    #input: arrays of features
    test_data = TensorDataset(*test_features)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)
    return test_data, test_sampler, test_dataloader
def inference_test(model, dataloader):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    A_preds = []
    A_out_label_ids = None
    for input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        # create eval loss and other metric required by the task
        loss_fct = CrossEntropyLoss()
        #print(logits.shape)
        #print(logits.view(-1, num_labels).shape)
        #raise
        tmp_eval_loss = loss_fct(logits, label_ids.view(-1))


        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(A_preds) == 0:
            A_preds.append(logits.detach().cpu().numpy())
            A_out_label_ids = label_ids.detach().cpu().numpy()
        else:
            A_preds[0] = np.append(
                A_preds[0], logits.detach().cpu().numpy(), axis=0)
            A_out_label_ids = np.append(
                A_out_label_ids, label_ids.detach().cpu().numpy(), axis=0)
    A_preds = A_preds[0] 
    A_preds = np.argmax(A_preds, axis=1)
    return A_out_label_ids, A_preds


epochs = 3
lr = 5e-5
max_seq_length = 64
warmup_proportion = 0.1
cased = "uncased"
#bert_type = "bert-base-cased"
model_path = 'classification_models'
exp_name = "single111"

t1_BertModel = BertForClassification
t1_args = {'num_choices':2}
t1_output_dir = '{}/t1_{}/{}_{}_{}_{}finetune'.format(model_path, exp_name, lr, epochs, max_seq_length, cased)
t1_loss_fct = CrossEntropyLoss()
if not os.path.exists(t1_output_dir):
    os.makedirs(t1_output_dir)


t2_BertModel = BertForClassification
t2_args = {'num_choices':2}
t2_output_dir = '{}/t2_{}/{}_{}_{}_{}finetune'.format(model_path, exp_name, lr, epochs, max_seq_length, cased)
t2_loss_fct = CrossEntropyLoss(weight=torch.Tensor([0.88, 0.12]).cuda())
if not os.path.exists(t2_output_dir):
    os.makedirs(t2_output_dir)

t3_BertModel = BertForClassification
t3_args = {'num_choices':3}
t3_output_dir = '{}/t3_{}/{}_{}_{}_{}finetune'.format(model_path, exp_name, lr, epochs, max_seq_length, cased)
t3_loss_fct = CrossEntropyLoss()
if not os.path.exists(t3_output_dir):
    os.makedirs(t3_output_dir)

#t1_BertModel, t2_BertModel, t3BertModel = BertForClassification, BertForClassification
#bert_type inited
#t1_args, t2_args = {'num_choices':2}, {'num_choices':4}
#t1_single_directory = 'classification_models/t1_single/{}_{}_{}_{}finetune'.format(lr, epochs, max_seq_length, cased)
#t2_single_directory = 
#output_dir = t2_single_directory
tb_writer = SummaryWriter()
bert_type = "bert-base-{}".format(cased)
tokenizer = BertTokenizer.from_pretrained(bert_type, 'False')
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
n_gpu = torch.cuda.device_count()

print("Program started {}".format(datetime.now()))

import pickle
with open ('classification/train.pickle', 'rb') as f:
    train = pickle.load(f)
with open('classification/test.pickle', 'rb') as f:
    test = pickle.load(f)

t1_all_ids, t1_all_segment_ids, t1_all_input_mask, t1_all_label = [],[],[],[]
t2_all_ids, t2_all_segment_ids, t2_all_input_mask, t2_all_label = [],[],[],[]
t3_all_ids, t3_all_segment_ids, t3_all_input_mask, t3_all_label = [],[],[],[]

for row in train:
    r_id, sentence, off, tin, t_type = row
    ids, segment_ids, input_mask = tokenize_sentence(sentence, max_seq_length)
    label_1, label_2, label_3 = make_class_label_3stage(off, tin, t_type)
    t1_all_ids.append(ids)
    t1_all_segment_ids.append(segment_ids)
    t1_all_input_mask.append(input_mask)
    t1_all_label.append(label_1)
    if label_1==1:
        t2_all_ids.append(ids)
        t2_all_segment_ids.append(segment_ids)
        t2_all_input_mask.append(input_mask)
        t2_all_label.append(label_2)
        if label_2 == 1:
            t3_all_ids.append(ids)
            t3_all_segment_ids.append(segment_ids)
            t3_all_input_mask.append(input_mask)
            t3_all_label.append(label_3)
        
t1_all_ids = torch.tensor([f for f in t1_all_ids], dtype=torch.long)
t1_all_input_mask = torch.tensor([f for f in t1_all_input_mask], dtype=torch.long)
t1_all_segment_ids = torch.tensor([f for f in t1_all_segment_ids], dtype=torch.long)
t1_all_label = torch.tensor(t1_all_label, dtype=torch.long)
t1_train_data = TensorDataset(t1_all_ids, t1_all_input_mask, t1_all_segment_ids, t1_all_label)

t2_all_ids = torch.tensor([f for f in t2_all_ids], dtype=torch.long)
t2_all_input_mask = torch.tensor([f for f in t2_all_input_mask], dtype=torch.long)
t2_all_segment_ids = torch.tensor([f for f in t2_all_segment_ids], dtype=torch.long)
t2_all_label = torch.tensor(t2_all_label, dtype=torch.long)
t2_train_data = TensorDataset(t2_all_ids, t2_all_input_mask, t2_all_segment_ids, t2_all_label)

t3_all_ids = torch.tensor([f for f in t3_all_ids], dtype=torch.long)
t3_all_input_mask = torch.tensor([f for f in t3_all_input_mask], dtype=torch.long)
t3_all_segment_ids = torch.tensor([f for f in t3_all_segment_ids], dtype=torch.long)
t3_all_label = torch.tensor(t3_all_label, dtype=torch.long)
t3_train_data = TensorDataset(t3_all_ids, t3_all_input_mask, t3_all_segment_ids, t3_all_label)

#train single t1
t1_bert_model, t1_train_sampler, t1_train_dataloader, t1_num_train_optimization_steps, t1_optimizer =\
init_bert(t1_BertModel, bert_type, t1_args, t1_train_data, device)
train_bert(t1_bert_model, t1_train_dataloader, t1_optimizer, tb_writer, t1_loss_fct)
dump_bert(t1_output_dir, t1_bert_model)
#train single t2
t2_bert_model, t2_train_sampler, t2_train_dataloader, t2_num_train_optimization_steps, t2_optimizer =\
init_bert(t2_BertModel, bert_type, t2_args, t2_train_data, device)
train_bert(t2_bert_model, t2_train_dataloader, t2_optimizer, tb_writer, t2_loss_fct)
dump_bert(t2_output_dir, t2_bert_model)

#train single t3
t3_bert_model, t3_train_sampler, t3_train_dataloader, t3_num_train_optimization_steps, t3_optimizer =\
init_bert(t3_BertModel, bert_type, t3_args, t3_train_data, device)
train_bert(t3_bert_model, t3_train_dataloader, t3_optimizer, tb_writer, t3_loss_fct)
dump_bert(t3_output_dir, t3_bert_model)

#train multitask t3
t3_bert_model, t3_train_sampler, t3_train_dataloader, t3_num_train_optimization_steps, t3_optimizer =\
init_bert(t3_BertModel, bert_type, t3_args, t3_train_data, device)

#multitask
from pytorch_pretrained_bert import BertForTokenClassification
ner_BertModel = BertForTokenClassification
ner_args = {'num_labels':13}
ner_train_data = torch.load("CONLL2003/NER.pt")

ner_bert_model, ner_train_sampler, ner_train_dataloader, ner_num_train_optimization_steps, ner_optimizer =\
init_bert(ner_BertModel, bert_type, ner_args, ner_train_data, device)

share = 9 #best among 0,3,6,9,12
#ner_bert_model.module.bert = None
ner_bert_model.module.bert.embeddings = t3_bert_model.module.bert.embeddings
for i in range(share):
    ner_bert_model.module.bert.encoder.layer[i] = t3_bert_model.module.bert.encoder.layer[i]
#assert ner_bert_model.module.bert is t3_bert_model.module.bert



ner_loss_fct = CrossEntropyLoss()
train_bert_multitask([ner_bert_model, t3_bert_model], 
                     [ner_train_dataloader, t3_train_dataloader], 
                     [ner_optimizer, t3_optimizer], 
                     tb_writer, 
                     [ner_loss_fct, t3_loss_fct])
t3_output_dir = '{}/t3_{}/{}_{}_{}_{}_share{}_finetune'.format(model_path, 'mtner', lr, epochs, max_seq_length, cased, share)
if not os.path.exists(t3_output_dir):
    os.makedirs(t3_output_dir)
dump_bert(t3_output_dir, t3_bert_model)