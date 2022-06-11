
import tensorflow as tf
import torch

from transformers import pipeline
from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertForMultipleChoice
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

import pandas as pd
import numpy as np
import random
import time
import datetime
import random
import matplotlib.pyplot as plt

RAND_SEED = random.randint(1, 3000)
VALIDATION_RATE = 0.1

# train_data.txt 와 test_data.txt 를 읽어온다.
train_txt = open('./train_data.txt', 'r')
train = pd.read_csv(train_txt, sep='\t')

test_txt = open('./test_data.txt', 'r')
test = pd.read_csv(test_txt, sep='\t')

queries = train['query']    # train_data.txt 의 query 들 양 옆으로 "[CLS]", "[SEP]" 를 붙인다.
queries = ["[CLS] " + str(query[:-1]) + " [SEP]" for query in queries]

NUM_LABELS = 6

labels_before_Encoding = train['cost']     
labels = []

for cost in labels_before_Encoding:
    cost_str = str(cost)
    if len(cost_str) == 5: cost_str = "0" + cost_str
    elif len(cost_str) == 4 : cost_str = "00" + cost_str
    elif len(cost_str) == 3 : cost_str = "000" + cost_str
    elif len(cost_str) == 2 : cost_str = "0000" + cost_str
    elif len(cost_str) == 1 : cost_str = "00000" + cost_str
    elif len(cost_str) == 0 : cost_str = "000000"

    label_temp = [0, 0, 0, 0, 0, 0]

    if cost_str[0] == '1': label_temp[0] = 1
    if cost_str[1] == '1': label_temp[1] = 1
    if cost_str[2] == '1': label_temp[2] = 1
    if cost_str[3] == '1': label_temp[3] = 1
    if cost_str[4] == '1': label_temp[4] = 1
    if cost_str[5] == '1': label_temp[5] = 1

    labels.append(label_temp)

# ------------------------------------------- Data preProcessing -------------------------------------------
''' 1. Tokenizing
'''

# 구현된 vocab.txt 를 가지고 tokenizer 를 구현한다.
tokenizer = BertTokenizer.from_pretrained("./vocab.txt")
tokenized_queries = [tokenizer.tokenize(query) for query in queries]

MAX_LEN = 256

input_ids = [tokenizer.convert_tokens_to_ids(tokenized_query) for tokenized_query in tokenized_queries]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')

attention_masks = []

for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

# -------------------------------- train_data.txt 로 train/validation set 을 얻는다. --------------------------
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                    labels, 
                                                                                    random_state=RAND_SEED, 
                                                                                    test_size=VALIDATION_RATE)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, 
                                                       input_ids,
                                                       random_state=RAND_SEED, 
                                                       test_size=VALIDATION_RATE)
train_inputs = torch.tensor(train_inputs).float()
train_labels = torch.tensor(train_labels).float()
train_masks = torch.tensor(train_masks).float()
validation_inputs = torch.tensor(validation_inputs).float()
validation_labels = torch.tensor(validation_labels).float()
validation_masks = torch.tensor(validation_masks).float()          

BATCH_SIZE = 16

# 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
# 학습시 배치 사이즈 만큼 데이터를 가져옴
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

# ----------------------------- test_data.txt 에서 query, cost(label) data를 가져온다. -------------------------
test_queries = test['query']    # test_data.txt 의 query 들 양 옆으로 "[CLS]", "[SEP]" 를 붙인다.
test_queries = ["[CLS] " + str(query[:-1]) + " [SEP]" for query in test_queries]

test_labels = test['cost'].values
test_labels_temp = []

for cost in test_labels:
    cost_str = str(cost)
    if len(cost_str) == 5: cost_str = "0" + cost_str
    elif len(cost_str) == 4 : cost_str = "00" + cost_str
    elif len(cost_str) == 3 : cost_str = "000" + cost_str
    elif len(cost_str) == 2 : cost_str = "0000" + cost_str
    elif len(cost_str) == 1 : cost_str = "00000" + cost_str
    elif len(cost_str) == 0 : cost_str = "000000"

    labels = [0, 0, 0, 0, 0, 0]

    if cost_str[0] == 1: labels[0] = 1
    if cost_str[1] == 1: labels[1] = 1
    if cost_str[2] == 1: labels[2] = 1
    if cost_str[3] == 1: labels[3] = 1
    if cost_str[4] == 1: labels[4] = 1
    if cost_str[5] == 1: labels[5] = 1

    test_labels_temp.append(labels)

test_labels = test_labels_temp

# 모든 Test query 들을 Tokenize 한다.
tokenized_test_queries = [tokenizer.tokenize(query) for query in test_queries]
test_input_ids = [tokenizer.convert_tokens_to_ids(tokenized_query) for tokenized_query in tokenized_test_queries]
test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')

test_attention_masks = []

for seq in test_input_ids:
    seq_mask = [float(i > 0) for i in seq]
    test_attention_masks.append(seq_mask)

test_inputs = torch.tensor(test_input_ids).float()
test_labels = torch.tensor(test_labels).float()
test_masks = torch.tensor(test_attention_masks).float()
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

device = torch.device("cpu")

# ---------------------------------- model 생성 -------------------------------------

config = BertConfig.from_pretrained('bert-base-uncased', problem_type="regression")
config.num_labels = NUM_LABELS
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = NUM_LABELS)
# print(model.parameters) -> 확인 결과: (classifier): Linear(in_features=768, out_features=6, bias=True)

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # 학습률
                  eps = 1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값
                )

# 에폭수
EPOCHS = 20

# 총 훈련 스텝 : 배치반복 횟수 * 에폭
total_steps = len(train_dataloader) * EPOCHS

# 처음에 학습률을 조금씩 변화시키는 스케줄러 생성
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

# model_path = './test.model'
# checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
#                                          verbose=1, save_best_only=True)
# early_stopping = EarlyStopping(monitor='val_loss', patience=6)

# Accuracy 를 구하는 function 이다.\
def flat_accuracy(preds, labels):
    sze = len(labels)   # len(preds) == len(labels)

    cnt = 0
    total_cnt = 0

    for idx, pred in enumerate(preds):
        total_cnt += 1
        for i in range(0, 6, 1):
            if pred[i] < 0: pred[i] = 0.0
            else: pred[i] = 1.0
            if pred[i] == labels[idx][i] :
                cnt += 1
                break
            
    print(" acc : ", cnt / total_cnt)
    return cnt / 18

def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))

# ---------------------- 긴 modeling 부분 --------------------

# 재현을 위해 랜덤시드 고정
seed_val = RAND_SEED
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 그래디언트 초기화
model.zero_grad()

# 에폭만큼 반복
for epoch_i in range(0, EPOCHS):
    
    # ========================================
    #               Training
    # ========================================
    
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
    print('Training...')

    t0 = time.time()    # 시작 시간 설정
    total_loss = 0      # 로스 초기화
    model.train()       # 훈련모드로 변경
        
    # data_loader 에서 batch 만큼 반복하여 data를 가져온다.
    for step, batch in enumerate(train_dataloader):
        # 경과 정보 표시
        #if step % 500 == 0 and not step == 0:
        if True:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # 배치를 GPU에 넣음
        # batch = tuple(t.to(device) for t in batch)
        
        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = torch.tensor(b_input_ids).to(device).long()
        # b_labels = b_labels.squeeze(0)

        # print("batch :", batch)
        # print("b_input_ids :", b_input_ids, b_input_ids.shape)   # int64 torch.Size([4, 8])
        # print("b_input_mask: ", b_input_mask, b_input_mask.shape) # float32 torch.Size([4, 8])
        # print("labels :", b_labels, b_labels.shape) # float32 torch.Size([4])

        # Forward 수행                
        outputs = model(b_input_ids,
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)

        # print("outputs : ", outputs)  # Output: loss, logits, hidden_states, attentions

        loss = outputs[0]           # 로스 구함
        total_loss += loss.item()   # 총 로스 계산
        loss.backward()             # Backward 수행으로 그래디언트 계산

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 그래디언트 클리핑
        optimizer.step()                                        # 그래디언트를 통해 가중치 파라미터 업데이트
        scheduler.step()                                        # 스케줄러로 학습률 감소
        model.zero_grad()                                       # 그래디언트 초기화

    avg_train_loss = total_loss / len(train_dataloader)         # 평균 로스 계산

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    #시작 시간 설정
    t0 = time.time()

    # 평가모드로 변경
    model.eval()

    # 변수 초기화
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for batch in validation_dataloader:
        # 배치를 GPU에 넣음
        # batch = tuple(t.to(device) for t in batch)
        
        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = torch.tensor(b_input_ids).to(device).long()
        
        # 그래디언트 계산 안함
        with torch.no_grad():     
            # Forward 수행
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        # 출력 로짓 구함
        logits = outputs[0]

        # CPU로 데이터 이동
        # logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # 출력 로짓과 라벨을 비교하여 정확도 계산
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
        
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")

# ---------------------------------- 테스트셋 평가 -------------------------------------
print("Test set evaluation")
#시작 시간 설정
t0 = time.time()

# 평가모드로 변경
model.eval()

# 변수 초기화
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

# 데이터로더에서 배치만큼 반복하여 가져옴
for step, batch in enumerate(test_dataloader):
    # 경과 정보 표시
    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    # 배치를 CPU에 넣음
    batch = tuple(t.to(device) for t in batch)
    
    # 배치에서 데이터 추출
    b_input_ids, b_input_mask, b_labels = batch
    b_input_ids = torch.tensor(b_input_ids).to(device).long()
    
    # 그래디언트 계산 안함
    with torch.no_grad():     
        # Forward 수행
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
    
    # 출력 로짓 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    # 출력 로짓과 라벨을 비교하여 정확도 계산
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

print("")
print("Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
print("Test took: {:}".format(format_time(time.time() - t0)))
print("Test set evaluation complete!")

# ---------------------- 새로운 문장 테스트 --------------------
# 입력 데이터 변환
def convert_input_data(sentences):
    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 128

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    
    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # 어텐션 마스크 초기화
    attention_masks = []

    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # 데이터를 파이토치의 텐서로 변환
    inputs = torch.tensor(input_ids) # 데이터 타입: torch.int64
    masks = torch.tensor(attention_masks) # 데이터 타입: torch.float32

    return inputs, masks

    # 문장 테스트
def test_sentences(sentences):

    # 평가모드로 변경
    model.eval()
    # print("======== 최종 Result ========\n=== input data: ", sentences)

    # 문장을 입력 데이터로 변환
    inputs, masks = convert_input_data(sentences)
    # print("=== convert_input_data 결과 \ninputs: ", inputs, "\nmasks: ", masks)

    # 데이터를 GPU에 넣음
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
            
    # 그래디언트 계산 안함
    with torch.no_grad():     
        # Forward 수행
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
        # print("=== outputs: ", outputs)

    # 출력 로짓 구함
    logits = outputs[0]
    
    ret = [0, 0, 0, 0, 0, 0]
    for val in logits:
        if val[0] == 1: ret[0] = 1
        if val[1] == 1: ret[1] = 1
        if val[2] == 1: ret[2] = 1
        if val[3] == 1: ret[3] = 1
        if val[4] == 1: ret[4] = 1
        if val[5] == 1: ret[5] = 1

    print("=== result : ", ret)

# ——————————— 새로운 문장 테스트 입력 ——————————

logits_test1 = test_sentences(['select c17 c18 sum c13 sum c14 sum c14 c15 sum c14 c15 c16 avg c13 avg c14 avg c15 count from t2 where c19 date group by c17 c18 order by c17 c18'])
