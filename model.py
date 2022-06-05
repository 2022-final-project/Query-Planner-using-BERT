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

# ---------------------- 새로운 테스트 쿼리문 입력 --------------------

test_query1=(['SELECT T3, T4 FROM T1;'])
test_query2=(['select c26 sum c14 c15 from t1 t4 t2 t8 t3 t7 where c1 c30 and c9 c29 and c11 c47 and c4 c50 and c50 c25 and c27 c59 and c60 and c33 date and c33 date year group by c26 order by desc;'])
test_query3=(['select c6 c5 from t3 where c1 and c2;'])

# ---------------------- Modeling --------------------

RAND_SEED = random.randint(1, 3000)
train_txt = open('./train_data.txt', 'r')
train = pd.read_csv(train_txt, sep='\t')
test_txt = open('./test_data.txt', 'r')
test = pd.read_csv(test_txt, sep='\t')

print("train data: \n", train)

print("[train shape]\n", train.shape)
print("[test shape]\n", test.shape)

queries = train['query']
queries = ["[CLS] " + str(query[:-1]) + " [SEP]" for query in queries]
print("Queries: \n", queries)

NUM_LABELS=6

labels_before_Encoding = train['cost']    
labels=[]
for cost in labels_before_Encoding:
    cost_str = str(cost)
    if len(cost_str) == 5: cost_str = "0" + cost_str
    elif len(cost_str) == 4 : cost_str = "00" + cost_str
    elif len(cost_str) == 3 : cost_str = "000" + cost_str
    elif len(cost_str) == 2 : cost_str = "0000" + cost_str
    elif len(cost_str) == 1 : cost_str = "00000" + cost_str
    elif len(cost_str) == 0 : cost_str = "000000"

    print(cost_str)

    label_temp = [0, 0, 0, 0, 0, 0]

    if cost_str[0] == '1': label_temp[0] = 1
    if cost_str[1] == '1': label_temp[1] = 1
    if cost_str[2] == '1': label_temp[2] = 1
    if cost_str[3] == '1': label_temp[3] = 1
    if cost_str[4] == '1': label_temp[4] = 1
    if cost_str[5] == '1': label_temp[5] = 1

    labels.append(label_temp)

print(labels)


tokenizer = BertTokenizer.from_pretrained("./vocab.txt")
tokenized_queries = [tokenizer.tokenize(query) for query in queries]

for i, x in enumerate(tokenized_queries):
    print(i, " : ", x)

max_len = 8

input_ids = [tokenizer.convert_tokens_to_ids(tokenized_query) for tokenized_query in tokenized_queries]
print("input ids\n", input_ids)
input_ids = pad_sequences(input_ids, maxlen=max_len, dtype='long', truncating='post', padding='post')

print("input ids[0]: ", input_ids[0])

attention_masks = []

for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

print("attention_masks[0]: ", attention_masks[0])

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                    labels, 
                                                                                    random_state=2018, 
                                                                                    test_size=0.1)

print("train labels", train_labels)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, 
                                                       input_ids,
                                                       random_state=2018, 
                                                       test_size=0.1)

train_inputs = torch.tensor(train_inputs).float()
train_labels = torch.tensor(train_labels).float()
train_masks = torch.tensor(train_masks).float()
validation_inputs = torch.tensor(validation_inputs).float()
validation_labels = torch.tensor(validation_labels).float()
validation_masks = torch.tensor(validation_masks).float()            

print("train_inputs[0]: ", train_inputs[0])
print("train_labels[0]: ", train_labels[0])
print("train_masks[0]: ", train_masks[0])
print("validation_inputs[0]: ", validation_inputs[0])
print("validation_labels[0]: ", validation_labels[0])
print("validation_masks[0]: ", validation_masks[0])

batch_size = 4

# 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
# 학습시 배치 사이즈 만큼 데이터를 가져옴
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

test_queries = test['query']

test_queries = ["[CLS] " + str(query[:-1]) + " [SEP]" for query in test_queries]
print("test queries: ", test_queries)

test_labels = test['cost'].values
print("test_labels ", test_labels)

tokenized_test_queries = [tokenizer.tokenize(query) for query in test_queries]

test_input_ids = [tokenizer.convert_tokens_to_ids(tokenized_query) for tokenized_query in tokenized_test_queries]
print("input ids\n", test_input_ids)
test_input_ids = pad_sequences(test_input_ids, maxlen=max_len, dtype='long', truncating='post', padding='post')

test_attention_masks = []

for seq in test_input_ids:
    seq_mask = [float(i > 0) for i in seq]
    test_attention_masks.append(seq_mask)

test_inputs = torch.tensor(test_input_ids).float()
test_labels = torch.tensor(test_labels).float()
test_masks = torch.tensor(test_attention_masks).float()

print("test_inputs[0]: ", test_inputs[0])
print("test_labels[0]: ", test_labels[0])
print("test_masks[0]: ", test_masks[0])

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

device = torch.device("cpu")

# ---------------------------------- model 생성 -------------------------------------

config = BertConfig.from_pretrained('bert-base-uncased', problem_type="regression")
config.num_labels = NUM_LABELS
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=NUM_LABELS)
# print(model.parameters) -> 확인 결과: (classifier): Linear(in_features=768, out_features=6, bias=True)

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # 학습률
                  eps = 1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값
                )

# 에폭수
epochs = 4

# 총 훈련 스텝 : 배치반복 횟수 * 에폭
total_steps = len(train_dataloader) * epochs

# 처음에 학습률을 조금씩 변화시키는 스케줄러 생성
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

print("##### check A")

# model_path = './test.model'
# checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
#                                          verbose=1, save_best_only=True)
# early_stopping = EarlyStopping(monitor='val_loss', patience=6)

def flat_accuracy(preds, labels):
    sze = len(labels)   # len(preds) == len(labels)

    cnt = 0

    for idx, pred in enumerate(preds):
        for i in range(0, 6, 1):
            if pred[i] < 0: pred[i] = 0.0
            else: pred[i] = 1.0
            if pred[i] == labels[idx][i] : cnt += 1

    print("preds :", preds)
    print("labels :", labels)
    print(" acc : ", cnt / 18)

    return cnt / 18

def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))

# ---------------------- 긴 modeling 부분 --------------------

# 재현을 위해 랜덤시드 고정
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 그래디언트 초기화
model.zero_grad()

# 에폭만큼 반복
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # 시작 시간 설정
    t0 = time.time()

    # 로스 초기화
    total_loss = 0

    # 훈련모드로 변경
    model.train()
        
    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(train_dataloader):
        # 경과 정보 표시
        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # 배치를 GPU에 넣음
        # batch = tuple(t.to(device) for t in batch)
        
        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = torch.tensor(b_input_ids).to(device).long()
        # b_labels = b_labels.squeeze(0)

        print("batch :", batch)
        print("b_input_ids :", b_input_ids, b_input_ids.shape)   # int64 torch.Size([4, 8])
        print("b_input_mask: ", b_input_mask, b_input_mask.shape) # float32 torch.Size([4, 8])
        print("labels :", b_labels, b_labels.shape) # float32 torch.Size([4])

        # Forward 수행                
        outputs = model(b_input_ids,
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)

        print("outputs : ", outputs)  # Output: loss, logits, hidden_states, attentions

        
        # 로스 구함
        loss = outputs[0]
        print("loss: ",loss)

        # 총 로스 계산
        total_loss += loss.item()

        # Backward 수행으로 그래디언트 계산
        loss.backward()

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 그래디언트를 통해 가중치 파라미터 업데이트
        optimizer.step()

        # 스케줄러로 학습률 감소
        scheduler.step()

        # 그래디언트 초기화
        model.zero_grad()

    # 평균 로스 계산
    avg_train_loss = total_loss / len(train_dataloader)            
      
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
        logits = logits.detach().cpu().numpy()
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
    print("======== 최종 Result ========\n=== input data: ", sentences)

    # 문장을 입력 데이터로 변환
    inputs, masks = convert_input_data(sentences)
    print("=== convert_input_data 결과 \ninputs: ", inputs, "\nmasks: ", masks)

    # 데이터를 GPU에 넣음
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
            
    # 그래디언트 계산 안함
    with torch.no_grad():     
        # Forward 수행
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
        print("=== outputs: ", outputs)

    # 출력 로짓 구함
    logits = outputs[0]
    print("=== logits: ", logits)

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()
    if np.argmax(logits)==0: result=1
    elif np.argmax(logits)==1: result=2
    elif np.argmax(logits)==2: result=4
    elif np.argmax(logits)==3: result=8
    elif np.argmax(logits)==4: result=16
    elif np.argmax(logits)==5: result=32
    
    print("=== result: ", result)
    return result


# ---------------------- 새로운 쿼리문 테스트 결과 출력 --------------------

print(test_sentences(test_query1))
print(test_sentences(test_query2))
print(test_sentences(test_query3))