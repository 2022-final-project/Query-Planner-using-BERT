import tensorflow as tf
import modeling as md
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences
#from keras import pad_sequences
import pandas

# Config
customConfig=md.BertConfig(
               vocab_size=256,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02
)

# # Model
# customBert=md.BertModel(
#                 customConfig, 
#                 is_training=False, 
#                 input_ids=tf.Tensor(shape=(512,), dtype='int32')
#                 # input_ids=[5,10] # Batch size and seq_length
# )

# Tokenizer
Vocab = open("./vocab.txt", "r")
Tokenizer = BertTokenizer.from_pretrained("./vocab.txt")

# Dataset

# 데이터셋 불러와 확인 & Query와 cost value 분리하여 추출
train = pandas.read_csv("train_data.txt", sep='\t')
print(train.head(50))
# print(train.shape)
query = train['query']
cost = train['cost'].values
# print(query[:10])
# print(cost)

# BERT 입력 형식에 맞게 변환 & Tokenization
query = ["[CLS] " + str(q) + " [SEP]" for q in query]
tokenized_query= [Tokenizer.tokenize(q) for q in query]
print(tokenized_query)

# Token -> Index 변환 (input_ids)
MAX_LEN = 64 # 입력 토큰의 최대 시퀀스 길이
input_ids = [Tokenizer.convert_tokens_to_ids(x) for x in tokenized_query] # 토큰을 숫자 인덱스로 변환
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post") # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
print(input_ids)

# Training
