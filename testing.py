import tensorflow as tf
import modeling as md
from transformers import BertTokenizer
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

# 데이터셋 불러와 확인
train = pandas.read_csv("sample_data.txt", sep='\t')
print(train.head(50))
# print(train.shape)

# Query와 cost value 분리하여 추출
query = train['query']
cost = train['costValue'].values
# print(query[:10])
# print(cost)

# BERT 입력 형식에 맞게 변환 & Tokenization
query = ["[CLS] " + str(q) + " [SEP]" for q in query]
tokenized_query= [Tokenizer.tokenize(q) for q in query]
print(tokenized_query)


# Training
