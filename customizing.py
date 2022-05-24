import tensorflow as tf
import modeling as md
from transformers import BertTokenizer
# 바꿔야 할 파라미터
# vocab size
# hidden size : Transformer 인코더 layer 개수 및 pooler layer 개수
# num_hidden_layers : Transformer 인코더에서 hidden layer의 개수
# num_attention_heads : Number of attention heads for each attention layer in
#        the Transformer encoder.
# # intermediate size : Transformer 인코더에서 intermediate layer의 개수
# max_position_embeddings : The maximum sequence length that this model might
#        ever be used with. Typically set this to something large just in case
#        (e.g., 512 or 1024 or 2048).
# pad_token_id : 어디?


Batch_size = 5
Seq_length = 10
Vocab_size = 256
Type_vocab_size = 128           # The vocabulary size of the `token_type_ids` passed into `BertModel`.
Use_input_mask = True
Use_token_type_ids = True
Is_training = False

Vocab = open("./vocab.txt", "r")
Tokenizer = BertTokenizer.from_pretrained("./vocab.txt")
sess = tf.Session()

Input_ids = None
Input_mask = None
Token_type_ids = None

Input_ids = md.ids_tensor([Batch_size, Seq_length], Vocab_size)

if Use_input_mask:
    Input_mask = md.ids_tensor([Batch_size, Seq_length], 2)

if Use_token_type_ids:
    Token_type_ids = md.ids_tensor([Batch_size, Seq_length], Type_vocab_size)

print("Input_ids :", sess.run(Input_ids))
print("Input_mask :", sess.run(Input_mask))
print("Token_type_ids :", sess.run(Token_type_ids))

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
customBert=md.BertModel(customConfig,
                Is_training,
                Input_ids,
                Input_mask,
                Token_type_ids
)

print("Model's all_encoder_layers :", Model.get_all_encoder_layers())

# outputs = {
#     "embedding_output":Model.get_embedding_output(),
#     "sequence_output":Model.get_sequence_output(),
#     "pooled_output":Model.get_pooled_output(),
#     "all_encoder_layers":Model.get_all_encoder_layers(),
# }