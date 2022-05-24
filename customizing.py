import modeling

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


customConfig=modeling.BertConfig(
               vocab_size=512,
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

customBert=modeling.BertModel(config=customConfig)

print("Done")