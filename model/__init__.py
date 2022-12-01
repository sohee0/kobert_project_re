import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import dill
import re
import transformers
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras import layers, initializers, losses, optimizers, metrics, callbacks
import tensorflow_addons as tfa

def emo_model(max_length=300):

    bert_base_model = TFBertModel.from_pretrained('klue/bert-base', from_pt=True) 

    input_token_ids   = layers.Input((max_length,), dtype=tf.int32, name='input_token_ids')   
    input_masks       = layers.Input((max_length,), dtype=tf.int32, name='input_masks')       
    input_segments    = layers.Input((max_length,), dtype=tf.int32, name='input_segments')      

    bert_outputs = bert_base_model([input_token_ids, input_masks, input_segments]) 
    

    bert_outputs = bert_outputs[1] 
    bert_outputs = layers.Dropout(0.5)(bert_outputs) # 과대적합 해결을 위해 0.2-> 0.5로 변경
    final_output = layers.Dense(units=6, activation='softmax', kernel_initializer=initializers.TruncatedNormal(stddev=0.02), name='classifier')(bert_outputs)
    # unit=6 <-6가지 감정

    model = tf.keras.Model(inputs=[input_token_ids, input_masks, input_segments], outputs=final_output)

    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=0.0001, weight_decay=0.0025, warmup_proportion=0.05)
    loss = losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer,
                  loss=loss, 
                  metrics=[metrics.SparseCategoricalAccuracy()])
    
    return model

# with open('./model/model_BERTfunction_fin.pkl', 'rb') as f:
#     emo_model = dill.loads(pickle.load(f)) # use dill to pickle a python function

model = emo_model(max_length=300)
model.load_weights(filepath='./model/best_bert_weights_v3.h5')
print( dir(model) )

# 2) Load the Bert-tokenizer 
# with open('./model/tokenizer-bert_fin.pkl', 'rb') as f:
    # tokenizer = pickle.load(f)

tokenizer = BertTokenizer.from_pretrained('klue/bert-base')


def df_save(fName, src_df):
  with open(fName, 'wb') as f:
    pickle.dump( src_df, f )

def df_load(fName):
  with open(fName, 'rb') as f:
    df = pickle.load( f )
  return df

def predict_sentiment(sentence):
    
    SEQ_LEN = 300 

    # Tokenizing / Tokens to sequence numbers / Padding
    encoded_dict = tokenizer.encode_plus(text=re.sub('[^\s0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]', '', sentence),
                                         padding='max_length', 
                                         truncation = True,
                                         max_length=SEQ_LEN) 
    
    token_ids = np.array(encoded_dict['input_ids']).reshape(1, -1) 
    token_masks = np.array(encoded_dict['attention_mask']).reshape(1, -1)
    token_segments = np.array(encoded_dict['token_type_ids']).reshape(1, -1)
    
    new_inputs = (token_ids, token_masks, token_segments)

    # Prediction
 
    prediction = model.predict(new_inputs)
    predicted_probability = np.round(np.max(prediction) * 100, 2) 
    predicted_class = ['분노', '혐오','놀람','공포','슬픔','행복'][np.argmax(prediction, axis=1)[0]] 
    
    print('{}% 확률로 {} 텍스트입니다.'.format(predicted_probability, predicted_class))

