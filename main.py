# coding=utf-8
import tensorflow as tf
from data import *
# from model import Model
from model import Model
from my_metrics import *
from tensorflow.python import debug as tf_debug
import numpy as np
import jieba
from flask import Flask
from flask import request
import json
import requests

#a = requests.post("http://52.224.222.139:80")

app = Flask(__name__)

input_steps = 50
embedding_size = 64
hidden_size = 150
n_layers = 2
batch_size = 1
vocab_size = 42452
slot_size = 16
intent_size = 14
epoch_num = 1

model_path = './model/ijia/'


def transform(inputs_strs):
    utterance_temp = inputs_strs.split( ' ' )

    # print("len utterance_temp: ", len(utterance_temp))

    slot_format = ""
    for i in range( 0, (len( utterance_temp ) - 1) ):
        slot_format = slot_format + "O "

    iob_format = inputs_strs + "\t" + slot_format + "<UNK>"
    # print("iob_format: ") + str(iob_format).decode('string_escape')
    # print ('iob_format: '+ iob_format)

    return iob_format

def get_model():
    model = Model( input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                   intent_size, epoch_num, batch_size, n_layers )
    model.build()

    return model

@app.route('/BuildModel', methods=['POST'])
def build_model():

    error_message = 'build model success'
    try:
        global model_run, sess
        model_run = get_model()
        sess = tf.Session()
        sess.run( tf.global_variables_initializer() )
    except:
        error_message = 'get model error'

    try:
        # JerryB +++ 建立 saver 物件
        saver = tf.train.Saver( tf.global_variables() )
        ckpt = tf.train.get_checkpoint_state( model_path )
        saver.restore( sess, ckpt.model_checkpoint_path )
        # JerryB ---
    except:
        error_message = 'get saver error'

    try:
        train_data = open( "dataset/ijia_dataset_musk.iob", "r" ).readlines()
        train_data_ed = data_pipeline( train_data )

        global word2index, index2word, slot2index, index2slot, intent2index, index2intent
        word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
            get_info_from_training_data( train_data_ed )
    except:
        error_message = 'get dataset error'

    return error_message

@app.route('/LoadDict', methods=['POST'])
def load_dict():
    try:
        jieba_dict = "./jieba-zh_TW/dict.txt"
        corpus_dict = "./jieba-zh_TW/corpus_jieba.txt"
        jieba.set_dictionary( jieba_dict )
        jieba.load_userdict( corpus_dict )
        error_message = "load dict seccess"
    except:
        error_message = "get something error when laod dict"
    return error_message

@app.route('/GetIntentEntity', methods=['POST'])
def get_intent_entity():
    #print(request)
    inputs_strs = request.form["input_str"]
    test_data = []
    action = False

    input_seg = ('BOS ' + ' '.join( jieba.cut( inputs_strs ) ) + ' EOS').encode( 'utf-8' )

    iob_format = transform( input_seg )
    test_data = []
    test_data.append( iob_format )

    test_data_ed = data_pipeline( test_data )

    index_test = to_index( test_data_ed, word2index, slot2index, intent2index )

    # 每训一个epoch，测试一次
    decoder_prediction, intent, intent_accuracy = model_run.step( sess, "test", index_test )
    decoder_prediction = np.transpose( decoder_prediction, [1, 0] )
    sen_len = index_test[0][1]


    Input_Sentence = " ".join(index_seq2word( index_test[0][0], index2word )[:sen_len])
    #Slot_Prediction = " ".join(index_seq2slot( decoder_prediction[0], index2slot )[:sen_len])
    Slot_Prediction = " ".join([tag.decode( "utf-8" ) for tag in index_seq2slot( decoder_prediction[0], index2slot )[:sen_len]])
    Intent_Prediction = str(index2intent[intent[0]]).decode("utf-8")

    return_dict = {'slot':Slot_Prediction,'intent':Intent_Prediction,'intent_acc':str(intent_accuracy).decode("utf-8")}
    return_form = json.dumps(return_dict,ensure_ascii=False)

    return return_form

@app.route('/')
def HelloWorld():
    return "Hello world !!!"

if __name__ == '__main__':
    # jieba dict
    app.run(host='0.0.0.0',port=80)
