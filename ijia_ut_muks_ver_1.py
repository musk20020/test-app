# coding=utf-8
import tensorflow as tf
from data import *
# from model import Model
from model import Model
from model_musk_ver_1 import Model_musk
from my_metrics import *
from tensorflow.python import debug as tf_debug
import numpy as np
# import jieba
import xlrd,xlwt

input_steps = 50
embedding_size = 4809
#embedding_size = 64
hidden_size = 150
n_layers = 2
batch_size = 1

'''
vocab_size = 780
slot_size = 21
intent_size = 19
'''

# after embedding
vocab_size = 42452
slot_size = 16
intent_size = 14
epoch_num = 1

model_path = './model/ijia/'

def transform(inputs_strs):
    utterance_temp = inputs_strs.split(' ')

    #print("len utterance_temp: ", len(utterance_temp))

    slot_format = ""
    for i in range(0, (len(utterance_temp)-1)):
        slot_format = slot_format + "O "

    iob_format = inputs_strs + "\t" + slot_format + "<UNK>"
    # print("iob_format: ") + str(iob_format).decode('string_escape')
    #print ('iob_format: '+ iob_format)

    return iob_format

def get_model():
    model = Model_musk(input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, batch_size, n_layers)
    model.build()
    return model

def infer():
    model = get_model()
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    #JerryB +++ 建立 saver 物件
    saver = tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt is not None:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("没找到模型")
    #JerryB ---

    test_data = []
    action = False


    #train_data = open("dataset/ijia_dataset.iob", "r").readlines()
    #test_data = open("dataset/ijia_primiry130.iob", "r").readlines()
    train_data = open( "dataset/ijia_dataset_musk.iob", "r" ).readlines()
    test_data = open( "dataset/ijia_primiry130_musk.iob", "r" ).readlines()

    #print("JerryB train_data[0]: ", train_data[0])
    #print("JerryB train_data[1]: ", train_data[1])
    #print("JerryB train_data[2]: ", train_data[2])
    train_data_ed = data_pipeline(train_data)
    test_data_ed = data_pipeline(test_data)
    # print("test_data_ed[0]: ") + str(test_data_ed[0]).decode('string_escape')
    word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
        get_info_from_training_data(train_data_ed)
    #print("JerryB word2index: ", word2index)
    #print("JerryB index2word: ", index2word)
    #print("JerryB slot2index: ", slot2index)
    #print("JerryB index2slot: ", index2slot)
    #print("JerryB intent2index: ", intent2index)
    #print("JerryB index2intent: ", index2intent)
    #index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
    #index_test = to_index(test_data_ed, word2index, slot2index, intent2index)
    # print ('len(index_test):' + str(len(index_test)))
    #print("JerryB index_train[0]: ", index_train[0])
    #print("JerryB index_train[1]: ", index_train[1])
    #print("JerryB index_train[2]: ", index_train[2])
    #print("JerryB index_test[0]: ", index_test[0])

    #indexxxx = np.concatenate((index_train,index_test))


    pred_slots = []
    slot_accs = []
    intent_accs = []


    # prepare workbook
    fail_workbook = xlwt.Workbook()
    wt_sheet = fail_workbook.add_sheet('fails')
    wt_style = xlwt.XFStyle()
    wt_style.alignment.wrap = 1
    total_fail_item_count = 1

    # write header
    wt_sheet.write(0, 0, "ID")
    wt_sheet.write(0, 1, "Utterance")
    wt_sheet.write(0, 2, "Correct Intent")
    wt_sheet.write(0, 3, "Correct Entities")
    wt_sheet.write(0, 4, "Err Intent")
    wt_sheet.write(0, 5, "Err Entities")

    intnet_err_count = 0
    slot_err_count = 0

    # Musk_embedding +++
    word_list = pd.read_csv( "word_list.csv" )
    word_dic = {}
    num = 0
    # a = word_list.iloc[:,3]
    for i in word_list.iloc[:, 3]:
        word_dic[i.decode( "utf-8" )] = num
        num += 1
    word_dic[u"non"] = num

    index_train = to_index_musk_ver_1( train_data_ed, word2index, slot2index, intent2index )
    index_test = to_index_musk_ver_1( test_data_ed, word2index, slot2index, intent2index )
    # Musk_embedding ---

    count = 0
    #embed_npy_musk = np.load( 'embedding_4809.npy' )
    #embeddings = embed_npy_musk.astype( np.float32 )

    for j, batch in enumerate(getBatchUT(batch_size, index_test)):
        # print (batch)
        # print (type(j))

        decoder_prediction, intent, intent_accuracy = model.step(sess, "test", batch, word_dic)
        #prediction_confidence = tf.Variable.eval(prediction_confidence)
        decoder_prediction = np.transpose(decoder_prediction, [1, 0])
        count += 1

        index = 0   # since batch size is 1
        sen_len = batch[index][1]
        input_sentence = " ".join(batch[index][0][:sen_len])
        #input_sentence  = str(index_seq2word(batch[index][0], index2word)[:sen_len]).decode('string_escape')
        slot_truth      = str(index_seq2slot(batch[index][2], index2slot)[:sen_len]).decode('string_escape')
        slot_prediction = str(index_seq2slot(decoder_prediction[index], index2slot)[:sen_len]).decode('string_escape')
        intent_truth    = str(index2intent[batch[index][3]]).decode('string_escape')
        intent_prediction = str(index2intent[intent[index]]).decode('string_escape')
        print ("Input Sentence        : ") + input_sentence
        print ("Slot Truth            : ") + slot_truth
        print ("Slot Prediction       : ") + slot_prediction
        print ("Intent Truth          : ") + intent_truth
        print ("Intent Prediction     : ") + intent_prediction
        # print("Input Sentence        : ", index_seq2word(batch[index][0], index2word)[:sen_len])
        #print("Slot Truth            : ", index_seq2slot(batch[index][2], index2slot)[:sen_len])
        #print("Slot Prediction       : ", index_seq2slot(decoder_prediction[index], index2slot)[:sen_len])
        #print("Intent Truth          : ", index2intent[batch[index][3]])
        #print("Intent Prediction     : ", index2intent[intent[index]])
        slot_pred_length = list(np.shape(decoder_prediction))[1]
        pred_padded = np.lib.pad(decoder_prediction, ((0, 0), (0, input_steps-slot_pred_length)),
                                 mode="constant", constant_values=0)
        pred_slots.append(pred_padded)
        # print("slot_pred_length: ", slot_pred_length)
        true_slot = np.array((list(zip(*batch))[2]))
        true_length = np.array((list(zip(*batch))[1]))
        true_slot = true_slot[:, :slot_pred_length]
        # print(np.shape(true_slot), np.shape(decoder_prediction))
        # print(true_slot, decoder_prediction)
        slot_acc = accuracy_score(true_slot, decoder_prediction, true_length)
        intent_acc = accuracy_score(list(zip(*batch))[3], intent)
        print("slot accuracy: {}, intent accuracy: {}".format(slot_acc, intent_acc))

        if intent_acc < 1.0:
            wt_sheet.write(total_fail_item_count, 0, str(j+1))   # ID
            wt_sheet.write(total_fail_item_count, 1, input_sentence.decode('utf-8'))   # segmented sentence
            wt_sheet.write(total_fail_item_count, 2, intent_truth.decode('utf-8'), wt_style)   # right intent
            wt_sheet.write(total_fail_item_count, 4, intent_prediction.decode('utf-8') + "(" +  str(round(intent_accuracy, 3)) + ")", wt_style)   # wrong intent
            total_fail_item_count +=1
            intnet_err_count += 1

        if slot_acc < 1.0:
            wt_sheet.write(total_fail_item_count, 0, str(j+1))
            wt_sheet.write(total_fail_item_count, 1, input_sentence.decode('utf-8'))   # sentence
            wt_sheet.write(total_fail_item_count, 3, slot_truth.decode('utf-8'), wt_style)   # right entity
            wt_sheet.write(total_fail_item_count, 5, slot_prediction.decode('utf-8'), wt_style)   # wrong entity
            total_fail_item_count +=1
            slot_err_count += 1

        '''
        if intent_accuracy <= 1:
            wt_sheet.write( total_fail_item_count, 0, str( j + 1 ) )  # ID
            wt_sheet.write( total_fail_item_count, 1, input_sentence.decode( 'utf-8' ) )  # segmented sentence
            #wt_sheet.write( total_fail_item_count, 2, intent_truth.decode( 'utf-8' ), wt_style )  # right intent
            wt_sheet.write( total_fail_item_count, 4,
                            intent_prediction.decode( 'utf-8' ) + "(" + str( round( intent_accuracy, 3 ) ) + ")",
                            wt_style )  # wrong intent
            total_fail_item_count += 1
            intnet_err_count += 1
        '''

        slot_accs.append(slot_acc)
        intent_accs.append(intent_acc)
        fail_workbook.save('UT_result.xls')
        print ('processed sentence #: ' + str(count))
    pred_slots_a = np.vstack(pred_slots)
    print("pred_slots_a: ", pred_slots_a.shape)
    true_slots_a = np.array(list(zip(*index_test))[2])[:pred_slots_a.shape[0]]
    print("true_slots_a: ", true_slots_a.shape)
    print("Intent accuracy for epoch 0: {}".format(np.average(intent_accs)))
    print("Slot accuracy for epoch 0: {}".format(np.average(slot_accs)))
    print("Slot F1 score for epoch 0: {}".format(f1_for_sequence_batch(true_slots_a, pred_slots_a)))

    print('Intent error count: {}'.format(intnet_err_count))
    print('Slot error count: {}'.format(slot_err_count))



if __name__ == '__main__':
    # jieba dict
    # jieba_dict = "./jieba-zh_TW/dict.txt"
    # corpus_dict = "./jieba-zh_TW/corpus_jieba.txt"
    # jieba.set_dictionary(jieba_dict)
    # jieba.load_userdict(corpus_dict)
    infer()