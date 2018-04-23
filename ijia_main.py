# coding=utf-8
import tensorflow as tf
from data import *
# from model import Model
from model import Model
from model_musk import Model_musk
from my_metrics import *
from tensorflow.python import debug as tf_debug
import numpy as np
import random

input_steps = 50
#要連動data_pipeline後面的參數
#embedding_size = 4809
embedding_size = 64
hidden_size = 150
n_layers = 2
batch_size = 16
#vocab_size = 871
#slot_size = 122
#intent_size = 22
#epoch_num = 50

# ijia chinese version
# vocab_size = 780
# slot_size = 21
# intent_size = 19
# epoch_num = 100

# after embedding
#vocab_size = 92783 ---janet
vocab_size = 42452 #42367
slot_size = 16
intent_size = 14
epoch_num = 41

def get_model():
    model = Model(input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, batch_size, n_layers)
    model.build()
    return model


def train(is_debug=False):
    model = get_model()
    sess = tf.Session()
    if is_debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    sess.run(tf.global_variables_initializer())
    #JerryB +++ 建立 saver 物件
    saver = tf.train.Saver(tf.global_variables())
    #JerryB ---
    train_data = open("dataset/ijia_dataset_musk.iob", "r").readlines()
    test_data = open("dataset/ijia_dataset_musk.iob", "r").readlines()
    #train_data_othertype = open("dataset/ijia_dataset_othertype.iob").readlines()

    train_data_ed = data_pipeline(train_data)
    test_data_ed = data_pipeline(test_data)
    #train_data_othertype_ed = data_pipeline( train_data_othertype )
    word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
        get_info_from_training_data(train_data_ed)

    print("len word2index: ", len(word2index))
    print("len slot2index: ", len(slot2index))
    print("len intent2index: ", len(intent2index))

    # Musk_embedding +++
    index_train = to_index( train_data_ed, word2index, slot2index, intent2index )
    index_test = to_index( test_data_ed, word2index, slot2index, intent2index )
    #index_train = to_index_musk(train_data_ed, word2index, slot2index, intent2index)
    #index_test = to_index_musk(test_data_ed, word2index, slot2index, intent2index)
    # Musk_embedding ---

    #embed_npy_musk = np.load( 'embedding_4809.npy' )
    #embeddings = embed_npy_musk.astype( np.float32 )

    #word2index_file = open( 'word2index_calibrated.txt', 'r' )
    #vocab_size = len( word2index_file.readlines())
    #word2index_file.close

    for epoch in range(epoch_num):
        mean_loss = 0.0
        train_loss = 0.0

        '''
        if eindex < len( index_train ):
            batch = index_train[sindex:eindex]
            temp = eindex
            eindex = eindex + epoch_size
            sindex = temp
        else:
            batch = index_train[sindex:-1]
            sindex = 0
            eindex = 2000
    
        index_input = np.concatenate((batch,index_train_othertype))
        '''

        for i, batch in enumerate(getBatch(batch_size, index_train)):
            # 执行一个batch的训练
            _, loss, decoder_prediction, intent, mask, slot_W = model.step(sess, "train", batch)
            # if i == 0:
            #     index = 0
            #     print("training debug:")
            #     print("input:", list(zip(*batch))[0][index])
            #     print("length:", list(zip(*batch))[1][index])
            #     print("mask:", mask[index])
            #     print("target:", list(zip(*batch))[2][index])
            #     # print("decoder_targets_one_hot:")
            #     # for one in decoder_targets_one_hot[index]:
            #     #     print(" ".join(map(str, one)))
            #     print("decoder_logits: ")
            #     for one in decoder_logits[index]:
            #         print(" ".join(map(str, one)))
            #     print("slot_W:", slot_W)
            #     print("decoder_prediction:", decoder_prediction[index])
            #     print("intent:", list(zip(*batch))[3][index])
            mean_loss += loss
            train_loss += loss
            if i % 10 == 0:
                if i > 0:
                    mean_loss = mean_loss / 10.0
                print('Average train loss at epoch %d, step %d: %f' % (epoch, i, mean_loss))
                mean_loss = 0
        train_loss /= (i + 1)
        print("[Epoch {}] Average train loss: {}".format(epoch, train_loss))

        # 每训一个epoch，测试一次
        pred_slots = []
        slot_accs = []
        intent_accs = []
        for j, batch in enumerate(getBatch(batch_size, index_test)):
            decoder_prediction, intent, intent_accuracy = model.step(sess, "test", batch)
            decoder_prediction = np.transpose(decoder_prediction, [1, 0])
            if j == 0:
                index = random.choice(range(len(batch)))
                # index = 0
                sen_len = batch[index][1]
                input_sentence = index_seq2word(batch[index][0], index2word)[:sen_len]
                print ("Input Sentence        : ") + str(index_seq2word(batch[index][0], index2word)[:sen_len]).decode('string_escape')
                print ("Slot Truth            : ") + str(index_seq2slot(batch[index][2], index2slot)[:sen_len]).decode('string_escape')
                print ("Slot Prediction       : ") + str(index_seq2slot(decoder_prediction[index], index2slot)[:sen_len]).decode('string_escape')
                print ("Intent Truth          : ") + str(index2intent[batch[index][3]]).decode('string_escape')
                print ("Intent Prediction     : ") + str(index2intent[intent[index]]).decode('string_escape')
                #print("Input Sentence        : ", index_seq2word(batch[index][0], index2word)[:sen_len])
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
            # print("slot accuracy: {}, intent accuracy: {}".format(slot_acc, intent_acc))
            slot_accs.append(slot_acc)
            intent_accs.append(intent_acc)
        pred_slots_a = np.vstack(pred_slots)
        # print("pred_slots_a: ", pred_slots_a.shape)
        true_slots_a = np.array(list(zip(*index_test))[2])[:pred_slots_a.shape[0]]
        # print("true_slots_a: ", true_slots_a.shape)
        print("Intent accuracy for epoch {}: {}".format(epoch, np.average(intent_accs)))
        print("Slot accuracy for epoch {}: {}".format(epoch, np.average(slot_accs)))
        print("Slot F1 score for epoch {}: {}".format(epoch, f1_for_sequence_batch(true_slots_a, pred_slots_a)))

        #JerryB +++
        if epoch % 10 == 0:
            print('[save checkpoint %d]' % (epoch))
            saver.save(sess, "model/ijia/model.ckpt", global_step = epoch)
        #JerryB ---


def test_data():
    train_data = open("dataset/atis-2.train.w-intent.iob", "r").readlines()
    test_data = open("dataset/atis-2.dev.w-intent.iob", "r").readlines()
    train_data_ed = data_pipeline(train_data)
    test_data_ed = data_pipeline(test_data)
    word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
        get_info_from_training_data(train_data_ed)
    # print("slot2index: ", slot2index)
    # print("index2slot: ", index2slot)
    index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
    index_test = to_index(test_data_ed, word2index, slot2index, intent2index)
    batch = next(getBatch(batch_size, index_test))
    unziped = list(zip(*batch))
    print("word num: ", len(word2index.keys()), "slot num: ", len(slot2index.keys()), "intent num: ",
          len(intent2index.keys()))
    print(np.shape(unziped[0]), np.shape(unziped[1]), np.shape(unziped[2]), np.shape(unziped[3]))
    print(np.transpose(unziped[0], [1, 0]))
    print(unziped[1])
    print(np.shape(list(zip(*index_test))[2]))


if __name__ == '__main__':
    #train(is_debug=True)
    #test_data()
    train()
