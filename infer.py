# coding=utf-8
import tensorflow as tf
from data import *
# from model import Model
from model import Model
from my_metrics import *
from tensorflow.python import debug as tf_debug
import numpy as np

input_steps = 50
embedding_size = 64
hidden_size = 100
n_layers = 2
batch_size = 1
vocab_size = 871
slot_size = 122
intent_size = 22
#vocab_size = 126
#slot_size = 38
#intent_size = 6
epoch_num = 1

model_path = './model/'

def get_model():
    model = Model(input_steps, embedding_size, hidden_size, vocab_size, slot_size,
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
    # print(tf.trainable_variables())
    train_data = open("dataset/atis-2.train.w-intent.iob", "r").readlines()
    #train_data = open("dataset/jerryb_test.train.w-intent.iob", "r").readlines()
    #test_data = open("dataset/atis-2.dev.w-intent.iob", "r").readlines()
    #test_data = open("dataset/jerryb_test.dev.w-intent.iob", "r").readlines()
    test_data = open("dataset/jerryb_test.iob", "r").readlines()
    #print("JerryB train_data[0]: ", train_data[0])
    #print("JerryB train_data[1]: ", train_data[1])
    #print("JerryB train_data[2]: ", train_data[2])
    train_data_ed = data_pipeline(train_data)
    test_data_ed = data_pipeline(test_data)
    word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
        get_info_from_training_data(train_data_ed)
    #print("JerryB word2index: ", word2index)
    #print("JerryB index2word: ", index2word)
    #print("JerryB slot2index: ", slot2index)
    #print("JerryB index2slot: ", index2slot)
    #print("JerryB intent2index: ", intent2index)
    #print("JerryB index2intent: ", index2intent)
    index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
    index_test = to_index(test_data_ed, word2index, slot2index, intent2index)
    #print("JerryB index_train[0]: ", index_train[0])
    #print("JerryB index_train[1]: ", index_train[1])
    #print("JerryB index_train[2]: ", index_train[2])
    
    #for epoch in range(epoch_num):
    #mean_loss = 0.0
    #train_loss = 0.0
    #for i, batch in enumerate(getBatch(batch_size, index_train)):
    #    # 执行一个batch的训练
    #    _, loss, decoder_prediction, intent, mask, slot_W = model.step(sess, "train", batch)

    #    mean_loss += loss
    #    train_loss += loss
    #    if i % 10 == 0:
    #        if i > 0:
    #            mean_loss = mean_loss / 10.0
    #        print('Average train loss at epoch 0, step %d: %f' % (i, mean_loss))
    #        mean_loss = 0
    #train_loss /= (i + 1)
    #print("[Epoch 0] Average train loss: {}".format(train_loss))

    # 每训一个epoch，测试一次
    pred_slots = []
    slot_accs = []
    intent_accs = []
    for j, batch in enumerate(getBatch(batch_size, index_test)):
        decoder_prediction, intent = model.step(sess, "test", batch)
        decoder_prediction = np.transpose(decoder_prediction, [1, 0])
        if j == 0:
            index = random.choice(range(len(batch)))
            #index = 0
            sen_len = batch[index][1]
            print("Input Sentence        : ", index_seq2word(batch[index][0], index2word)[:sen_len])
            print("Slot Truth            : ", index_seq2slot(batch[index][2], index2slot)[:sen_len])
            print("Slot Prediction       : ", index_seq2slot(decoder_prediction[index], index2slot)[:sen_len])
            print("Intent Truth          : ", index2intent[batch[index][3]])
            print("Intent Prediction     : ", index2intent[intent[index]])
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
    print("Intent accuracy for epoch 0: {}".format(np.average(intent_accs)))
    print("Slot accuracy for epoch 0: {}".format(np.average(slot_accs)))
    print("Slot F1 score for epoch 0: {}".format(f1_for_sequence_batch(true_slots_a, pred_slots_a)))

if __name__ == '__main__':
    infer()
