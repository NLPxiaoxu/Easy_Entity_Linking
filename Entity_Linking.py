#! -*- coding:utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import json
from transformers import BertTokenizer, TFBertModel
from tqdm import tqdm
import random
from random import choice

import logging
logging.disable(30)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

_train_data = []
with open('./data/train.json', encoding='utf-8') as f:
    for l in tqdm(f):
        _ = json.loads(l)
        _train_data.append({
            'text': _['text'].lower(),
            'mention_data': [(x['mention'].lower(), int(x['offset']), x['kb_id'])
                for x in _['mention_data'] if x['kb_id'] != 'NIL'
            ]
        })

random.shuffle(_train_data)
train_data = _train_data[:85000]
dev_data = _train_data[85000:]


id2kb = {}
with open('./data/kb_data', encoding='utf-8') as f:
    for l in tqdm(f):
        _ = json.loads(l)
        subject_id = _['subject_id']
        subject_alias = list(set([_['subject']] + _.get('alias', [])))
        subject_alias = [alias.lower() for alias in subject_alias]
        subject_desc = '\n'.join(u'%s：%s' % (i['predicate'], i['object']) for i in _['data'])
        subject_desc = subject_desc.lower()
        subject_desc = subject_desc.replace('\n', '，')
        if subject_desc:
            id2kb[subject_id] = {'subject_alias': subject_alias, 'subject_desc': subject_desc}


kb2id = {}
for i, j in id2kb.items():
    for k in j['subject_alias']:
        if k not in kb2id:
            kb2id[k] = []
        kb2id[k].append(i)

# bert配置
checkpoint_path = "/home/xxj/NLP_Project/bert_model/chinese_L-12_H-768_A-12/"
tokenizer = BertTokenizer.from_pretrained(checkpoint_path, lowercase=True, add_special_tokens=True)
bert_model = TFBertModel.from_pretrained(checkpoint_path)


label_class = 3
lr = 1e-5
epsilon = 1e-06
num_epochs = 20
batch_size = 8
keep_prob = 0.5
max_len = 64
max_len_cat = 280

def get_input_bert(data):
    input_x, input_segment, input_mask, input_ner, position_s, position_e, S, \
    input_y, input_y_segment, input_y_mask = [], [], [], [], [], [], [], [], [], []
    for l in tqdm(range(85000)):
        line = data[l]
        text = line['text'][0:max_len-2]
        word_list = [key for key in text]
        word_list.insert(0, "[CLS]")
        word_list.append("[SEP]")
        token_ids = tokenizer.convert_tokens_to_ids(word_list)
        segment_ids = np.zeros(len(token_ids))
        mask = np.ones(len(token_ids))


        ner_s = np.zeros(len(token_ids), dtype=np.int32)
        mds = {}
        for j in line['mention_data']:
            if j[0] in kb2id:
                s = j[1] + 1
                e = s + len(j[0])
                ner_s[s] = 1
                ner_s[s+1:e] = 2
                mds[(s, e)] = (j[0], j[2])


        if mds:
            input_x.append(token_ids)
            input_segment.append(segment_ids)
            input_mask.append(mask)
            input_ner.append(ner_s)
            s, e = choice(list(mds))
            id = choice(kb2id[mds[(s, e)][0]])
            if id == mds[(s, e)][1]:
                S.append([1])
            else:
                S.append([0])
            position_s.append(s)
            position_e.append(e)

            text_y = id2kb[id]['subject_desc'][0:max_len_cat-max_len-1]
            word_list_y = [key for key in text_y]
            word_list_y.insert(0, "[SEP]")
            y_word_list = word_list + word_list_y
            y_word_list.insert(0, "[CLS]")
            y_word_list.append("[SEP]")
            token_ids_y = tokenizer.convert_tokens_to_ids(y_word_list)
            segment_ids_y = np.zeros(len(token_ids_y))
            mask_y = np.ones(len(token_ids_y))
            input_y.append(token_ids_y)
            input_y_segment.append(segment_ids_y)
            input_y_mask.append(mask_y)

    position_s = np.array(position_s, dtype=np.int32)
    position_e = np.array(position_e, dtype=np.int32)
    S = np.array(S, dtype=np.float32)

    input_x = tf.keras.preprocessing.sequence.pad_sequences(input_x, max_len, padding='post', truncating='post')
    input_segment = tf.keras.preprocessing.sequence.pad_sequences(input_segment, max_len, padding='post', truncating='post')
    input_mask = tf.keras.preprocessing.sequence.pad_sequences(input_mask, max_len, padding='post', truncating='post')
    input_ner = tf.keras.preprocessing.sequence.pad_sequences(input_ner, max_len, padding='post', truncating='post')

    input_y = tf.keras.preprocessing.sequence.pad_sequences(input_y, max_len_cat, padding='post', truncating='post')
    input_y_segment = tf.keras.preprocessing.sequence.pad_sequences(input_y_segment, max_len_cat, padding='post', truncating='post')
    input_y_mask = tf.keras.preprocessing.sequence.pad_sequences(input_y_mask, max_len_cat, padding='post', truncating='post')

    return input_x, input_segment, input_mask, input_ner, position_s, position_e, S, input_y, input_y_segment, input_y_mask


class data_loader():
    def __init__(self):
        self.input_x, self.input_segment, self.input_mask, self.input_ner, self.position_s, self.position_e, self.S, self.input_y, self.input_y_segment, self.input_y_mask = get_input_bert(train_data)

        self.input_x = self.input_x.astype(np.int32)
        self.input_segment = self.input_segment.astype(np.int32)
        self.input_mask = self.input_mask.astype(np.int32)
        self.input_ner = self.input_ner.astype(np.int32)
        self.position_s = self.position_s.astype(np.int32)
        self.position_e = self.position_e.astype(np.int32)
        self.S = self.S.astype(np.float32)
        self.input_y = self.input_y.astype(np.int32)
        self.input_y_segment = self.input_y_segment.astype(np.int32)
        self.input_y_mask = self.input_y_mask.astype(np.int32)


        self.num_train = self.input_x.shape[0]
        self.db_train = tf.data.Dataset.from_tensor_slices((self.input_x, self.input_segment, self.input_mask, self.input_ner, self.position_s,self.position_e, self.S, self.input_y, self.input_y_segment, self.input_y_mask))
        self.db_train = self.db_train.shuffle(self.num_train).batch(batch_size, drop_remainder=True)

    def get_batch(self, batch_s):
        indics = np.random.randint(0, self.num_train, batch_s)
        return self.input_x[indics], self.input_segment[indics], self.input_mask[indics], self.input_ner[indics], self.position_s[indics], self.position_e[indics], self.S[indics], self.input_y[indics], self.input_y_segment[indics], self.input_y_mask[indics]


class Ner_model(tf.keras.Model):
    def __init__(self, bert_model):
        super(Ner_model, self).__init__()
        self.bert = bert_model
        #self.dense_fuc = tf.keras.layers.Dense(100, use_bias=False) #全连接层
        self.dense = tf.keras.layers.Dense(label_class)


    def call(self, inputs, mask, segment):
        output_encode, _ = self.bert([inputs, mask, segment])
        logits = self.dense(output_encode)
        output = tf.nn.softmax(logits)
        return output


class Similarity_model(tf.keras.Model):
    def __init__(self, bert_model):
        super(Similarity_model, self).__init__()
        self.bert = bert_model
        self.drop_out = tf.keras.layers.Dropout(keep_prob)
        self.dense = tf.keras.layers.Dense(1)
        self.average = tf.keras.layers.Average()


    def call(self, inputs, mask, segment, position_s, position_e):
        output_encode, _ = self.bert([inputs, mask, segment])
        cls_encode = output_encode[:, 0, :]

        subject_encode = []
        for i, k in enumerate(position_s):
            average_encode = []
            v_encode = output_encode[i, :, :]
            for n in range(k, position_e[i]):
                v_subject = v_encode[k, :]
                average_encode.append(v_subject)
            subject_encode.append(self.average(average_encode))

        subject_encode = tf.reshape(subject_encode, [-1, cls_encode.shape[1]])

        final_encode = tf.concat([cls_encode, subject_encode], 1)
        # print(final_encode.shape)
        #final_encode = self.drop_out(final_encode)
        final_encode = self.dense(final_encode)
        output = tf.nn.sigmoid(final_encode)
        return output


def loss_function(ner_pre, input_ner, y_pre, input_s):
    input_ner_onehot = tf.one_hot(input_ner, depth=3, dtype=tf.float32)
    loss_ner = tf.keras.losses.categorical_crossentropy(y_true=input_ner_onehot, y_pred=ner_pre)
    loss_1 = tf.reduce_sum(loss_ner)

    # print(input_s.shape, y_pre.shape)
    loss_sm = tf.keras.losses.binary_crossentropy(y_true=input_s, y_pred=y_pre)
    loss_sm = tf.reduce_sum(loss_sm)

    return (loss_1+loss_sm), loss_1, loss_sm


class Extra_result(object):
    def __init__(self, text):
        self.text = text
    def call(self):
        word_list = [key for key in self.text]
        word_list.insert(0, "[CLS]")
        word_list.append("[SEP]")
        segment_ids = np.zeros(len(word_list))
        mask = np.ones(len(word_list))
        token = tf.constant(tokenizer.convert_tokens_to_ids(word_list), dtype=tf.int32)[None, :]
        segment_ids = tf.constant(segment_ids, dtype=tf.int32)[None, :]
        mask = tf.constant(mask, dtype=tf.int32)[None, :]
        ner_label = model_Ner(token, mask, segment_ids)
        subjects = self.extra_sujects(ner_label)
        result = []


        for _s in subjects:
            p_s, p_e, y, y_mask, y_segment = [], [], [], [], []
            ids = {}
            ids[_s] = kb2id.get(_s[0], [])
            for i in ids[_s]:
                sentence = id2kb[i]['subject_desc']
                p_s.append(_s[1] + 1)
                p_e.append(_s[2] + 1)
                word_sen = [key for key in sentence]
                word_sen.insert(0, "[SEP]")
                cat_sen = word_list + word_sen
                cat_sen.insert(0, "[CLS]")
                cat_sen.append("[SEP]")
                token_y = tokenizer.convert_tokens_to_ids(cat_sen)
                y.append(token_y)
                y_mask.append(np.ones(len(token_y)))
                y_segment.append(np.ones(len(token_y)))

            if len(y) >= 1 and len(ids[_s]) >= 1:
                p_s = np.array(p_s, dtype=np.int32)
                p_e = np.array(p_e, dtype=np.int32)
                y = tf.keras.preprocessing.sequence.pad_sequences(y, max_len_cat, padding='post', truncating='post')
                y_segment = tf.keras.preprocessing.sequence.pad_sequences(y_segment, max_len_cat, padding='post',
                                                                          truncating='post')
                y_mask = tf.keras.preprocessing.sequence.pad_sequences(y_mask, max_len_cat, padding='post',
                                                                       truncating='post')

                # print(y.shape)
                # print(len(ids[_s]))
                scores = model_Similarity(y, y_mask, y_segment, p_s, p_e)
                # print(scores.shape)
                score = [k[0] for k in scores]
                kbid = ids[_s][np.argmax(score)]
                result.append((_s[0], _s[1], kbid))
        print(self.text)
        print(subjects)
        print(result)
        print('\n')
        return result

    def extra_sujects(self, ner_label):
        ner = ner_label[0]
        #ner = tf.round(ner)
        ner = [tf.argmax(ner[k]) for k in range(ner.shape[0])]
        ner = list(np.array(ner))[1:-1]
        ner.append(0)#防止最后一位不为0
        text_list = [key for key in self.text]
        subject = []
        for i, k in enumerate(text_list):
            if int(ner[i]) == 0 or int(ner[i]) == 2:
                continue
            elif int(ner[i]) == 1:
                ner_back = [int(j) for j in ner[i + 1:]]
                if 1 in ner_back and 0 in ner_back:
                    indics_1 = ner_back.index(1) + i
                    indics_0 = ner_back.index(0) + i
                    subject.append((''.join(text_list[i: min(indics_0, indics_1) + 1]), i, min(indics_0, indics_1)+1))
                elif 1 not in ner_back:
                    indics = ner_back.index(0) + i
                    subject.append((''.join(text_list[i:indics + 1]), i, indics+1))
        return subject

class Evaluate(object):
    def __init__(self):
        pass
    def evaluate(self, data):
        A, B, C = 1e-10, 1e-10, 1e-10
        for d in data[0:10]:
            extra_items = Extra_result(d['text'])
            R = set(extra_items.call())
            T = set([(key[0], key[1], key[2]) for key in d['mention_data']])
            A += len(R & T)#抽取正确数量
            B += len(R) #抽取数量
            C += len(T)#原正确数量
        return (2 * A / (B + C)), (A / B), (A / C)


#建立模型
model_Ner = Ner_model(bert_model)
model_Similarity = Similarity_model(bert_model)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon)

#保存模型
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model_Ner=model_Ner, model_Similarity=model_Similarity)

evaluate = Evaluate()
data_loader = data_loader()

best = 0.0

for epoch in range(num_epochs):
    print('Epoch:', epoch + 1)

    num_batchs = int(data_loader.num_train / batch_size) + 1
    for batch_index in range(num_batchs):
        input_x, input_segment, input_mask, input_ner, position_s, position_e, S, input_y, input_y_segment, input_y_mask = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            logits = model_Ner(input_x, input_mask, input_segment) #预测ner
            sm_output = model_Similarity(input_y, input_y_mask, input_y_segment, position_s, position_e)
            loss, loss1, loss2 = loss_function(logits, input_ner, S, sm_output)
            if (batch_index+1) % 500 == 0:
                print("batch %d: loss %f: loss1 %f: loss2 %f" % (batch_index+1, loss.numpy(), loss1.numpy(), loss2.numpy()))

        variables = (model_Ner.variables + model_Similarity.variables)
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

    #f, p, r = evaluate.evaluate(train_data)
    F, P, R = evaluate.evaluate(dev_data)
    #print('训练集:', "f %f: p %f: r %f: " % (f, p, r))
    print('测试集:', "F %f: P %f: R %f: " % (F, P, F))
    if round(F, 2) > best and round(F, 2) > 0.50:
        best = F
        print('saving_model')
        #model.save('./save/Entity_Relationshaip_version2.h5')
        checkpoint.save('./save/Entity_Linking/EL_checkpoints.ckpt')
