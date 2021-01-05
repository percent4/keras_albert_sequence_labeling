# -*- coding: utf-8 -*-
# @Time : 2020/12/25 11:38
# @Author : Jclian91
# @File : model.py
# @Place : Yangpu, Shanghai
import json
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

from util import event_type
from albert import load_brightmart_albert_zh_checkpoint


# 创建BERT-BiLSTM-CRF模型
class BertBilstmCRF:
    def __init__(self, max_seq_length, lstm_dim):
        self.max_seq_length = max_seq_length
        self.lstmDim = lstm_dim
        self.label = self.load_label()

    # 抽取的标签
    def load_label(self):
        label_path = "./{}_label2id.json".format(event_type)
        with open(label_path, 'r', encoding='utf-8') as f_label:
            label = json.loads(f_label.read())

        return label

    # 模型
    def create_model(self):
        # load albert model
        model_path = "./albert_xlarge_zh_183k/"
        albert_model = load_brightmart_albert_zh_checkpoint(model_path, training=False)
        # make bert layer trainable
        for layer in albert_model.layers:
            layer.trainable = True
        x1 = Input(shape=(None,))
        x2 = Input(shape=(None,))
        bert_out = albert_model([x1, x2])
        lstm_out = Bidirectional(LSTM(self.lstmDim,
                                      return_sequences=True,
                                      dropout=0.2,
                                      recurrent_dropout=0.2))(bert_out)
        crf_out = CRF(len(self.label), sparse_target=True)(lstm_out)
        model = Model([x1, x2], crf_out)
        model.summary()
        model.compile(
            optimizer=Adam(1e-4),
            loss=crf_loss,
            metrics=[crf_accuracy]
        )
        return model
