import os

import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import ReLU, PReLU, Activation, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

from model import Model
from util import Util

# tensorflowの警告抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def set_tf_random_seed(seed=0):
    """
    tensorflow v2.0の乱数固定
    https://qiita.com/Rin-P/items/acacbb6bd93d88d1ca1b
    ※tensorflow-determinism が無いとgpuについては固定できないみたい
    　tensorflow-determinism はpipでしか取れない($ pip install tensorflow-determinism)ので未確認
    """
    import random
    import numpy as np
    import tensorflow as tf
    ## ソースコード上でGPUの計算順序の固定を記述
    #from tfdeterminism import patch
    #patch()
    # 乱数のseed値の固定
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)  # v1.0系だとtf.set_random_seed(seed)


class ModelNN(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        set_tf_random_seed()

        # データのセット・スケーリング
        validation = va_x is not None
        scaler = StandardScaler()
        scaler.fit(tr_x)
        tr_x = scaler.transform(tr_x)
        tr_y = to_categorical(tr_y, num_classes=9)

        if validation:
            va_x = scaler.transform(va_x)
            va_y = to_categorical(va_y, num_classes=9)

        # パラメータ
        nb_classes = 9
        layers = self.params['layers']
        dropout = self.params['dropout']
        units = self.params['units']
        nb_epoch = self.params['nb_epoch']
        patience = self.params['patience']

        # モデルの構築
        model = Sequential()
        model.add(Dense(units, input_shape=(tr_x.shape[1],)))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        for l in range(layers - 1):
            model.add(Dense(units))
            model.add(PReLU())
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        if validation:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                           verbose=1, restore_best_weights=True)
            model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=128, verbose=2,
                      validation_data=(va_x, va_y), callbacks=[early_stopping])
        else:
            model.fit(tr_x, tr_y, nb_epoch=nb_epoch, batch_size=128, verbose=2)

        # モデル・スケーラーの保持
        self.model = model
        self.scaler = scaler

    def predict(self, te_x):
        te_x = self.scaler.transform(te_x)
        pred = self.model.predict_proba(te_x)
        return pred

    def save_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join('../model/model', f'{self.run_fold_name}-scaler.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        Util.dump(self.scaler, scaler_path)

    def load_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join('../model/model', f'{self.run_fold_name}-scaler.pkl')
        self.model = load_model(model_path)
        self.scaler = Util.load(scaler_path)
