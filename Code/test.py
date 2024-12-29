# loading in human data
# load in the data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ProjectDraftCode import x_mouse_positive

human_positive = pd.read_csv('Data/promoter.csv')
human_positive.columns = ['Sequence']
human_negative = pd.read_csv('Data/non_promoter.csv')
human_negative.columns = ['Sequence']
# clip all the sequences to the same length
y_human = np.ones(human_positive.shape[0]).tolist() + np.zeros(human_negative.shape[0]).tolist()
x_human = pd.concat([human_positive, human_negative], axis=0)
min_len = min(x_human['Sequence'].str.len())
# loading in the data
thaliana = pd.read_csv('Data/ara_tha_final.csv')
mouse = pd.read_csv('Data/mouse_mus_final.csv')
x_mouse = mouse[['Sequence']]
y_mouse = mouse['Promoter ID']
y_mouse = (y_mouse.str.contains('Non-promoter').astype(int) == 0).tolist()
x_thal = thaliana[['Sequence']]
y_thal = thaliana['Promoter ID']
y_thal = (y_thal.str.contains('Non-promoter').astype(int) == 0).tolist()
x_mouse_negative= x_mouse[~y_mouse]
x_mouse_positive = x_mouse[y_mouse].head(len(x_mouse_negative))
x_mouse = pd.concat([x_mouse_negative, x_mouse_positive], axis=0)
y_mouse = np.zeros(x_mouse_negative.shape[0]).tolist() + np.ones(x_mouse_positive.shape[0]).tolist()
print(x_mouse)
# cutting random 200bp segments
x_mouse= x_mouse['Sequence'].apply(lambda x: x[np.random.randint(0, len(x)-200):np.random.randint(0, len(x)-200)+200])
x_human= x_human['Sequence'].apply(lambda x: x[np.random.randint(0, len(x)-200):np.random.randint(0, len(x)-200)+200])
x_thal= x_thal['Sequence'].apply(lambda x: x[np.random.randint(0, len(x)-200):np.random.randint(0, len(x)-200)+200])

# split human and thal into train, test, and validation
human_x_train, human_x_test, human_y_train, human_y_test  = train_test_split(x_human, y_human, test_size=0.2,)
human_x_train, human_x_val, human_y_train, human_y_val = train_test_split(human_x_train, human_y_train, test_size=0.2)
# split mouse into train, test
mouse_x_train, mouse_x_test, mouse_y_train, mouse_y_test = train_test_split(X_mouse, y_mouse, test_size=0.2)
# split thal and thal into train, test, and validation
thal_x_train, thal_x_test, thal_y_train, thal_y_test = train_test_split(x_thal, y_thal, test_size=0.7)
thal_x_train, thal_x_val, thal_y_train, thal_y_val = train_test_split(thal_x_train, thal_y_train, test_size=0.4)
# combine human and thal train and validation
cross_x_train = pd.concat([human_x_train, thal_x_train], axis=0)
cross_x_val = pd.concat([human_x_val, thal_x_val], axis=0)
cross_y_train = human_y_train + thal_y_train
cross_y_val = human_y_val + thal_y_val


# load k_4 keras model
from keras.models import load_model
model = load_model('k_4.h5')
def create_embedding_model_with_transformer(
        max_length, vocab_size, embedding_dim=64,
):
    inputs = Input(shape=(max_length,), dtype=tf.int32)
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    positional_encoding = keras_hub.layers.SinePositionEncoding(max_wavelength=10000)(embedding)
    embedding = embedding + positional_encoding
    lstm = Bidirectional(LSTM(64, return_sequences=True))(embedding)
    # attention =MultiHeadAttention(num_heads=8, key_dim=embedding_dim)(lstm, lstm)
    # adding pass through connection
    for i in range(2):
        x = transformer_block(embedding, embedding_dim, 8, 128,  dropout_rate=0.5)
    # add passthrough
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
