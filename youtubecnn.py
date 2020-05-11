#!/usr/bin/env python
# coding: utf-8

# In[4]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Activation, Dropout
from tensorflow.keras.layers import Conv1d, MaxPooling1D, GlobalMaxPooling1D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[5]:


df = pd.read_csv('tweetgalau')
df.head()
#nanti hasilnya bentuk tabel indexs, label sentimen, tweet


# In[ ]:


text = df["Tweet"].tolist()
print(len(text))
#hitung jumlah data semuanya


# In[ ]:


text = df["Tweet"].tolist()
print(text)
#tampil data tweetnya saja tanpa label 0,1,2


# In[ ]:


y = df["sentimen"]
y = to_categorical(y)
print(y)
#nanti hasilnya matriks 3


# In[ ]:


df["sentimen"].value_counts() 
#0 netral 1positif 2negatif keluar jumlah masing masing berdasarkan kategori


# In[ ]:


#lakukan tokenization
token = Tokenizer()
token.fit_on_texts(text)


# In[ ]:


token.index_word
#keluar indeks dari kata kata


# In[ ]:


vocab = len(token.index_word)+1
vocab
#nanti keluar jumlah indeks atau berapa kata yang dibuat dalam indeks 17715


# In[ ]:


x =['aku lagi galau nih']
token.texts_to_sequences(x)
# keluar nanti aku di indeks 1, lagi di 26, galau di 1994, nih di 120


# In[ ]:


encode_text = token.texts_to_sequences(text)
#nanti muncul tweetnya tapi berubah jadi angka indeks tempat kata disimpan


# In[ ]:


# agar semua kalimat sama panjang mempermudah
max_kata = 100
X = pad_sequences(encode_text, maxlen = max_kata, padding="post") #encode_text itu variabel yg mau di ubah, lalu maxlen di samakan dengan jumlah kata maksimal. kalau paddingnya "post" maka diisi ke belakang jadi nanti isinya 0000. kalau mau di depan pakainya "pre"
X


# In[ ]:


#memotong data dengan sklearn
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 40, test_size = 0.3, stratify=y)
# data latih x, data tes x


# In[ ]:


#diganti ke numpy
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
#data preprocess selesai (preparasi data)


# In[ ]:


#tentukan vector size untuk embedding. ubah menjadi wordembedding
vec_size = 300
model = Sequential()
model.add(Embedding(vocab, vec_size, input_length= max_kata))
model.add(Conv1d(64,8, activation="relu"))
model.add(MaxPooling1D(2))
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model,add(GlobalMaxPooling1D())
model.add(Dense(3, activation='softmax'))
model.summary()
#keluar sekian layers


# In[ ]:


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])


# In[ ]:


#mari kita training
model.fit(X_train, y_train, epochs=100, validation_data = (X_text, y_test)) #epochnya 2 biar cepat bisa pakai 10
#train on 5154 sample, validation on 2210 sample epoch 1/2 hasil running
#tunggu running trainingnya. semakin banyak epoch akurasi nya lebih akurat
#cek accuracy terakhir bagusnya 0.7 keatas. main di modelnya untuk dapat accuracy yang bagus


# In[ ]:


#klasifikasi kalimat
def get_encode(x) : #kalimat input di encode
    x = token.texts_to_sequences(x)
    x = pad_sequences(x, maxlen = max_kata, padding="post")
    return x


# In[ ]:


#coba tes kalimat / testing
x = ['saya sedih dan galau juga bingung']
x = get_encode(x)


# In[ ]:


model.predict_classes (x)
#keluar 2 negatif


# In[ ]:


#coba tes kalimat /testing
x = ['shari ini ibu memberi hadiah jadi aku senang']
x = get_encode(x)


# In[ ]:


model.predict_classes (x)
#keluarnya 1 negatif


# In[ ]:





# In[ ]:




