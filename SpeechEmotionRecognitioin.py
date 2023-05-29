# Import Library
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

#Audio
import librosa
import librosa.display

#Play Audio
from IPython.display import Audio

#Import warnings
import warnings
warnings.filterwarnings('ignore')
# Load Dataset
paths=[]
labels=[]

for dirname, _,filenames in os.walk(TESS Toronto emotional speech set data'):
    for filename in filenames:
        paths.append(os.path.join(dirname,filename))
        # print(filename)
        label=filename.split('_')[-1]   #In ABC_DEF_GHI split('_')[-1] will be GHI
        # print(label)
        label=label.split('.')[0]   # In ABC.wav split('.')[0] will be ABC.
        labels.append(label.lower())
        # print(label.lower())
        # break

print('Dataset is Loaded')
paths[:5]
labels[:5]
# Create a dataframe
df=pd.DataFrame()
df['speech']=paths
df['label']=labels
df.head()
df['label'].value_counts()
# Exploratory Data Analysis
# sns.countplot(df['label'])
sns.displot(df,x=df['label'])
# Display wave form for audio
def waveplot(data,sr,emotion):  #Display waveplot
    plt.figure(figsize=(10,4))
    plt.title(emotion,size=20)
    librosa.display.waveshow(data,sr=sr)
    # librosa.display.AdaptiveWaveplot(data,sr)
    plt.show()


def spectogram(data,sr,emotion):    #Display the spectogram

    x=librosa.stft(data)
    xdb=librosa.amplitude_to_db(abs(x))

    plt.figure(figsize=(10,4))
    plt.title(emotion,size=20)
    librosa.display.specshow(xdb,sr=sr,x_axis='time',y_axis='hz')
    plt.colorbar()
# Angry
emotion='angry'
# emotion='fear'
path=np.array(df['speech'][df['label']==emotion])[0]
data,sampling_rate=librosa.load(path)
waveplot(data,sampling_rate,emotion)
spectogram(data,sampling_rate,emotion)
Audio(path)
# Fear
emotion='fear'
path=np.array(df['speech'][df['label']==emotion])[0]
data,sampling_rate=librosa.load(path)
waveplot(data,sampling_rate,emotion)
spectogram(data,sampling_rate,emotion)
Audio(path)
# Disgust
emotion='disgust'
path=np.array(df['speech'][df['label']==emotion])[0]
data,sampling_rate=librosa.load(path)
waveplot(data,sampling_rate,emotion)
spectogram(data,sampling_rate,emotion)
Audio(path)
# Happy
emotion='happy'
path=np.array(df['speech'][df['label']==emotion])[0]
data,sampling_rate=librosa.load(path)
waveplot(data,sampling_rate,emotion)
spectogram(data,sampling_rate,emotion)
Audio(path)
# Neutral
emotion='neutral'
path=np.array(df['speech'][df['label']==emotion])[0]
data,sampling_rate=librosa.load(path)
waveplot(data,sampling_rate,emotion)
spectogram(data,sampling_rate,emotion)
Audio(path)
# Sad
emotion='sad'
path=np.array(df['speech'][df['label']==emotion])[0]
data,sampling_rate=librosa.load(path)
waveplot(data,sampling_rate,emotion)
spectogram(data,sampling_rate,emotion)
Audio(path)
# PS
emotion='ps'    #Pleasant Surprise
path=np.array(df['speech'][df['label']==emotion])[0]
data,sampling_rate=librosa.load(path)
waveplot(data,sampling_rate,emotion)
spectogram(data,sampling_rate,emotion)
Audio(path)
# Feature Extraction
def extract_mfcc(filename):
    y,sr=librosa.load(filename,duration=3,offset=0.5)
    mfcc=np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
    return mfcc
extract_mfcc(df['speech'][0])
X_mfcc=df['speech'].apply(lambda x: extract_mfcc(x))
X_mfcc
X=[x for x in X_mfcc]
X=np.array(X)
X.shape     # 1->No. of samples, 2->No. of features.
## Input Split
X=np.expand_dims(X,-1)
X.shape
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
y=enc.fit_transform(df[['label']])
y=y.toarray()
y.shape
# Create LSTM Model
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout

model=Sequential([
    LSTM(123,return_sequences=False,input_shape=(40,1)),
    Dense(64,activation='relu'),
    Dropout(0.2),
    Dense(32,activation='relu'),
    Dropout(0.2),
    Dense(7,activation='softmax')
])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
# Train the model
history=model.fit(X,y,validation_split=0.2,epochs=100,batch_size=512,shuffle=True)
# Plot Results
epochs=list(range(100))

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

plt.plot(epochs,acc,label='train_accuracy')
plt.plot(epochs,val_acc,label='val_accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.legend()
plt.show()
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.plot(epochs,loss,label='train_loss')
plt.plot(epochs,val_loss,label='val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.legend()
plt.show()
