from PIL import Image, ImageDraw
import numpy as np
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import load_model
from matplotlib import pyplot
import pandas as pd
import random
import glob
import librosa
import IPython.display as ipd

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

import streamlit as st

st.set_option('deprecation.showfileUploaderEncoding', False)

model = load_model('emo_model_v1.h5')

def extract_face(filename, required_size=(48, 48)): 
    # load image from file
    
    faces = []
    image = Image.open(filename)
    image_main = Image.open(filename)
        
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    st.text(pixels.shape[1])
    st.text(pixels.shape[0])
    st.image(pixels, caption='original picture', use_column_width=True)
    # create the detector, using default weights
    detector = MTCNN()

    # detect faces in the image
    results = detector.detect_faces(pixels)
    #print(len(results))
    for face in results:
        x1, y1, width, height = face['box']
        rect_img = [(x1, y1), (x1 + width, y1 + height)]

        img1 = ImageDraw.Draw(image_main)   
        img1.rectangle(rect_img, outline ="red", width=5)

        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        #st.image(image, caption = 'Face',width=face.shape[1]//2)
        image = image.resize(required_size)
        faces.append(np.array(image))
    st.image(image_main, use_column_width=True)
    return np.array(faces)

def pre_process(file):
    img_ = extract_face(file)
    images=[]
    #pyplot.imshow(img)
    #print(type(img))
    i=0
    for img in img_:
        img = Image.fromarray(img)
        pyplot.subplot(1, len(img_), i+1)
        pyplot.axis('off')
        pyplot.imshow(img)
        #plt.imshow(img)
        img = img.convert('L')
        img = np.array(img, np.float32)
        img = img/255
        img = img.reshape(1,48,48,1)
        images.append(img)
        i+=1
        #print(img.shape)
    pyplot.show()
    return np.array(images)

def helper(gener):
    
    rand_genere = random.choice(gener)[0]

    st.text("Lets play "+rand_genere[:-4]+" songs")

    final_gener = pd.read_csv(rand_genere)
    li = final_gener.sample(5)
    li = li.values.tolist()

    sample = 'Music-Genere/Data/genres_original/' + rand_genere[:-4] + '/'

    audio_list=[]

    for i in range(5):
        #y, sr = librosa.load(sample+li[i][0],sr=5000, offset=0.0, duration=30)
        y = open(sample+li[i][0],'rb')
        audio_list.append(y)

    for i in audio_list:
        #ipd.display(ipd.Audio(i,rate=7000))
        aud_fil = i.read()
        st.audio(aud_fil)

def emo_to_music(emotion = "Happy"):
    
    pop = glob.glob("pop.csv")
    country = glob.glob("country.csv")
    disco = glob.glob("disco.csv")
    blues = glob.glob("blues.csv")
    reggae = glob.glob("reggae.csv")
    jazz = glob.glob("jazz.csv")
    metal = glob.glob("metal.csv")
    hiphop = glob.glob("hiphop.csv")
    classical = glob.glob("classical.csv")
    rock = glob.glob("rock.csv")
    
    if emotion == "Happy":
        gener = [country,disco,pop,hiphop]
    elif emotion == "Sad":
        gener = [blues,jazz,classical]
    elif emotion == "Angry":
        gener = [metal,rock]
    elif emotion == "Disgust":
        gener = [raggae,metal,rock]
    elif emotion == "Fear":
        gener = [blues,jazz,classical,metal,rock]
    elif emotion == "Surprise":
        gener = [disco,hiphop]
    elif emotion == "Neutral":
        gener = [metal,rock]
    
    helper(gener)

pred=[]
#options=["3"]
#choice = st.selectbox("Selct picture",options)

file = st.file_uploader("Upload a picture", type=["jpg"])
#file = '3.jpg'
#if choice == "3":
#    file = '3.jpg'
if file is not None:
    faceimages = pre_process(file)
    #print(faceimages[0].shape)
    for faces in faceimages:
        cur_pred = (model.predict(
            faces, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
            workers=1, use_multiprocessing=False))
        #print(np.around(cur_pred,3))
        pred.append(cur_pred)
        
    pred = np.array(pred)
    pred = np.sum(pred,0)
    pred = pred/len(faceimages)
    #print(np.around(pred,3))
    emos = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    
    
                       
    detected_emo = emos[np.argmax(pred, axis=1)[0]]
    if len(faceimages)>1:
    	st.text("The crowd seems to be- "+detected_emo)
    else:
        st.text("The person in the picture seems to be- "+detected_emo)

    emo_to_music(detected_emo)

