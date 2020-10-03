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

class emoMusic():

    def __init__(self, model = 'emo_model_v1.h5'):
        self.model = load_model(model)

    def extract_face(self,filename, checkbox_face_detected = False, checkbox_org_img = True, required_size=(48, 48)): 
        # load image from file
        
        faces = []
        faces_org = []
        image = Image.open(filename)
        image_main = Image.open(filename)
            
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = np.asarray(image)
        st.text("Image size- "+str(pixels.shape[1]) + " x " + str(pixels.shape[0]))
        if checkbox_org_img == True:
            st.image(pixels, caption='original picture', use_column_width=True)
        # create the detector, using default weights
        detector = MTCNN()

        # detect faces in the image
        results = detector.detect_faces(pixels)
        if len(results)<1:
            return [],[]
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
            faces_org.append(np.array(image))
            image = image.resize(required_size)
            faces.append(np.array(image))
        if checkbox_face_detected:
            st.image(image_main, use_column_width=True)
        return np.array(faces), np.array(faces_org)

    def pre_process(self,file, checkbox_face_detected, checkbox_org_img):
        img_, img_org = self.extract_face(file, checkbox_face_detected, checkbox_org_img)
        if len(img_) == 0:
            return [],[]
        images=[]
        i=0
        for img in img_:
            img = Image.fromarray(img)
            img = img.convert('L')
            img = np.array(img, np.float32)
            img = img/255
            img = img.reshape(1,48,48,1)
            images.append(img)
            i+=1
            #print(img.shape)
        pyplot.show()
        return np.array(images), img_org

    def helper(self,gener):
        st.button("Re-recommend")

        rand_genere = random.choice(gener)[0]

        st.text("Lets play "+rand_genere[:-4]+" songs")
        
        choice = st.slider("No. of songs",1,100,5)

        final_gener = pd.read_csv(rand_genere)
        li = final_gener.sample(choice)
        li = li.values.tolist()

        sample = 'Music-Genere/Data/genres_original/' + rand_genere[:-4] + '/'

        audio_list=[]

        for i in range(choice):
            #y, sr = librosa.load(sample+li[i][0],sr=5000, offset=0.0, duration=30)
            y = open(sample+li[i][0],'rb')
            audio_list.append(y)

        for i in audio_list:
            #ipd.display(ipd.Audio(i,rate=7000))
            aud_fil = i.read()
            st.audio(aud_fil)

    def emo_to_music(self,checkbox_recommeded_music_genere, emotion = "Happy"):
        
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
            gener_string = "country, disco, pop, hiphop"
        elif emotion == "Sad":
            gener = [blues,jazz,classical]
            gener_string = "blues, jazz, classical"
        elif emotion == "Angry":
            gener = [metal,rock]
            gener_string = "metal, rock"
        elif emotion == "Disgust":
            gener = [raggae,metal,rock]
            gener_string = "raggae, metal, rock"
        elif emotion == "Fear":
            gener = [blues,jazz,classical,metal,rock]
            gener_string = "blues, jazz, classical, metal, rock"
        elif emotion == "Surprise":
            gener = [disco,hiphop]
            gener_string = "disco, hiphop"
        elif emotion == "Neutral":
            gener = [metal,rock]
            gener_string = "metal, rock"
        


        if checkbox_recommeded_music_genere:
            st.text("Recommeded Generes are "+gener_string)

        self.helper(gener)



    def print_individual_faces(self,faceimages_org):
        for faces in faceimages_org:
            st.image(faces, wide = faces.shape[0])
        return

def main():
    emoMusic_v1 = emoMusic()
    pred=[]

    st.title('Moodi Music - A system to suggest music according to your mood')
    st.text("**********  Please do not upload images with face mask, else you will be sad *************")
    file = st.file_uploader("Upload a picture", type=["jpg"])

    st.sidebar.text("Filters")

    checkbox_org_img = st.sidebar.checkbox("Original uploaded image",True)
    checkbox_face_detected = st.sidebar.checkbox("Face detected in the image")
    checkbox_face_individual = st.sidebar.checkbox("Individual faces")
    checkbox_recommeded_music_genere = st.sidebar.checkbox("Suggested Music Generes")
    
    print(checkbox_org_img,checkbox_face_detected,checkbox_face_individual,checkbox_recommeded_music_genere)


    if file is not None:
        st.spinner(text='In progress...')
        faceimages, faceimages_org = emoMusic_v1.pre_process(file, checkbox_face_detected, checkbox_org_img)
        if faceimages == []:
            st.text("Sorry no faces found")
            return
        if checkbox_face_individual:
            emoMusic_v1.print_individual_faces(faceimages_org)

        for faces in faceimages:
            cur_pred = (emoMusic_v1.model.predict(
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
 
        emoMusic_v1.emo_to_music(checkbox_recommeded_music_genere, detected_emo)

main()

