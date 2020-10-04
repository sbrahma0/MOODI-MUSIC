# MOODI-MUSIC
## A system which identifies huuman emotion from an image and suggests music

On a highlevel this project uses human emotion and music genere dataset to develop a web app that identifies human emotion and recommend music so that users dont have to manually create music playlist.

### Workflow Diagram
![image](results/work_flow.png)

### Setup Instructions
Clone this repository and get in it
```
git clone https://github.com/sbrahma0/MOODI-MUSIC.git
cd ./MOODI-MUSIC
```
Install all the dependencies
```
pip install -r requirements.txt 
```
Download the Emotion recognition model from the following [link](https://drive.google.com/file/d/19su4fmTbqQkLQxQiV1vhOO1zTMQgClc1/view?usp=sharing) and save it in the repository.

Download the music dataset from the following [link](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) and place the "Music-Genere" folder inside the repo folder.

Run this command
```
streamlit run demo_multiple.py
```
If the browser doesnt not open automatically, copy and paste the web address shown in the terminal.

Once you are in the browser, upload an image and have fun 

## About Me
#### ACADEMIC JOURNEY
Bachelors in Mechatronics Engineering  >  Masters in Robotics  >  Perception and AI enthusiast
