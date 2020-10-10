# Dockerfile for building streamline app

# pull miniconda image
FROM python:3.7.6

RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

# copy local files into container
COPY demo_multiple.py /tmp/
COPY requirements.txt /tmp/
COPY emo_model_v1.h5 /tmp/

COPY Music-Genere/Data/genres_original /tmp/Music-Genere/Data/genres_original

COPY blues.csv /tmp/blues.csv
COPY classical.csv /tmp/classical.csv
COPY country.csv /tmp/country.csv
COPY disco.csv /tmp/disco.csv
COPY hiphop.csv /tmp/hiphop.csv
COPY jazz.csv /tmp/jazz.csv
COPY metal.csv /tmp/metal.csv
COPY pop.csv /tmp/pop.csv
COPY reggae.csv /tmp/reggae.csv
COPY rock.csv /tmp/rock.csv

# change directory
WORKDIR /tmp

# install dependencies
RUN apt-get update
RUN pip install -r requirements.txt

EXPOSE 8501

# run commands
CMD ["streamlit", "run", "demo_multiple.py"]
