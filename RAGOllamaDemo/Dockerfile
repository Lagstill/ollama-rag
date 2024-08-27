# Use the official Python image as a base
FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt install -y libgl1-mesa-glx

COPY /app/index_documents.py ./

COPY /app/requirements.txt .

RUN pip install -r requirements.txt
COPY /app/data/docker.txt ./app/data/docker.txt
COPY /app/ .

RUN echo '#!/bin/sh\n\
echo "OLLAMA_HOST: $OLLAMA_HOST"\n\
python index_documents.py\n\
streamlit run main.py' > /start.sh && chmod +x /start.sh

EXPOSE 8501

CMD ["/start.sh"]
