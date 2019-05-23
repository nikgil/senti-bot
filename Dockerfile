# base image
FROM python:3.6.8-stretch

# gotta rep myself
LABEL mainainer="ngilevskiy@gmail.com"

# copy everything to work directory
COPY . /senti-bot
WORKDIR /senti-bot

COPY .env .

# install all reqs for script
RUN pip install -r requirements.txt
RUN python pickle_files.py

ARG ENV_KEY

# Finally run the actual target command
CMD ["python", "main.py", "$ENV_KEY"]