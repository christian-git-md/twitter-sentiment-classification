# Twitter sentiment classification
This code is my solution proposal for the home assignment. This readme contains a brief rundown on how to serve the model, a description of what was implemented here can be found in the notebook [twitter_sentiment_presentation.ipynb](twitter_sentiment_presentation.ipynb).
## Prerequisites
- docker installed
- ports 9090 & 9091 available
- only tested on Ubuntu 20.04
## Setup
Clone this repo, then run:
```bash
cd docker
docker build . -t twitter-sentiment
```
## Run
To serve the model, run (from the repo root directory):

`bash docker/run_docker.sh` if nvidia-docker and a gpu are available or `docker/run_docker_cpu.sh` to serve with cpu.
## Make a request
Send any text to the served model to evaluate its sentiment:
```bash
curl -X POST http://0.0.0.0:9090/ -d "This game was great"
# [{"label":"Positive","score":0.9881765246391296}]
```