#!/bin/bash
docker run \
 -it \
 --rm \
 -v $(pwd):/twitter_sentiment \
 -w /twitter_sentiment \
 -p 9090:9090 \
 -p 9091:9091 \
 twitter-sentiment:latest
