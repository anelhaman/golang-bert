#!/bin/bash -x

docker build -t bert-fine-tune .
docker run --rm -v $(pwd):/app bert-fine-tune
