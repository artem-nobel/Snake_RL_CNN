FROM ubuntu:latest
LABEL authors="artemhorkov"

ENTRYPOINT ["top", "-b"]