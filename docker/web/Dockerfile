
# docker build -t keti/image_retriever_web_demo:v1 .
# docker tag keti/image_retriever_web_demo:v1 10.0.0.161:5000/image_retriever_web_demo:v1
# docker push 10.0.0.161:5000/image_retriever_web_demo:v1

FROM node:10
LABEL maintainer "KETI AIRC sankim <kimsan0622@keti.re.kr>"

ENV EXPOSE_HTTP_PORT=8080
ENV EXPOSE_HTTPS_PORT=8443

# RETRIEVER_API_URL, for example:
ENV RETRIEVER_API_URL="http://111.111.111.111:5000/api/task"

WORKDIR /usr/src/app

# npm package json
COPY package*.json ./

# install npm packages which are indicated in package.json
RUN npm install

# copy all the files in current directory to working directory in container
COPY . .

# expose npm
EXPOSE ${EXPOSE_HTTP_PORT}
EXPOSE ${EXPOSE_HTTPS_PORT}

# run npm
CMD [ "npm", "start" ]

