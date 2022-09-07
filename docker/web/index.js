const express = require('express');
const bodyParser = require('body-parser');
const http = require('http');
const https = require('https');
const fs = require('fs');

const HTTP_PORT = 8080;
const HTTPS_PORT = 8443;

const options = {
    key: fs.readFileSync('./web.key'),
    cert: fs.readFileSync('./web.crt')
};

var app = express();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

app.use(express.static(__dirname + '/public/'));

app.set('views', __dirname + '/views');
app.set('view engine', 'ejs');
app.engine('html', require('ejs').renderFile);

const base_uri = process.env.RETRIEVER_API_URL
console.log("RETRIEVER_API_URL: ", base_uri)
var router = require('./router/main')(app, base_uri);

// Create an HTTP server.
http.createServer(app).listen(HTTP_PORT, function (){
    console.log("Express server has started on port" + HTTP_PORT);
});

// Create an HTTPS server.
https.createServer(options, app).listen(HTTPS_PORT, function(){
    console.log("Express server has started on port" + HTTPS_PORT)
});

// var server = app.listen(3000, function () {
//     console.log("Express server has started on port 3000")
// })


