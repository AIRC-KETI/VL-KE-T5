module.exports = function (app) {
   app.get('/', function (req, res) {
      res.render('index.html')
   });

   const request = require('request');
   const base_uri = "http://10.1.92.1:5000/api/task"
   app.post('/search', function(req, res){

       const options = {
         uri:base_uri, 
         method: 'POST',
         body: req.body,
         json:true
       }

       request.post(options, function(err,httpResponse,body){
         res.writeHead("200", {"Content-Type":"application/json; charset=UTF-8"});
         res.write(JSON.stringify(body));
         res.end();
       })
      
      
      
   });
}