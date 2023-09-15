module.exports = function (app, base_uri) {
   app.get('/', function (req, res) {
      res.render('index.html')
   });

   const request = require('request');
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