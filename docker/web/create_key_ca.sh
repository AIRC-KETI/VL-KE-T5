openssl req -new -newkey rsa:4096 \
-days 3650 \
-nodes -x509 \
-keyout web.key \
-out web.crt
