# WhatsApp Bot

## Stack
PyWa
Serveo

## Activar Serveo
Estamos usando https://serveo.net/ para crear el tunel SSH. Para activar el tunel, hay que correr el siguiente comando:

```bash
ssh -R detoxigram.serveo.net:80:localhost:8080 serveo.net
```

Pra configurar bien esto, fue másc complicado q eso. Tuve que:
- Crear una SSH public key
- 
```bash
ssh-keygen -t rsa -b 4096 -C "detoxi_id"
``` 
te va a tirar.
Generating public/private rsa key pair. 
Enter file in which to save the key (/Users/luzalbaposse/.ssh/id_rsa):  -> aca das enter
Enter passphrase (empty for no passphrase):  _> metes una pass
Enter same passphrase again:  - > pass
Your identification has been saved in /Users/luzalbaposse/.ssh/id_rsa
Your public key has been saved in /Users/luzalbaposse/.ssh/id_rsa.pub
The key fingerprint is:
SHA256:CUgCadkC/zwNlTFM1cFMz5nR6ej7vZLJSWUzhsJ+zl4 detoxi_id
The key's randomart image is:
bla bla


y despues corres: ssh -i /.ssh/id_rsa -R detoxigram.serveo.net:80:localhost:8080 serveo.net

ahi te va a pedir q te registres  con google y te va a dar un link para verificar tu cuenta.

- cortas ese tunel
- Iniciar de nuevo el tunel

Si hay errores, hay que chequear que uvicorn esté en el puerto 8080 y que el tunel esté bien configurado.

-> si tira un Get bla bla bal challenge 200 OK está bien : ) 

Entonces, para arrancar los servers:
uvicorn wa:fastapi_app --host 0.0.0.0 --port 8080

ssh -i ~/.ssh/id_rsa -R detoxigram.serveo.net:80:localhost:8080 serveo.net


primero hay q correr serveo y despues uvicorn


Problema q estoy teniendo ahora: no puedo mandar mensajes a pesar de q el numero esté registrado.

> en testing, anda mal esto. apenas lo pasemos a producción, voy a tener q cambiar unas cosas del codigo.

Corres wa.py con uvicorn wa:fastapi_app --host 0.0.0.0 --port 8080

