#!/bin/bash

echo "STARTING start.sh ..."

if [ X$1 = "Xnophoton" ] 
then 
   echo "No photon"
else
   echo "Running Photon"
   cd /photon
   java -jar photon-*.jar &
fi

echo "Running libpostal REST service"

cd /nominatim_wrapper

gunicorn -w ${NB_LPOST_WORKERS:-1} -b 0.0.0.0:8080 wsgi_libpostal:app &  

echo "Running REST service"
cd /nominatim_wrapper


gunicorn -w ${NB_WORKERS:-1} -b 0.0.0.0:5000 -e OSM_HOST=${OSM_HOST} AddressCleanserREST:app


