#!/bin/bash

echo "STARTING start.sh ..."

if [[ X$1 == "X" ]]; then 
    what="photon;libpostal;wrapper"
else
    what=$1
fi
   
if [[ $what == *"photon"* ]]; then
   echo "Running Photon"
   cd /photon
   java -jar photon-*.jar &
fi


if [[ $what == *"libpostal"* ]]; then
    echo "Running libpostal REST service"
    cd /NominatimWrapper
    gunicorn -w ${NB_LPOST_WORKERS:-1} -b 0.0.0.0:8080 LibpostalREST:app &
fi

if [[ $what == *"wrapper"* ]]; then
    echo "Running REST service"
    cd /NominatimWrapper
    gunicorn -w ${NB_WORKERS:-1} -b 0.0.0.0:5000 -e OSM_HOST=${OSM_HOST} AddressCleanserREST:app &
fi


while :; do sleep 3600 ; done

echo "END ... (shouldn't be seen)"