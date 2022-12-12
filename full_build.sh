#!/bin/bash

# check that docker is installed
if ! command -v docker &> /dev/null
then
    echo "'docker' not found, please install if first"
    exit
fi

# check that docker-composed is installed

if ! command -v docker-compose &> /dev/null
then
    echo "'docker-compose' not found, please install if first"
    exit
fi

# check that docker is up&running and user has access

if ! docker ps &> /dev/null
then
    echo "Cannot run 'docker ps' ; please check your config"
    exit
fi


NOMINATIM_CNT=nominatim
date
echo 
echo "#####################"
echo "## BUILD NOMINATIM ##"
echo "#####################"
echo 

# docker-compose -f docker-compose-full.yml up -d $NOMINATIM_CNT
# 
# echo "Waiting for nominatim container to be ready (type 'docker logs -f $NOMINATIM_CNT' to follow progression) ..."
# (docker logs $NOMINATIM_CNT -f 2>&1 & ) | grep -q "database system is ready to accept connections"
# 
# docker logs $NOMINATIM_CNT
# 
# docker ps 
# 
# if docker ps |grep -q "mediagis/nominatim"; 
# then
#     echo "nominatim is running"
# else
#     echo "nominatim is not running please check logs!"
#     exit
# fi


echo 
date
echo "#########################"
echo "## PREPARE PHOTON DATA ##"
echo "#########################"
echo 

wget --progress=dot:mega https://github.com/komoot/photon/releases/download/0.3.5/photon-0.3.5.jar

docker cp photon-0.3.5.jar $NOMINATIM_CNT:/
rm photon-0.3.5.jar

# Set Postgres password
docker exec -it $NOMINATIM_CNT su postgres -c "psql -c \"ALTER USER nominatim WITH ENCRYPTED PASSWORD 'mysecretpassword'\" "

# Insall Java 
docker exec -it $NOMINATIM_CNT sudo apt update # sudo ?
docker exec -it $NOMINATIM_CNT sudo apt install default-jre -y

# Build photo data
docker exec -it $NOMINATIM_CNT java -jar /photon-0.3.5.jar -nominatim-import -host localhost -port 5432 -database nominatim -user nominatim -password mysecretpassword -languages en,fr,nl
 
# Archive photon data
docker exec -it $NOMINATIM_CNT tar czf /photon.tar.gz photon_data/

# Download photon data
docker cp $NOMINATIM_CNT:/photon.tar.gz Docker

# Remove work data
docker exec -it $NOMINATIM_CNT rm -rf /photon.tar.gz /photon-0.3.5.jar photon_data

# Shutdown nominatim

docker stop $NOMINATIM_CNT
docker rm $NOMINATIM_CNT # Otherwise next "up" fails (why ???)

echo 
date
echo "############################"
echo "## BUILD NOMINATIMWRAPPER ##"
echo "############################"
echo 

# docker-compose -f docker-compose-full.yml build
