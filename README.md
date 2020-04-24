# Context

This tools can be seen as a wrapper around Nominatim (OpenStreetMap geocoder). It sends addresses to Nominatim (local or public instance), and for those giving no results, try several ways to make the addresses acceptable by Nominatim by "cleaning it".

It requires Nominatim, Photon and Libpostal.

Nominatim should be provided separetely. 

We provide a Docker file, allowing to build an image containing Photon and Libpostal, as well as our code.

Several configurations are possible : 

1. Using our Python script, connecting to your own Photon and Libpostal instance (no further details given here)
2. Using our Docker image, and run our batch script within the docker image 
3. Using our Docker image, and use our REST API

## Nominatim 

More info on https://hub.docker.com/r/mediagis/nominatim/

- Get the data (pbf files. Exemple with Belgian data) :  

    - `mkdir nominatimdata ; cd nominatimdata `
    - `wget http://download.geofabrik.de/europe/belgium-latest.osm.pbf `
- Build/run docker: 
    - `docker pull mediagis/nominatim`
    - `docker run -t -v ~/nominatimdata:/data mediagis/nominatim  sh /app/init.sh /data/belgium-latest.osm.pbf postgresdata 4`

- Run : 
    - `docker run --restart=always -p 6432:5432 -p 7070:8080 -d --name nominatim -v ~/nominatimdata/postgresdata:/var/lib/postgresql/11/main mediagis/nominatim bash /app/start.sh`

## Photon 

### Using your own instance : 

See https://github.com/komoot/photon/#installation 

### Using our Docker

Our Docker build downloads Photon code, but you need to provide Data separetely. 

- To build the Elastic Search DB from the local Nominatim server: 
    - https://github.com/komoot/photon/#customized-search-data 
    - Should be done on the Nominatim machine! 
    - For Belgium, photon data takes ~1.7 GB.
- To get the worldwide data : 
    - https://github.com/komoot/photon/#installation 
    - Takes ~53 Gb

- Prepare the data : 
     - Find folder where "photon_data" is
     - `tar czf  photon.tar.gz photon_data/`


# Docker

Go to the repository root folder (nominatim_wrapper), and copy the file "photon.tar.gz" (see above)

## Build

`docker build -f Docker/Dockerfile -t nominatim_wrapper --build-arg photon_data=photon.tar.gz .`

### Build without embedding Photon

`docker build -f Docker/Dockerfile_nophoton -t nominatim_wrapper_nophoton`


## Run 

Below, "nominatim_wrapper" is the name of our the docker image, "nomin_wrapper", the name of the container

- To keep the default parameters:   `docker run -d  --name nomin_wrapper nominatim_wrapper`
- To change the default parameters: 
   -  Ex 1 (using internal instances of OSM and Photon) : 
   
     ` docker run -d --name nomin_wrapper  -e OSM_HOST=<nominatim_host>  -e NB_WORKERS=8 -e NB_LPOST_WORKERS=2  nominatim_wrapper`

   -  Ex 2 (using internal instances of OSM and your own instance of photon) : 
   
     ` docker run -d --name nomin_wrapper  -e OSM_HOST=<nominatim_host>  -e PHOTON_HOST=<photon_ip>2322 -e NB_WORKERS=8 -e NB_LPOST_WORKERS=2  nominatim_wrapper`

   -  Ex 2 (using public instances): 
   
    ` docker run -d --name nomin_wrapper  -e OSM_HOST=nominatim.openstreetmap.org -e PHOTON_HOST=photon.komoot.de -e NB_WORKERS=8 -e NB_LPOST_WORKERS=2 nominatim_wrapper`

- To change port mapping : `docker run -d -p 7070:8080 -p 2322:2322 -p 5000:5000  -e OSM_HOST=172.17.0.2:8080 -e NB_WORKERS=8 nominatim_wrapper`


## Run batch

- Get IP Address of Nominatim (if using docker) : `docker inspect nominatim |grep \"IPAd`
- Get IP Address of Photon/Libpostal :  `docker inspect nomin_wrapper |grep \"IPAd`
- Adapt file config_Batch.py accordingly (photon_host, libpostal_host, osm_host)
- Copy config file and addresses within container: 
   - `docker cp config_Batch.py  nomin_wrapper:/nominatim_wrapper`
   - `docker cp address.csv.gz   nomin_wrapper:/`
- `docker exec -it nomin_wrapper python3 /nominatim_wrapper/AddressCleanserBatch.py -c config_Batch -a address.csv.gz`
 

## Move

To build the docker image on a machine with Internet access ("build machine") and the run it on another one, without internet access ("run machine") :
- On the "build machine":  `docker save  nominatim_wrapper | gzip >nominatim_wrapper.tar.gz`
- Transfer file nominatim_wrapper.tar.gz to the "run machine"
- On the "run machine":  `docker load < nominatim_wrapper.tar.gz`

# TEST

## Photon

`curl -X GET  "[docker ip]:2322/api?q=chaussée+de+tervueren,+1160+Auderghem"`

## Libpostal

- Without port mapping : `curl -X POST  "[docker ip]:8080/parser?query=chaussée+de+tervueren,+1160+Auderghem"`
- With port mapping : `curl -X POST  "localhost:7070/parser?query=chaussée+de+tervueren,+1160+Auderghem"`

## REST API

`curl -X POST  "[docker ip]:5000/search/?street=chaussee+de+tervuren&city=Auderghem&postcode=1160"`

# TODO 
- supprimer "with_dask"
- use_osm_parent à la bonne place
- country (avec default country)
- return city if street not found
- osm_keep_relevant_results : sur base de l'input ou du dernier transform ?

