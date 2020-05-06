# Context

This tools can be seen as a wrapper around Nominatim (OpenStreetMap geocoder). It sends addresses to Nominatim (local or public instance), and for those giving no results, try several ways to make the addresses acceptable by Nominatim by "cleaning it".

More explaination is given (in French) on: 
- https://www.smalsresearch.be/geocodage-contourner-les-lacunes-dopenstreetmap-partie-1/
- https://www.smalsresearch.be/geocodage-contourner-les-lacunes-dopenstreetmap-partie-2/

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
    - For Belgium, photon data takes 900 MB.
- To get the worldwide data : 
    - https://github.com/komoot/photon/#installation 
    - Takes ~53 Gb

- Prepare the data : 
     - Find folder where "photon_data" is
     - `tar czf  photon.tar.gz photon_data/`

- Assuming using mediagis/nominatim of above : 
    - Get photon jar : `wget https://github.com/komoot/photon/releases/download/0.3.1/photon-0.3.1.jar`
    - Copy it to the docker machine : `docker cp photon-0.3.1.jar nominatim:/`
    - Enter the docker machine : `docker exec -it nominatim bash`
    - Within the docker machine : 
        - Add a postgresql password : 
        ```
            su postgres
            psql
            ALTER USER nominatim WITH ENCRYPTED PASSWORD 'mysecretpassword';
            \q
            exit
         ```
        - Build the Photon data: `java -jar photon-*.jar -nominatim-import -host localhost -port 5432 -database nominatim -user nominatim -password mysecretpassword -languages en,fr,nl`
        - Warning: folder "photon_data" should not exist before running this java command. Please remove it if needed.
        - Prepare the tar.gz file : `tar czf photon.tar.gz photon_data/`
        - `exit`
    - Get the tar.gz file : `docker cp nominatim:photon.tar.gz .`
    - Delete it on the nominatim machine : `tar czf photon.tar.gz photon_data/`
    
# Docker

Go to the repository root folder (NominatimWrapper), and copy the file "photon.tar.gz" in "Docker" folder (see above)

## Build

Only one container, embedding Libpostal, Photon and NominatimWrapper 

`docker build -f Docker/Dockerfile -t nominatim_wrapper --build-arg photon_data=Docker/photon.tar.gz .`

### Build with docker-compose
Split in three containers : Photon, Libpostal, and NominatimWrapper

`docker-compose -f docker-compose.yml build`


### Light version

Based on "alpine" instead of Centos. Might be more difficult to build/use. Building time is much longer (many libraries have to be compiled), but image size is smaller (~6 GB vs 6.3 GB).

`docker-compose -f docker-compose_alp.yml build`

## Run 

Below, "nominatim_wrapper" is the name of our the docker image, "nomin_wrapper", the name of the container.

Nominatim IP can be get by running : 

`docker inspect nominatim |grep \"IPAd`


- To keep the default parameters:   `docker run -d  --name nomin_wrapper nominatim_wrapper`
- To change the default parameters: 
   -  Ex 1 (using internal instances of OSM and Photon) : 
   
     ` docker run -d --name nomin_wrapper  -e OSM_HOST=<nominatim_host>  -e NB_WORKERS=8 -e NB_LPOST_WORKERS=2  nominatim_wrapper`

   -  Ex 2 (using internal instances of OSM and your own instance of photon) : 
   
     ` docker run -d --name nomin_wrapper  -e OSM_HOST=<nominatim_host>  -e PHOTON_HOST=<photon_ip>:2322 -e NB_WORKERS=8 -e NB_LPOST_WORKERS=2  nominatim_wrapper`

   -  Ex 2 (using public instances): 
   
    ` docker run -d --name nomin_wrapper  -e OSM_HOST=nominatim.openstreetmap.org -e PHOTON_HOST=photon.komoot.de -e NB_WORKERS=8 -e NB_LPOST_WORKERS=2 nominatim_wrapper`

- To change port mapping : `docker run -d -p 7070:8080 -p 2322:2322 -p 5000:5000  -e OSM_HOST=172.17.0.2:8080 -e NB_WORKERS=8 nominatim_wrapper`

### With docker-compose 
In "docker-compose.yml", change "OSM_HOST=172.24.0.1:7070" to reflect the address of the Nominatim server.

Then, 

`docker-compose -f docker-compose.yml up`

## Run batch
- Adapt file config_batch.py according to your data file
- Copy config file and addresses within container: 
   - `docker cp config_batch.py  nomin_wrapper:/NominatimWrapper`
   - `docker cp address.csv.gz   nomin_wrapper:/`
- `docker exec -it nomin_wrapper python3 /NominatimWrapper/AddressCleanserBatch.py -c config_batch -a address.csv.gz`
- Other available options: 
   - '-s 1000': Take a sample of 1000 records
   - '-q': quiet (less outputs)
   - '-v': verbose (more outputs)

## Move

To build the docker image on a machine with Internet access ("build machine") and the run it on another one, without internet access ("run machine") :
- On the "build machine":  `docker save nominatim_wrapper | gzip >nominatim_wrapper.tar.gz`
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

