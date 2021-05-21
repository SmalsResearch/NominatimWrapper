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

In one command (for belgian data): 

`docker run -it --rm --shm-size=1g   
    -e PBF_URL=https://download.geofabrik.de/europe/belgium-latest.osm.pbf   \
    -e REPLICATION_URL=https://download.geofabrik.de/europe/belgium-updates/   \
    -e IMPORT_WIKIPEDIA=true   \
    -e NOMINATIM_PASSWORD=very_secure_password_1234   \
    -v nominatim-data:/var/lib/postgresql/12/main   \
    -p 8080:8080   
    --name nominatim 
    mediagis/nominatim:3.7`



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
    - Get photon jar : `wget https://github.com/komoot/photon/releases/download/0.3.4/photon-0.3.4.jar`
    - Copy it to the docker machine : `docker cp photon-0.3.4.jar nominatim:/`
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
        - Build the Photon data: `java -jar photon-*.jar -nominatim-import -host localhost -port 5432 -database nominatim -user nominatim -password mysecretpassword -languages en,fr,nl` (might require to install Java : sudo apt update ; sudo apt install default-jre)
        - Warning: folder "photon_data" should not exist before running this java command. Please remove it if needed.
        - Prepare the tar.gz file : `tar czf photon.tar.gz photon_data/`
        - `exit`
    - Get the tar.gz file : `docker cp nominatim:photon.tar.gz .`
    - Delete files (photon-0.3.1.jar and photon.tar.gz )
    
# Docker

Go to the repository root folder (NominatimWrapper), and copy the file "photon.tar.gz" in "Docker" folder (see above)

## Build

### Build with docker-compose (prefered version)
Split in three containers : Photon, Libpostal, and NominatimWrapper

`docker-compose -f docker-compose.yml build`

### One container
Only one container, embedding Libpostal, Photon and NominatimWrapper 

`docker build -f Docker/Dockerfile -t nominatim_wrapper --build-arg photon_data=Docker/photon.tar.gz .`


## Run 

Below, "nominatim_wrapper" is the name of our the docker image, "nomin_wrapper", the name of the container.

Nominatim IP can be get by running : 

`docker inspect nominatim |grep \"IPAd`


### With docker-compose (prefered)
In "docker-compose.yml", change "OSM_HOST=xx.xx.xx.xx:pppp" to reflect the address of the Nominatim server.

PHOTON_HOST and LIBPOSTAL_HOST do not need to be changed

Then, 

`docker-compose -f docker-compose.yml up`


### Without docker-compose
- To keep the default parameters:   `docker run -d  --name nomin_wrapper nominatim_wrapper`
- To change the default parameters: 
   -  Ex 1 (using internal instances of OSM and Photon) : 
   
     ` docker run -d --name nomin_wrapper  -e OSM_HOST=<nominatim_host>  -e NB_WORKERS=8 -e NB_LPOST_WORKERS=2  nominatim_wrapper`

   -  Ex 2 (using internal instances of OSM and your own instance of photon) : 
   
     ` docker run -d --name nomin_wrapper  -e OSM_HOST=<nominatim_host>  -e PHOTON_HOST=<photon_ip>:2322 -e NB_WORKERS=8 -e NB_LPOST_WORKERS=2  nominatim_wrapper`

   -  Ex 2 (using public instances): 
   
    ` docker run -d --name nomin_wrapper  -e OSM_HOST=nominatim.openstreetmap.org -e PHOTON_HOST=photon.komoot.de -e NB_WORKERS=8 -e NB_LPOST_WORKERS=2 nominatim_wrapper`

- To change port mapping : `docker run -d -p 7070:8080 -p 2322:2322 -p 5000:5000  -e OSM_HOST=172.17.0.2:8080 -e NB_WORKERS=8 nominatim_wrapper`


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

Single call:
`curl -X POST  "[docker ip]:5000/search/?street=chaussee+de+tervuren&city=Auderghem&postcode=1160"`


Batch call:
`curl -F media=@addresses.csv "http://[docker ip]:5000/batch/" -F mode=short`

Assuming "addresses.csv" has the following header:
"addr_key","country","postcode","city","street","housenumber"

Other columns are allowed, but will just be ignored (but return in the result if mode=long)

To pretty print output: 

`curl -F media=@address_sample100.csv "http://127.0.0.1:5000/batch/" -F mode=short | python -m json.tool`


options : 
- mode=geo|short|long (default: short):
    - geo: only return lat/long
    - short: return lat/long, cleansed address (street, number, zipcode, city, country)
    - long: return all results from Nominatim
- with_rejected=yes|no (only for batch version; default:no): return rejected records in a field "rejected" for each result
- check_result=yes|no (default:yes): 
    - yes: checks that result from nominatim is "close enough" from input
    - no: result the first result from nominatim, following the "transformer flow"
- struct_osm=yes|no (default:no): 
    - yes: use "structured" version of Nominatim (example: https://nominatim.openstreetmap.org/search.php?street=avenue+fonsny&city=bruxelles&postalcode=1060&format=jsonv2)
    - no: use "unstructured" version of Nominatim (example: https://nominatim.openstreetmap.org/search.php?q=avenue+fonsny,+1060+bruxelles&format=jsonv2)

