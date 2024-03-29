# Context

This tools can be seen as a wrapper around Nominatim (OpenStreetMap geocoder). It sends addresses to Nominatim (local or public instance), and for those giving no results, try several ways to make the addresses acceptable by Nominatim by "cleaning it".

More explaination is given (in French) on: 
- https://www.smalsresearch.be/geocodage-contourner-les-lacunes-dopenstreetmap-partie-1/
- https://www.smalsresearch.be/geocodage-contourner-les-lacunes-dopenstreetmap-partie-2/

It requires Nominatim, Photon and Libpostal.

Nominatim might be provided separetely. 

We provide a Docker file (with docker-compose), allowing to build a set of images containing Photon and Libpostal, as well as our code. It is however possible to use your own instance of Photon, but we don't give any further details here.

# Build & Run

## Full build
With this option, a scripts "full_build.sh" will build Nominatim, extract from this container the data Photon will need, and then build Libpostal, Photon and NominatimWrapper containers.

### Build

- Edit docker-compose-full.yml to choose the right geographical zone (options PBF_URL and REPLICATION_URL)
- In NominatimWrapper folder, run `./full_build.sh`

### Run
`docker-compose -f docker-compose-full.yml up`

### Stop
`docker stop nominatimwrapper_wrapper_1  nominatimwrapper_photon_1  nominatimwrapper_libpostal_1  nominatim`

`docker rm nominatim` # otherwise next 'up' will (sometimes) fail. Why?? 

## Appart Nominatim

With this option, we build Nominatim in its own container, extract (manually) Photo data, and then build our Libpostal, Photon and NominatimWrapper as another group of containers using docker-compose

### Build Nominatim 

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
    mediagis/nominatim:4.0`
    

### Photon 

- Option 1: download worldwide data on https://github.com/komoot/photon/#installation. It takes ~53 Gb!!
- Option 2: Run the "PREPARE PHOTON DATA" block of "full_build.sh" script, adapting "NOMINATIM_CNT=nominatim" as needed

Both options provide a "photon.tar.gz" which has to be put in Docker folder

    
### Build NominatimWrapper

`docker-compose -f docker-compose.yml build`


### Run 

In "docker-compose.yml", change "OSM_HOST=xx.xx.xx.xx:pppp" to reflect the address of the Nominatim server given by:

`docker inspect nominatim |grep \"IPAd`

PHOTON_HOST and LIBPOSTAL_HOST do not need to be changed

Then, 

`docker-compose -f docker-compose.yml up`



# USAGE

## Photon

`curl -X GET  "[docker ip]:2322/api?q=chaussée+de+tervueren,+1160+Auderghem"`

## Libpostal

- Without port mapping : `curl -X POST  "[docker ip]:8080/parser?query=chaussée+de+tervueren,+1160+Auderghem"`
- With port mapping : `curl -X POST  "localhost:7070/parser?query=chaussée+de+tervueren,+1160+Auderghem"`

## REST API

Single call:
`curl -X 'POST' \
  'https://[docker ip]:5000/REST/nominatimWrapper/v1/geocode?mode=short&withRejected=false&checkResult=false&structOsm=false&extraHouseNumber=true' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "address": {
    "addrKey": "1",
    "streetName": "Avenue Fonsny",
    "houseNumber": "20",
    "postCode": "1160",
    "postName": "Bruxelles",
    "countryName": "Belgium"
  }
}'`


Batch call:
`curl -X 'POST' \
  'https://[docker ip]:5000/REST/nominatimWrapper/v1/batchGeocode?mode=short&withRejected=false&checkResult=false&structOsm=false&extraHouseNumber=true' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "addresses": [
    {
      "addrKey": "1",
      "streetName": "Avenue Fonsny",
      "houseNumber": "20",
      "postCode": "1160",
      "postName": "Bruxelles",
      "countryName": "Belgium"
    }, {
      "addrKey": "2",
      "streetName": "Rue de la loi",
      "houseNumber": "16",
      "postCode": "1000",
      "postName": "Bruxelles",
      "countryName": "Belgium"
    }

  ]
}'`

Full documentation is provide in the swagger file. Once the server is up&running, swagger UI is available on http://[docker ip]:5000/doc 

Options : 
- mode=geo|short|long (default: short):
    - geo: only return lat/long
    - short: return lat/long, cleansed address (street, number, zipcode, city, country)
    - long: return all results from Nominatim
- withRejected=true|false: return rejected records in a field "rejected" for each result
- checkResult=true|false (default:false): 
    - true: checks that result from nominatim is "close enough" from input
    - false: result the first result from nominatim, following the "transformer flow"
- structOsm=true|false (default:false): 
    - true: use "structured" version of Nominatim (example: https://nominatim.openstreetmap.org/search.php?street=avenue+fonsny&city=bruxelles&postalcode=1060&format=jsonv2)
    - false: use "unstructured" version of Nominatim (example: https://nominatim.openstreetmap.org/search.php?q=avenue+fonsny,+1060+bruxelles&format=jsonv2)
- extraHouseNumber=true|false (default: true). Often, OSM does not know exact positioning of a building, and locates only its street. In this case, "houseNumber" is empty. If house number contains a box, it may be removed in the output. This could be OK for geocoding, but not for data cleansing:
    - true: 3 extra fields are added in the input: "inHouseNumber" contains house number given in input ; "lpostHouseNumber" contains house number provided by libpostal receiving concatenation of street and house number (from input), and "lpostUnit" contains "unit" given by libpostal. If libpostal provides several "house numbers", they are joined by a ";".
    - false: extra fields are not computed. Avoid a call to libpostal for each address, which may improve performance if this information is not needed

## Quality indicators

Evaluating the quality of an anwser can be done in several ways. There are two aspects: 
- Reliability: are we confident about the answer?
- Precision: how small is the identified area? If the user provides a precise address, does the answer locates the building, the street, the city?



- Try first with "checkResult=true". If no result, try with “checkResult=false”. 
  If result in the first case, reliability is higher than in the second case.
- In result, check "method":
    - "orig" (or "fast"): address did not go through any transformer before being sent to OSM, which returns a result for this address. 
       Reliability is pretty good, especially if "check_result=true";
    - "nonum": "housenumber" field was not sent to OSM. Result can then not be at the house level:
    - "libpostal": address was transformed by libpostal;
    - "photon": address was transformed by photon.
- In result, check “place_rank”: 
    - 30 : precision is at house level;
    - 26 : precision is at street level;
    - 13-16 : city level;
    - 4 : country level.
    
## Fast mode

By default, single mode is actually converted in a batch mode with a single record. This drastically increases the overhead for all the cases where the response from Nominatim is already a good match.

When "FASTMODE" option is set to "yes" in docker-compose file (services > wrapper > environment) and "checkResult" is "false", a simple process is first tried. If Nominatim does not give any response, the full batch mode is started. Here are the steps being started (which roughly also corresponds to steps performed in batch mode): 

- The following address is sent to Nominatim : "street, houseNumber, postCode city, country" (if structOsm is 'false')
- If it gives results, we format all results, and set in "output" part: : 
    - "street":  first not null value in the following fields: ["road", "pedestrian","footway", "cycleway", "path", "address27", "construction", "hamlet", "park"]
    - "city": first value in ["town", "village", "city_district", "county", "city"],
    - "houseNumber":   "house_number",
    - "country":  "country",
    - "postCode": "postcode",
    - fields ["display_name", "place_id", "lat","lon", "place_rank"] as simply copied from Nominatim output
- "match" corresponds to the first result (with "method" set to "fast"), "reject" to all others (with "rejectReason" set to "tail") if any
- If place_rank in match record is below 30 and housenumber (in input) contains other characters than digits, we retry to call Nominatim by only considering the first digits of housenumber : "30A","30.3", "30 bt 2", "30-32" become "30". If it gives a result with place_rank = 30, we keep it (in this case, a "cleansedHouseNumber" appears in the output, with "30" in this example). Otherwise, we keep the original result 
- If extraHouseNumber is 'yes', we apply the method described above (in REST API > options) to enrich record with libpostal data. 
    
