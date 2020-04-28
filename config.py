with_dask = True
check_osm_results = True

check_with_transformed = True 
# If True (and check_osm_results = True): osm result is compared to output of transformer
# If False, osm result is compared to data input.
# True allows to accept more results (but also more false positive)


with_rest_libpostal = True

with_cleansed_number_on_26 = True

use_osm_parent = True

max_chunk_size = 10000
min_chunk_size = 100


import logging

import os
env_osm_host = os.getenv('OSM_HOST')
if env_osm_host: 
    logging.info(f"get OSM_HOST from env: {env_osm_host}")
    osm_host = env_osm_host
else : 
    osm_host = "10.0.2.15:7070"
    logging.info(f"Use default osm host : {osm_host}")

env_photon_host = os.getenv('PHOTON_HOST')
if env_photon_host: 
    logging.info(f"get PHOTON_HOST from env: {env_photon_host}")
    photon_host = env_photon_host
else : 
    photon_host = "localhost:2322"
    logging.info(f"Use default photon host: {photon_host}")


env_lpost_host = os.getenv('LPOST_HOST')
if env_lpost_host: 
    logging.info(f"get LPOST_HOST from env: {env_lpost_host}")
    libpostal_host = env_lpost_host
else : 
    libpostal_host = "localhost:8080"
    logging.info(f"Use default libpostal host: {photon_host}")



