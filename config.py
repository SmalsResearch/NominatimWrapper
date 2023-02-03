# pylint: disable=invalid-name

"""
Parameters

@author: Vandy Berten (vandy.berten@smals.be)
"""

import logging

import os


check_with_transformed = True
# If True (and check_osm_results = True): osm result is compared to output of
#  transformer
# If False, osm result is compared to data input.
# True allows to accept more results (but also more false positive)

photon_check_results = True
# Makes sense only if either data is sent as structured (i.e., not the full
# address inside the same field), or if photon is preceded by libpostal

osm_structured = False

with_rest_libpostal = True

with_cleansed_number_on_26 = True

use_osm_parent = True

max_chunk_size = 10000
min_chunk_size = 100


env_osm_host = os.getenv('OSM_HOST')
if env_osm_host:
    logging.info("get OSM_HOST from env: %s", env_osm_host)
    osm_host = env_osm_host
else :
    osm_host = "10.0.2.15:7070"
    logging.info("Use default osm host: %s", osm_host)

env_photon_host = os.getenv('PHOTON_HOST')
if env_photon_host:
    logging.info("get PHOTON_HOST from env: %s", env_photon_host)
    photon_host = env_photon_host
else :
    photon_host = "localhost:2322"
    logging.info("Use default photon host: %s", photon_host)


env_lpost_host = os.getenv('LPOST_HOST')
if env_lpost_host:
    logging.info("get LPOST_HOST from env: %s", env_lpost_host)
    libpostal_host = env_lpost_host
else :
    libpostal_host = "localhost:8080"
    logging.info("Use default libpostal host: %s", libpostal_host)


street_field  = "streetName"
housenbr_field = "houseNumber"
postcode_field = "postCode"
city_field  =    "city"
country_field =  "country"
addr_key_field = "addrKey"

regex_replacements = {
    "init": [
        [street_field, r"^(.+)\(((AV[E .]|CH[A .]|RUE|BOU|B[LVD]+|PL[A .]|SQ|ALL|GAL)[^\)]*)\)$",
                       r"\g<2> \g<1>"],
        [street_field, r"[, ]*(SN|ZN)$", ""],
        [street_field, r"' ", "'"],
        [street_field,  r"\(.+\)$", ""],
    ],
    "lpost": [
        # Keep only numbers
        [housenbr_field, r"^([0-9]*)(.*)$", r"\g<1>"],

        # Av --> avenue, chée...

        [street_field, r"^r[\. ]",  "rue "],
        [street_field, r"^av[\. ]", "avenue "],
        [street_field, r"^ch([ée]e)?[\. ]", "chaussée "],
        [street_field, r"^b[lvd]{0,3}[\. ]",     "boulevard "],

        # rue d anvers --> rue d'anvers
        [street_field, r"(avenue|rue|chauss[ée]e|boulevard) d ", r"\g<1> d'"],
        [street_field, r"(avenue|rue|chauss[ée]e|boulevard) de l ", r"\g<1> de l'"],

        [street_field, " de l ", " de l'"]
    ]
}



similarity_threshold = 0.5

default_transformers_sequence = [ ["orig"],
                          ["regex[init]"],
                          ["nonum"],
                          ["libpostal", "regex[lpost]"],
                          ["libpostal", "regex[lpost]", "nonum"],
                          ["libpostal", "regex[lpost]", "photon"],
                          ["libpostal", "regex[lpost]", "photon", "nonum"],
                          ["photon"],
                          ["photon", "nonum"],
                          ["nostreet"]
                        ]

# Mapping of nominatim results fields on our output fields
collapse_params = {
    "addr_out_street":   ["road", "pedestrian","footway", "cycleway",
                          "path", "address27", "construction", "hamlet", "park"],
    "addr_out_city"  :   ["town", "village", "city_district", "county", "city"],
    "addr_out_number":   ["house_number"],
    "addr_out_country":  ["country"],
    "addr_out_postcode": ["postcode"],
}


city_test_from = "Bruxelles"
city_test_to = ["Bruxelles", "Brussels"]
