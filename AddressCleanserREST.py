"""
Flask part of NominatimWrapper - DEPRECATED

@author: Vandy Berten (vandy.berten@smals.be)

"""

#!/usr/bin/env python
# coding: utf-8

# pylint: disable=invalid-name
# pylint: disable=line-too-long


"""
DEPRECATED!!!
"""

import os

import sys
import traceback

from datetime import datetime, timedelta
import time
import logging
import json


from flask import Flask,  request,jsonify

import pandas as pd

#import AddressCleanserUtils
from AddressCleanserUtils import (log, vlog,
                                  get_osm, parse_address, get_photon,
                                  addr_key_field, street_field, housenbr_field,
                                  postcode_field, city_field, country_field,
                                  process_address, process_addresses,
                                  update_timestats, timestats)


from config import osm_host, libpostal_host, photon_host


logger = logging.getLogger()

# WARNING : no logs
# INFO : a few logs
# DEBUG : lots of logs


env_log_level = os.getenv('LOG_LEVEL', "MEDIUM").upper().strip()
if env_log_level == "LOW":
    logger.setLevel(logging.WARNING)
elif env_log_level == "MEDIUM":
    logger.setLevel(logging.INFO)
elif env_log_level == "HIGH":
    logger.setLevel(logging.DEBUG)
else :
    print(f"Unkown log level '{env_log_level}'. Should be LOW/MEDIUM/HIGH")



# In[ ]:


with_timing = os.getenv('TIMING', "NO").upper().strip()
if with_timing == "NO":
    with_timing_info = False
elif with_timing == "YES":
    with_timing_info = True
else:
    print(f"Unkown TIMING '{with_timing}'. Should be YES/NO")
    with_timing_info = False
log(f"TIMING: {with_timing_info} ({with_timing})")


# In[ ]:


with_fastmode = os.getenv('FASTMODE', "NO").upper().strip()
if with_fastmode == "NO":
    fastmode = False
elif with_fastmode == "YES":
    fastmode = True
else:
    print(f"Unkown FASTMODE '{with_fastmode}'. Should be YES/NO")
    fastmode = False
log(f"FASTMODE: {fastmode} ({with_fastmode})")




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


transformers_sequence = os.getenv('TRANSFORMERS', "[none]")

if transformers_sequence == "[none]":
    transformers_sequence  =default_transformers_sequence

if isinstance(transformers_sequence, str):
    try:
        transformers_sequence = json.loads(transformers_sequence)
    except json.decoder.JSONDecodeError:
        log("Cannot parse TRANSFORMERS parameter... uses the default one")
        transformers_sequence = default_transformers_sequence



vlog("Transformers:")
vlog(transformers_sequence)



city_test_from = "Bruxelles"
city_test_to = ["Bruxelles", "Brussels"]




# Check that Nominatim server is running properly
# Adapt with the city of your choice!
delay=5
for i in range(10):
    osm = None
    try:
        osm = get_osm(city_test_from)
        assert city_test_to[0] in osm[0]["namedetails"]["name:fr"]

        log("Nominatim working properly")
        break

    except Exception:
        log("Nominatim not up & running")
        log(f"Try again in {delay} seconds")
        if osm is not None:
            log("Answer:")
            log(osm)

        log(f"Nominatim host: {osm_host}")

        #raise e
        time.sleep(delay)
if i == 9:
    log("Nominatim not up & running !")
    log(f"Nominatim: {osm_host}")



# Check that Libpostal is running properly
delay=5
for i in range(10):
    lpost=None
    try:
        lpost = parse_address(city_test_from)
        assert lpost[0][0].lower() == city_test_to[0].lower()
        log("Libpostal working properly")
        break

    except Exception:
        log("Libpostal not up & running ")
        log(f"Try again in {delay} seconds")
        if lpost is not None:
            log("Answer:")
            log(lpost)

        time.sleep(delay)
    #raise e
if i == 9:
    log("Libpostal not up & running !")
    log(f"Libpostal: {libpostal_host}")




# Check that Photon server is running properly
delay=5
for i in range(10):
    try:
        ph=""
        ph = get_photon("Bruxelles")
        assert city_test_to[0] in ph["features"][0]["properties"]["name"] or \
               city_test_to[1] in ph["features"][0]["properties"]["name"]
        log("Photon working properly")
        break


    except Exception:
        log("Photon not up & running ")
        log(f"Try again in {delay} seconds")
        log(ph)
        time.sleep(delay)

        #raise e
if i == 9:
    log("Photon not up & running ! ")
    log("Start it with 'nohup java -jar photon-*.jar &'")
    log(f"Photon host: {photon_host}")





def get_arg(argname, def_val):
    """
    Get argument from request form. Sometimes get it from request.form,
    sometimes from request.args.get

    Parameters
    ----------
    argname : str
        Argument name.
    def_val : str
        Default value.

    Returns
    -------
    str
        Argument value from request.

    """
    if argname in request.form:
        return request.form[argname]
    return request.args.get(argname, def_val)



def remove_empty_values(dct_lst):
    """
    Remove empty values in a list of dict

    Parameters
    ----------
    dct_lst : list (of dicts)
        List of dictionaries.

    Returns
    -------
    list
        Copy of input list, but all empty values in dicts are dropped

    """

    return [{k: v for k, v in item.items() if not pd.isnull(v) and v != ""} for item in dct_lst]



app = Flask(__name__)

@app.route('/search/', methods=['GET', 'POST'])
def search():
    """
    Geocode a single adress.
    Expected parameters:
        - address content:
            - street
            - housenumber
            - city
            - postcode
            - country
        - control flags:
            - no_reject: if True, rejected results are not returned
            - check_result: if True, will "double check" OSM results
            - struct_osm: if True, will call the structured version of OSM
            - extra_house_number: if True, will call libpostal on all addresses
                                  to get the house number

    Returns
    -------
    list of dictionaries, with all (OSM) result for the given address

    """

    t = datetime.now()

    for k in timestats:
        timestats[k]=timedelta(0)

    data= {street_field   : get_arg("street",      ""),
           housenbr_field : get_arg("housenumber", ""),
           city_field     : get_arg("city",        ""),
           postcode_field : get_arg("postcode",    ""),
           country_field  : get_arg("country",     ""),

          }
    no_reject = get_arg("no_reject", False)

    if get_arg("check_result", "yes") == "no":
        check_results = False
        log("Won't check OSM results")
    else:
        check_results = True
        log("Will check OSM results")

    if get_arg("struct_osm", "no") == "no":
        osm_structured = False
        log("Will call unstructured OSM")
    else:
        osm_structured = True
        log("Will call structured OSM")

    if get_arg("extra_house_nbr", "yes") == "no":
        with_extra_house_number = False
        vlog("Will skip extra house nbr")
    else:
        with_extra_house_number = True
        vlog("Will do extra house nbr")

#     update_timestats("check_args", t)

    res = process_address(data,
                          check_results=check_results,
                          osm_structured=osm_structured,
                          with_extra_house_number= with_extra_house_number,
                          fastmode=fastmode,
                          transformers_sequence=transformers_sequence)

    log(f"Input: {data}")
    log(f"Result: {res}")
    log(f"no_reject: {no_reject}")

    update_timestats("global", t)


    if with_timing_info:
        res["timing"] = {k: v.total_seconds()*1000 for k, v in timestats.items()}

    if no_reject :
        del res["reject"]


    return jsonify(res)



# Call to this : curl -F media=@address_sample100.csv http://127.0.0.1:5000/batch/ -X POST -F mode=long

@app.route('/batch/', methods=['POST'])
def batch():
    """
    Geocode all addresses in csv like file.


    Expected parameters:
        - address content: a CSV file with the following columns:
            - street
            - housenumber
            - postcode
            - city
            - country
            - addr_key (must be unique)
        - control flags:
            - mode:
                - geo:   result contains fields addr_key, lat, lon, place_rank, method
                - short: result contains fields addr_key, lat, lon, place_rank, method,
                            addr_out_street, addr_out_number,
                            in_house_nbr, lpost_house_nbr, lpost_unit,
                            addr_out_postcode, addr_out_city, addr_out_country
                - long: result contains all fields
            - with_rejected: if True, rejected results are returned in a "rejected" column
            - check_result: if True, will "double check" OSM results
            - struct_osm: if True, will call the structured version of OSM
            - extra_house_number: if True, will call libpostal on all addresses
                                  to get the house number

    Returns
    -------
    str
        json containing geocoded addresses.

    """
    log("batch")


    mode = "short"
    if "mode" in request.form :
        mode = request.form["mode"]
        if not mode in ["geo", "short", "long"]:
            return f'[{{"error": "Invalid mode {mode}"}}]'

    with_reject = False
    if "with_rejected" in request.form:
        if request.form["with_rejected"] == "yes":
            with_reject = True
            log("Will return rejects")
        elif request.form["with_rejected"] != "no":
            return f'[{{"error": "Invalid with_rejected {request.form["with_rejected"]}"}}]'

    if get_arg("check_result", "yes") == "no":
        check_results = False
        log("Won't check OSM results")
    else:
        check_results = True
        log("Will check OSM results")

    if get_arg("struct_osm", "no") == "no":
        osm_structured = False
        log("Will call unstructured OSM")
    else:
        osm_structured = True
        log("Will call structured OSM")

    if get_arg("extra_house_nbr", "yes") == "no":
        with_extra_house_number = False
        vlog("Will skip extra house nbr")
    else:
        with_extra_house_number = True
        vlog("Will do extra house nbr")



    key_name = (list(request.files.keys())[0])

    #print(request.files[0])

    df = pd.read_csv(request.files[key_name], dtype=str)
    log("Input: \n" + df.to_string(max_rows=10))

    mandatory_fields = [street_field, housenbr_field , postcode_field , city_field, country_field, addr_key_field]
    for field in mandatory_fields:
        if field not in df:
            return f"[{{\"error\": \"Field \'{field}\' mandatory in file. All mandatory fields are {';'.join(mandatory_fields)}\"}}]"

    if df[df[addr_key_field].duplicated()].shape[0]>0:
        return f"[{{\"error\": \"Field \'{field}\' cannot contain duplicated values!\"}}]"

    res, rejected_addresses = process_addresses(df,
                                                check_results=check_results,
                                                osm_structured=osm_structured,
                                                with_extra_house_number= with_extra_house_number,
                                                transformers_sequence=transformers_sequence)


    if isinstance(res, dict) :
        return {0:res}


    if res is None or res.shape[0] == 0:
        return '[]'

    try:
        if mode == "geo":
            res = res[[addr_key_field,"lat", "lon", "place_rank", "method"]]
        elif mode == "short":
            res = df.merge(res)[[addr_key_field,
                       "lat", "lon", "place_rank", "method", "place_id",
                       "addr_out_street", "addr_out_number", "in_house_nbr", "lpost_house_nbr", "lpost_unit", "addr_out_postcode", "addr_out_city",   "addr_out_country" ]]
        elif mode == "long":
            res = df.merge(res)


        if with_reject:
            rejected_rec = rejected_addresses.groupby(addr_key_field).apply(lambda rec: remove_empty_values(rec.to_dict(orient="records")) ).rename("rejected").reset_index()
            res = res.merge(rejected_rec, how="outer")


        res["lat"] = res["lat"].astype(float)
        res["lon"] = res["lon"].astype(float)
    except KeyError as e:
        log(f"Error during column selection: {e}")
        traceback.print_exc(file=sys.stdout)

    log("Output: \n"+res.iloc[:, 0:9].to_string(max_rows=9))

    return res.to_json(orient="records")
    #request.files:
