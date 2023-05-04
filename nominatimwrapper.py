#!/usr/bin/env python
# coding: utf-8

"""
Flask part of NominatimWrapper

@author: Vandy Berten (vandy.berten@smals.be)

"""



# pylint: disable=line-too-long
# pylint: disable=invalid-name


# TODO :
# - SSL ?
# - "namespace" is empty
# - Full address en batch
# - split/reorganise util.py
# - according to 'mode', avoid to compute useless things


import os

import sys
import traceback

from datetime import datetime, timedelta
import time
import logging
import json

import gzip


from flask import Flask,  request, url_for
from flask_restx import Api, Resource, reqparse

import werkzeug

import pandas as pd

logging.basicConfig(format='[%(asctime)s]  %(message)s', stream=sys.stdout) # does not work if I put if after the next import...


from base import (log, vlog, get_osm, get_photon, update_timestats, timestats)


from utils import (parse_address,
                                  addr_key_field, street_field, housenbr_field,
                                  postcode_field, city_field, country_field,
                                  process_address, process_addresses,
                                   to_camel_case,
                   #               convert_street_components,
                                   # remove_empty_values,
                                   multiindex_to_dict)


from config import osm_host, libpostal_host, photon_host, default_transformers_sequence, city_test_from, city_test_to


# logger = logging.getLogger()

# WARNING : no logs
# INFO : a few logs
# DEBUG : lots of logs

logger = logging.getLogger()

env_log_level = os.getenv('LOG_LEVEL', "MEDIUM").upper().strip()
if env_log_level == "LOW":
    logger.setLevel(logging.WARNING)
elif env_log_level == "MEDIUM":
    logger.setLevel(logging.INFO)
elif env_log_level == "HIGH":
    logger.setLevel(logging.DEBUG)
else :
    print(f"Unkown log level '{env_log_level}'. Should be LOW/MEDIUM/HIGH")

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)




with_timing = os.getenv('TIMING', "NO").upper().strip()
if with_timing == "NO":
    with_timing_info = False
elif with_timing == "YES":
    with_timing_info = True
else:
    print(f"Unkown TIMING '{with_timing}'. Should be YES/NO")
    with_timing_info = False
log(f"TIMING: {with_timing_info} ({with_timing})")


with_fastmode = os.getenv('FASTMODE', "NO").upper().strip()
if with_fastmode == "NO":
    fastmode = False
elif with_fastmode == "YES":
    fastmode = True
else:
    print(f"Unkown FASTMODE '{with_fastmode}'. Should be YES/NO")
    fastmode = False
log(f"FASTMODE: {fastmode} ({with_fastmode})")


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


def check_osm():
    try:
        osm_res = get_osm(city_test_from)
        if city_test_to[0] in osm_res[0]["namedetails"]["name:fr"]:
            return True # Everything is fine
        return osm  # Server answers, but gives an unexpected result
    except Exception as exc:
        vlog("Exception occured: ")
        vlog(exc)
        return False    # Server does not answer

def check_libpostal():
    try:
        lpost_res = parse_address(city_test_from)
        if lpost_res[0][0].lower() == city_test_to[0].lower():
            return True # Everything is fine
        return osm  # Server answers, but gives an unexpected result
    except Exception as exc:
        vlog("Exception occured: ")
        vlog(exc)
        return False    # Server does not answer


def check_photon():
    try:
        photon_res = get_photon("Bruxelles")
        if city_test_to[0] in photon_res["features"][0]["properties"]["name"] or \
               city_test_to[1] in photon_res["features"][0]["properties"]["name"]:
            return True # Everything is fine
        return osm  # Server answers, but gives an unexpected result
    except Exception as exc:
        vlog("Exception occured: ")
        vlog(exc)
        return False    # Server does not answer



# Check that Nominatim server is running properly
# Adapt with the city of your choice!
delay=2
for i in range(10):
    osm = check_osm()
    if osm is True:
        log("Nominatim working properly")
        break
    log("Nominatim not up & running")
    log(f"Try again in {delay} seconds")
    if osm is not False:
        log("Answer:")
        log(osm)

        log(f"Nominatim host: {osm_host}")

        #raise e
    time.sleep(delay)
    delay+=0.5
if i == 9:
    log("Nominatim not up & running !")
    log(f"Nominatim: {osm_host}")



# Check that Libpostal is running properly
delay=2
for i in range(10):
    lpost=check_libpostal()
    if lpost is True:
        log("Libpostal working properly")
        break
    log("Libpostal not up & running ")
    log(f"Try again in {delay} seconds")
    if lpost is not False:
        log("Answer:")
        log(lpost)

    time.sleep(delay)
    delay+=0.5
    #raise e
if i == 9:
    log("Libpostal not up & running !")
    log(f"Libpostal: {libpostal_host}")




# Check that Photon server is running properly
delay=2
for i in range(10):
    phot=check_photon()
    if phot is True:
        log("Photon working properly")
        break
    log("Photon not up & running ")
    log(f"Try again in {delay} seconds")
    if phot is not False:
        log(phot)
    time.sleep(delay)
    delay+=0.5

        #raise e
if i == 9:
    log("Photon not up & running ! ")
    log("Start it with 'nohup java -jar photon-*.jar &'")
    log(f"Photon host: {photon_host}")


log("Waiting for requests...")



def convert_bool(value):
    if isinstance(value, str):
        if value.lower()=="false":
            return False
        if value.lower()=="true":
            return True
    return value
        
    
    
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
        return convert_bool(request.form[argname])
    return convert_bool(request.args.get(argname, def_val))


app = Flask(__name__)
api = Api(app,
          version='0.1',
          title='NominatimWrapper API',
          description="""A service that allows geocoding (postal address cleansing and conversion into geographical coordinates), based on Nominatim (OpenStreetMap).
          
          Source available on: https://github.com/SmalsResearch/NominatimWrapper/

          """,
          doc='/doc',
          prefix='/REST/nominatimWrapper/v0.1',
          contact='Vandy BERTEN',
          contact_email='vandy.berten@smals.be',
          contact_url='https://www.smalsresearch.be/author/berten/'
)

namespace = api.namespace(
    '',
    'Main namespace')

# api.add_namespace(namespace)



with_https = os.getenv('HTTPS', "NO").upper().strip()

if with_https=="YES":
    # It runs behind a reverse proxy
    @property
    def specs_url(self):
        return url_for(self.endpoint('specs'), _external=True, _scheme='https')

    Api.specs_url = specs_url



single_parser = reqparse.RequestParser()
single_parser.add_argument('streetName',      type=str, help='Street name')
single_parser.add_argument('houseNumber', type=str, help='House number')
single_parser.add_argument('city',        type=str, help='City name')
single_parser.add_argument('postCode',    type=str, help='Postal code')
single_parser.add_argument('country',     type=str, help='Country name')
single_parser.add_argument('addrKey',     type=str, help='Address key (optional, simply copied in output)')
single_parser.add_argument('fullAddress',     type=str, help='Full address in a single field')

single_parser.add_argument('mode',
                          type=str,
                          choices=('geo', 'short', 'long'),
                          default='short',
                          help="""
Selection of columns in the ouput :

- geo: only return lat/long
- short: return lat/long, cleansed address (street, number, zipcode, city, country)
- long: return all results from Nominatim""")


single_parser.add_argument('withRejected',
                           type=str,
                           choices=(True, False),
                           default=False,
                           help='If "true", rejected results are returned')
single_parser.add_argument('checkResult',
                           type=str,
                           choices=(True, False),
                           default=False,
                           help='If "true", will "double check" OSM results')
single_parser.add_argument('structOsm',
                           type=str,
                           choices=(True, False),
                           default=False,
                           help='If "true", will call the structured version of OSM')
single_parser.add_argument('extraHouseNumber',
                           type=str,
                           choices=(True, False),
                           default=True,
                           help='If "true", will call libpostal on all addresses to get the house number')

@namespace.route('/geocode')
class Geocode(Resource):
    """ Single address geocoding"""
    @namespace.expect(single_parser)
    @namespace.response(400, 'Error in arguments')
    @namespace.response(500, 'Internal Server error')
    @namespace.response(200, 'Found a match for this address (or some rejected addresses)')
    @namespace.response(204, 'No address found, even rejected')


    def post(self):
        """
        Geocode (postal address cleansing and conversion into geographical coordinates) a single address.
        
        Cf GET version
        """
        return self.get()

    @namespace.expect(single_parser)
    @namespace.response(400, 'Error in arguments')
    @namespace.response(500, 'Internal Server error')
    @namespace.response(200, 'Found a match for this address (or some rejected addresses)')
    @namespace.response(204, 'No address found, even rejected')



    def get(self):
        """
Geocode a single address.

Returns
-------
Returns a dictionary with 2 parts. Depending of parameter "mode", various fields could be found.

In 'long' mode, each record will contain the following blocs:

- match: a single result, with the following blocs:
    - input : all columns present in input data, but at least "addrKey" (if provided), "streetName", "houseNumber", "postCode", "city", "country"
    - output: consolidated result of geocoding :
        - streetName: first non null value in ["road", "pedestrian","footway", "cycleway", "path", "address27", "construction", "hamlet", "park"]
        - houseNumber: house_number
        - postCode: postcode
        - city: first non null value in ["town", "village", "city_district", "county", "city"],
        - country: country
        - other: concatenate all values which were not picked by one of the above item
        - inHouseNumber: equivalent to input->houseNumber
        - lpostHouseNumber: "housenumber" provided by libpostal receiving concatenation of street and house number (from input)
        - lpostUnit: "unit"  provided by libpostal receiving concatenation of street and house number (from input)
    - work: some metadata describing geocoding process:
        - transformedAddress: what address (after possibly some sequence of transformations) is actually sent to Nominatim
        - method: which transformation methods were used before sending the address to Nominatim. If the address was found without any transformation, will be "orig" (or "fast")
        - osmOrder: what was the rank of this result in Nominatim result (more useful in 'rejected' part)
        - retryOn_26: If placeRank in match record is below 30 and housenumber (in input) contains other characters than digits, we retry to call Nominatim by only considering the first digits of housenumber: "30A","30.3", "30 bt 2", "30-32" become "30". If it gives a result with place_rank = 30, we keep it (in this case, a "cleansedHouseNumber" appears in the output, with "30" in this example), and this field is set to "True"
    - nominatim: selection of fields received from Nominatim:
        - lat
        - lon
        - placeRank
        - displayName
        - all fields in the "address" bloc
    - check:  Check results indicators (if checkResult='true'):
        - SIMStreetWhich
        - SIMStreet
        - SIMCity
        - SIMZip
        - SIMHouseNumber

- reject: list of rejected results, with most of the same fields, with additionnal fields:
     - rejectReason: 'mismatch" or "tail"
     - distToMatch: distance (in kilometer) to the result given in "match"

In 'geo' mode: only 'lat', 'lon', and 'placeRank' values from 'nominatim', 'addrKey' from 'input', and 'method' from 'work'

In 'short' mode: idem as 'geo', plus full 'output' bloc

        """

        start_time = datetime.now()

        for k in timestats:
            timestats[k]=timedelta(0)

        address = get_arg("fullAddress",  "")
        data= {
               street_field   : get_arg(street_field[1],      ""),
               housenbr_field : get_arg(housenbr_field[1],    ""),
               city_field     : get_arg(city_field[1],        ""),
               postcode_field : get_arg(postcode_field[1],    ""),
               country_field  : get_arg(country_field[1],     ""),
               addr_key_field  : get_arg(addr_key_field[1],     ""),
              }

        mode = get_arg("mode", "short")
        if not mode in ["geo", "short", "long"]:
            return [{"error": f"Invalid mode {mode}"}], 400



        used_fields = list(filter(lambda x: data[x]!="", data))

        vlog(f"used_fields: {used_fields}")

        error_msg = "Invalid value for '%s'. Possible values are 'true' or 'false' (received '%s')"
        with_rejected = get_arg("withRejected", False)
        if not with_rejected in [True, False]:
            return [{"error": error_msg%('withRejected', with_rejected)}], 400

        check_results = get_arg("checkResult", False)
        if not check_results in [True, False]:
            return [{"error": error_msg%('checkResult', check_results)}], 400

        osm_structured = get_arg("structOsm", False)
        if not osm_structured in [True, False]:
            return [{"error": error_msg%('structOsm', osm_structured)}], 400

        with_extra_house_number =  get_arg("extraHouseNumber", True)
        if not with_extra_house_number in [True, False]:
            return [{"error": error_msg%('extraHouseNumber', with_extra_house_number)}], 400

        if address != "":
            if len(used_fields)>0:
                return [{"error": "Field 'fullAddress' cannot be used together with fields "+";".join(used_fields)}],  400
            if osm_structured :
                return [{"error": "Field 'fullAddress' cannot be used together with fields 'structOsm=true'"}],   400
            if check_results :
                return [{"error": "Field 'fullAddress' cannot be used together with fields 'checkResult=true'"}], 400

            data[street_field] = address


        log("Parameters:")
        log(f"with_rejected:   {with_rejected}")
        log(f"check_results:   {check_results}")
        log(f"osm_structured:  {osm_structured}")
        log(f"extra_house_nbr: {with_extra_house_number}")
        log(f"mode:            {mode}")


        log(f"Input: {data}")

        res = process_address(data,
                              check_results=check_results,
                              osm_structured=osm_structured,
                              with_extra_house_number= with_extra_house_number,
                              fastmode=fastmode,
                              transformers_sequence=transformers_sequence,
                              with_rejected=with_rejected
                             )


        log(f"Result: {res}")

        def filter_dict(dict_data, fields):
            return { f1: {f2:dict_data[f1][f2] for f2 in fields[f1] if f2 in dict_data[f1] } for f1 in fields if f1 in dict_data}
        try:

            if mode == "geo":
                fields={addr_key_field[0]:[addr_key_field[1]], "nominatim": ["lat", "lon", "place_rank"], "work": ["method", "reject_reason"]}

            elif mode == "short":
                fields={addr_key_field[0]:[addr_key_field[1]],
                        "nominatim": ["lat", "lon", "place_rank"],
                        "work": ["method","reject_reason", "dist_to_match"],
                        "output": res["match"][0]["output"]}

            elif mode == "long":
                fields = None

            if fields:
                res["match"] =    [filter_dict(res["match"][0], fields)]

            if not with_rejected and "rejected" in res:
                del res["rejected"]
            elif fields:
                res["rejected"] = [filter_dict(rec, fields) for rec in res["rejected"]]

            log(f"Result after selection: {res}")

        except KeyError as ex:
            log(f"Error during column selection: {ex}")


        update_timestats("global", start_time)

        if with_timing_info:
            res["timing"] = {k: v.total_seconds()*1000 for k, v in timestats.items()}


        if "error" in res:
            return res, 500

        return_code = 200 if ("match" in res and len(res["match"])>0) or ("rejected" in res and len(res["rejected"])>0) else 204

        return to_camel_case(res), return_code



# Call to this : curl -F media=@address_sample100.csv http://127.0.0.1:5000/batch/ -X POST -F mode=long
batch_parser = reqparse.RequestParser()
batch_parser.add_argument('csv file',
                          type=werkzeug.datastructures.FileStorage,
                          location='files',
                          help="""
A CSV file with the following columns:

- streetName
- houseNumber
- postCode
- city
- country
- addrKey (must be unique)""")

batch_parser.add_argument('mode',
                          type=str,
                          choices=('geo', 'short', 'long'),
                          default='short',
                          help="""
Selection of columns in the ouput :

- geo: only return lat/long
- short: return lat/long, cleansed address (street, number, zipcode, city, country)
- long: return all results from Nominatim""")

batch_parser.add_argument('withRejected',
                          type=str,
                          choices=(True, False),
                          default=False,
                          help='if "true", rejected results are returned')

batch_parser.add_argument('checkResult',
                          type=str,
                          choices=(True, False),
                          default=False,
                          help='if "true", will "double check" OSM results')
batch_parser.add_argument('structOsm',
                          type=str,
                          choices=(True, False),
                          default=False,
                          help='if "true", will call the structured version of OSM')
batch_parser.add_argument('extraHouseNumber',
                          type=str,
                          choices=(True, False),
                          default=True,
                          help='if "true", will call libpostal on all addresses to get the house number')


@namespace.route('/batchGeocode', methods=['POST'])
class BatchGeocode(Resource):
    """Batch geocoding"""

    @namespace.expect(batch_parser)
    @namespace.response(400, 'Error in arguments')
    @namespace.response(500, 'Internal Server error')
    @namespace.response(200, 'Found some results for at least one address')
    @namespace.response(204, 'No result at all')

    def post(self):
        """
Geocode (postal address cleansing and conversion into geographical coordinates) all addresses in csv like file.

Returns
-------
A json dictionary of the shape {'match': [<list of dictionaries>]} containing geocoded addresses. Depending of parameter "mode", following fields could be found:
In 'long' mode, each record will contain the following blocs:

- input : all columns present in input data, but at least "addrKey", "streetName", "houseNumber", "postCode", "city", "country"
- output: consolidated result of geocoding :
    - streetName: first non null value in ["road", "pedestrian","footway", "cycleway", "path", "address27", "construction", "hamlet", "park"]
    - houseNumber: house_number
    - postCode: postcode
    - city: first non null value in ["town", "village", "city_district", "county", "city"],
    - country: country
    - other: concatenate all values which were not picked by one of the above item
    - inHouseNumber: equivalent to input->houseNumber
    - lpostHouseNumber: "housenumber" provided by libpostal receiving concatenation of street and house number (from input)
    - lpostUnit: "unit"  provided by libpostal receiving concatenation of street and house number (from input)
- work: some metadata describing geocoding process:
    - transformedAddress: what address (after possibly some sequence of transformations) is actually sent to Nominatim
    - method: which transformation methods were used before sending the address to Nominatim. If the address was found without any transformation, will be "orig" (or "fast")
    - osmOrder: what was the rank of this result in Nominatim result (more useful in 'rejected' part)
    - retryOn26: If placeRank in match record is below 30 and housenumber (in input) contains other characters than digits, we retry to call Nominatim by only considering the first digits of housenumber: "30A","30.3", "30 bt 2", "30-32" become "30". If it gives a result with place_rank = 30, we keep it (in this case, a "cleansedHouse" appears in the output, with "30" in this example), and this field is set to "True"
- nominatim: selection of fields received from Nominatim:
    - lat
    - lon
    - placeRank
    - displayName
    - all fields in the "address" bloc
- check:  Check results indicators (if checkResult='true'):
    - SIMStreetWhich
    - SIMStreet
    - SIMCity
    - SIMZip
    - SIMHouseNumber

In 'geo' mode: only 'lat', 'lon', and 'placeRank' values from 'nominatim', 'addrKey' from 'input', and 'method' from 'work'

In 'short' mode: idem as 'geo', plus full 'output' bloc


If "withRejected=true", an additional field 'rejected' with all rejected records is added, with the same field selection as above, according to "mode", plus one additional fields, 'rejectReason'. Equal to:
- 'mismatch' if 'checkResult=true', and this result is "too far away" from the original value
- 'tail' if it was just not the first record.


        """
        log("batch")

        mode = get_arg("mode", "short")
        if not mode in ["geo", "short", "long"]:
            return [{"error": f"Invalid mode {mode}"}], 400

        error_msg = "Invalid value for '%s'. Possible values are 'true' or 'false' (received '%s')"
        with_rejected = get_arg("withRejected", False)
        if not with_rejected in [True, False]:
            return [{"error": error_msg%('withRejected', with_rejected)}], 400

        check_results = get_arg("checkResult", False)
        if not check_results in [True, False]:
            return [{"error": error_msg%('checkResult', check_results)}], 400

        osm_structured = get_arg("structOsm", False)
        if not osm_structured in [True, False]:
            return [{"error": error_msg%('structOsm', osm_structured)}], 400

        with_extra_house_number =  get_arg("extraHouseNumber", True)
        if not with_extra_house_number in [True, False]:
            return [{"error": error_msg%('extraHouseNumber', with_extra_house_number)}], 400

        if len(list(request.files.keys()))==0:
            return [{"error": "No file data was provided"}], 400

        key_name = (list(request.files.keys())[0])

        #log(request.files[0])
        #log(key_name)
        try:
            df = pd.read_csv(request.files[key_name], dtype=str)

        except UnicodeDecodeError as ude:
            log("Could not parse file, try to decompress...")
            log(ude)

            gzip_f = request.files[key_name]
            gzip_f.seek(0)
            g_f = gzip.GzipFile(fileobj=gzip_f,  mode='rb')

            df = pd.read_csv(g_f, dtype=str)

        old_col_idx = df.columns.to_frame()
        old_col_idx.insert(0, 'L0', ["input"]*df.columns.shape[0])
        df.columns= pd.MultiIndex.from_frame(old_col_idx, names=["L0", "L1"])

        log("Input: \n" + df.to_string(max_rows=10))
        mandatory_fields = [street_field, housenbr_field , postcode_field , city_field, addr_key_field, country_field]
        for field in mandatory_fields:
            if field not in df:
                return [{"error": f"Field '{field[1]}' mandatory in file. All mandatory fields are {';'.join([f[1] for f in mandatory_fields])}"}], 400

        if df[df[addr_key_field].duplicated()].shape[0]>0:
            return [{"error": f"Field '{addr_key_field[1]}' cannot contain duplicated values!"}], 400

        res, rejected_addresses = process_addresses(df,
                                                    check_results=check_results,
                                                    osm_structured=osm_structured,
                                                    with_extra_house_number= with_extra_house_number and mode != "geo",
                                                    transformers_sequence=transformers_sequence,
                                                    with_rejected=with_rejected)


        if isinstance(res, dict) :
            return {0:res}, 400


        if res is None or res.shape[0] == 0:
            return [], 204


        res = res.reset_index(drop=True)

        log("res:")
        log(res)
        for field, f_type in [("place_id", int), ("place_rank", int), ("lat", float), ("lon", float)]:
            res[("nominatim", field)] = res[("nominatim", field)].astype(f_type)
            if ("nominatim", field) in rejected_addresses:
                rejected_addresses[("nominatim", field)] = rejected_addresses[("nominatim", field)].astype(f_type)
        try:
            if mode == "geo":
                fields= [addr_key_field, ("nominatim", "lat"), ("nominatim", "lon"), ("nominatim", "place_rank"), ("work", "method"), ("work", "reject_reason")]
                res = res[[ f for f in fields if f in res ] ]
                rejected_addresses = rejected_addresses[ [ f for f in fields if f in rejected_addresses ] ]
            elif mode == "short":
                fields=[addr_key_field,
                           ("nominatim", "lat"), ("nominatim", "lon"), ("nominatim", "place_rank"), ("nominatim", "place_id"), ("work", "method"),("work", "dist_to_match"), ("work", "reject_reason")
                            ] + [("output", f) for f in res[("output")].columns]

                res = df.merge(res)
                res = res[[ f for f in fields if f in res ]]
                

                rejected_addresses = rejected_addresses[[ f for f in fields if f in rejected_addresses ]]
            elif mode == "long":
                pass


        except KeyError as ex:
            log(f"Error during column selection: {ex}")
            traceback.print_exc(file=sys.stdout)

        log("Output: \n"+res.iloc[:, 0:9].to_string(max_rows=9))

        res = multiindex_to_dict(res)

        res= {'match': res}
        if with_rejected:
            rejected_addresses = multiindex_to_dict(rejected_addresses)

            res["rejected"] = rejected_addresses

        res = to_camel_case(res)

        return res, 200



@namespace.route('/health', methods=['GET'])
class Health(Resource):
    """ Check service status """
    @namespace.response(500, 'Internal Server error')
    @namespace.response(503, 'Service is "DOWN"')
    @namespace.response(200, 'Service is "UP" or "DEGRADED"')

    def get(self):
        """Health status

        Returns
        -------
        - {'status': 'DOWN'}: Nominatim server does not answer (or gives an unexpected answer)
        - {'status': 'DEGRADED'}: Either Libpostal or Photon is down (or gives an unexpected answer). Geocoding is still possible as long as it does not requires one of those transformers
        - {'status': 'UP'}: Service works correctly

        """
        # Checking Nominatim

        osm_res = check_osm()

        if osm_res is False:
            log("Nominatim not up & running")
            log(f"Nominatim host: {osm_host}")

            return {"status": "DOWN",
                    "details": {"errorMessage": "Nominatim server does not answer",
                                "details": "Nominatim server does not answer"}}, 503
        if osm_res is not True:
            return {"status": "DOWN",
                    "details": {"errorMessage": "Nominatim server answers, but gives an unexpected answer",
                                "details": f"Nominatim answer: {osm_res}"}}, 503



        # Checking Libpostal

        lpost_res = check_libpostal()
        if lpost_res is False:
            log("Libpostal not up & running ")

            return {"status": "DEGRADED",
                    "details": {"errorMessage": "Libpostal server does not answer",
                                "details": "Libpostal server does not answer"}}, 200

        if lpost_res is not True:
            return {"status": "DEGRADED",
                    "details": {"errorMessage": "Libpostal server answers, but gives an unexpected answer",
                                    "details": f"Libpostal answer: {lpost_res}"}}, 200

        # Checking Photon

        photon_res= check_photon()
        if photon_res is False:
            return {"status": "DEGRADED",
                "details": {"errorMessage": "Photon server does not answer",
                            "details":  "Photon server does not answer"}}, 200
        if photon_res is not True:
            return {"status": "DEGRADED",
                    "details": {"errorMessage": "Photon server answers, but gives an unexpected answer",
                            "details": f"Photon answer: {photon_res}"}}, 200

        return {"status": "UP"}, 200
