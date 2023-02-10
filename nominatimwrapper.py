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

import os

import sys
import traceback

from datetime import datetime, timedelta
import time
import logging
import json

import gzip


from flask import Flask,  request, url_for
from flask_restplus import Api, Resource, reqparse

import werkzeug

import pandas as pd

logging.basicConfig(format='[%(asctime)s]  %(message)s', stream=sys.stdout) # does not work if I put if after the next import...


from base import (log, vlog, get_osm, get_photon)


from utils import (parse_address,
                                  addr_key_field, street_field, housenbr_field,
                                  postcode_field, city_field, country_field,
                                  process_address, process_addresses,
                                  update_timestats, timestats, to_camel_case,
                                  convert_street_components, remove_empty_values)


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
        phot=""
        phot = get_photon("Bruxelles")
        assert city_test_to[0] in phot["features"][0]["properties"]["name"] or \
               city_test_to[1] in phot["features"][0]["properties"]["name"]
        log("Photon working properly")
        break


    except Exception:
        log("Photon not up & running ")
        log(f"Try again in {delay} seconds")
        log(phot)
        time.sleep(delay)

        #raise e
if i == 9:
    log("Photon not up & running ! ")
    log("Start it with 'nohup java -jar photon-*.jar &'")
    log(f"Photon host: {photon_host}")


log("Waiting for requests...")



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

def get_yesno_arg(argname, def_val):
    """
    Convert boolean argument from 'yes' and 'no' to True and False.
    If any other value, return None

    Parameters
    ----------
    argname : str
        Argument name.
    def_val : str
        Default value.

    Returns
    -------
    boolean
        True, False or None.

    """

    arg = get_arg(argname, def_val)
    if arg.lower() == "yes":
        return True
    if arg.lower() == "no":
        return False
    return None





app = Flask(__name__)
api = Api(app,
          version='0.1',
          title='NominatimWrapper API',
          description="""A geocoder built upon Nominatim.

          Source available on: https://github.com/SmalsResearch/NominatimWrapper/

          """,
          doc='/doc',
          prefix='/REST/nominatimWrapper/v0.1'
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
single_parser.add_argument('fullAddress',     type=str, help='Full address in a single field')

single_parser.add_argument('withRejected',
                           type=str,
                           choices=('yes', 'no'),
                           default='no',
                           help='If "yes", rejected results are returned')
single_parser.add_argument('checkResult',
                           type=str,
                           choices=('yes', 'no'),
                           default='no',
                           help='If "yes", will "double check" OSM results')
single_parser.add_argument('structOsm',
                           type=str,
                           choices=('yes', 'no'),
                           default='no',
                           help='If "yes", will call the structured version of OSM')
single_parser.add_argument('extraHouseNbr',
                           type=str,
                           choices=('yes', 'no'),
                           default='yes',
                           help='If "yes", will call libpostal on all addresses to get the house number')

@namespace.route('/search')
class Search(Resource):
    """ Single address geocoding"""
    @namespace.expect(single_parser)
    @namespace.response(400, 'Error in arguments')
    @namespace.response(500, 'Internal Server error')
    @namespace.response(200, 'Found a match for this address (or some rejected addresses)')
    @namespace.response(204, 'No address found, even rejected')


    def post(self):
        """
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

        Returns a dictionary with 2 parts

        - match: a single result, with the following fields:
            - Results comming straight from Nominatim:
                - placeId
                - lat
                - lon
                - displayName
                - placeRank
             - Structured address (in addressOut):
                - streetName: first non null value in ["road", "pedestrian","footway", "cycleway", "path", "address27", "construction", "hamlet", "park"]
                - houseNumber: house_number
                - postCode: postcode
                - city: first non null value in ["town", "village", "city_district", "county", "city"],
                - country: country
                - other: concatenate all values which were not picked by one of the above item
             - Check results indicators (if checkResult='yes'):
                - SIMStreetWhich
                - SIMStreet
                - SIMCity
                - SIMZip
                - SIMHouseNbr
             - Other information:
                - method: which transformation methods were used before sending the address to Nominatim. If the address was found without any transformation, will be "orig" (or "fast" if FASTMODE is activated at server level)
                - osmAddrIn: what address (after possibly some sequence of transformation) is actually sent to Nominatim
                - inHouseNbr:  house number given in input
                - lpostHouseNbr: "housenumber" provided by libpostal receiving concatenation of street and house number (from input)
                - lpostUnit: "unit"  provided by libpostal receiving concatenation of street and house number (from input)
        - reject: list of rejected results, with most of the same fields, with additionnal fields:
             - rejectReason: 'mismatch" or "tail"
             - distToMatch: distance (in kilometer) to the result given in "match"


        """

        start_time = datetime.now()

        for k in timestats:
            timestats[k]=timedelta(0)

        address = get_arg("fullAddress",  "")
        data= {
               street_field   : get_arg(street_field,      ""),
               housenbr_field : get_arg(housenbr_field,    ""),
               city_field     : get_arg(city_field,        ""),
               postcode_field : get_arg(postcode_field,    ""),
               country_field  : get_arg(country_field,     ""),
              }

        used_fields = list(filter(lambda x: data[x]!="", data))


        error_msg = "Invalid value for '%s'. Possible values are 'yes' or 'no'"
        with_rejected = get_yesno_arg("withRejected", "no")
        if with_rejected is None:
            return [{"error": error_msg%('withRejected')}], 400

        check_results = get_yesno_arg("checkResult", "no")
        if check_results is None:
            return [{"error": error_msg%('checkResult')}], 400

        osm_structured = get_yesno_arg("structOsm", "no")
        if osm_structured is None:
            return [{"error": error_msg%('structOsm')}], 400

        with_extra_house_number =  get_yesno_arg("extraHouseNbr", "yes")
        if with_extra_house_number is None:
            return [{"error": error_msg%('extraHouseNbr')}], 400

        if address != "":
            if len(used_fields)>0:
                return [{"error": "Field 'fullAddress' cannot be used together with fields "+";".join(used_fields)}],  400
            if osm_structured :
                return [{"error": "Field 'fullAddress' cannot be used together with fields 'structOsm=yes'"}],   400
            if check_results :
                return [{"error": "Field 'fullAddress' cannot be used together with fields 'checkResult=yes'"}], 400

            data[street_field] = address


        log("Parameters:")
        log(f"with_rejected:   {with_rejected}")
        log(f"check_results:   {check_results}")
        log(f"osm_structured:  {osm_structured}")
        log(f"extra_house_nbr: {with_extra_house_number}")



        res = process_address(data,
                              check_results=check_results,
                              osm_structured=osm_structured,
                              with_extra_house_number= with_extra_house_number,
                              fastmode=fastmode,
                              transformers_sequence=transformers_sequence)

        log(f"Input: {data}")
        log(f"Result: {res}")

        update_timestats("global", start_time)

        if with_timing_info:
            res["timing"] = {k: v.total_seconds()*1000 for k, v in timestats.items()}

        if not with_rejected and "reject" in res:
            del res["reject"]

        if "error" in res:
            return res, 500

        return_code = 200 if ("match" in res and len(res["match"])>0) or ("reject" in res and len(res["reject"])>0) else 204



        for part in ['match', 'reject']:
            if part in res:
                for i in range(len(res[part])):
                    res[part][i] = convert_street_components(res[part][i])




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
                          choices=('yes', 'no'),
                          default='no',
                          help='if "yes", rejected results are returned',

batch_parser.add_argument('checkResult',
                          type=str,
                          choices=('yes', 'no'),
                          default='no',
                          help='if "yes", will "double check" OSM results')
batch_parser.add_argument('structOsm',
                          type=str,
                          choices=('yes', 'no'),
                          default='no',
                          help='if "yes", will call the structured version of OSM')
batch_parser.add_argument('extraHouseNbr',
                          type=str,
                          choices=('yes', 'no'),
                          default='yes',
                          help='if "yes", will call libpostal on all addresses to get the house number')


@namespace.route('/batch', methods=['POST'])
class Batch(Resource):
    """Batch geocoding"""

    @namespace.expect(batch_parser)
    @namespace.response(400, 'Error in arguments')
    @namespace.response(500, 'Internal Server error')
    @namespace.response(200, 'Found some results for at least one address')
    @namespace.response(204, 'No result at all')

    def post(self):
        """
        Geocode all addresses in csv like file.

        Returns
        -------
        A json (list of dictionaries) containing geocoded addresses. Depending of parameter 'mode', following fields could be found:

- In 'geo' mode:
    - Results coming straight from Nominatim:
        - lat
        - lon
        - placeRank
    - Other fields:
        - addrKey: from input
        - method : which transformation methods were used before sending the address to Nominatim. If the address was found without any transformation, will be "orig" (or "fast")
- In 'short' mode, additional fields:
    - Results comming straight from Nominatim:
        - placeId
    - Structured address (in addressOut):
         - streetName: first non null value in ["road", "pedestrian","footway", "cycleway", "path", "address27", "construction", "hamlet", "park"]
        - houseNumber: house_number
        - postCode: postcode
        - city: first non null value in ["town", "village", "city_district", "county", "city"],
        - country: country
        - other: concatenate all values which were not picked by one of the above item
    - Other fields:
        - inHouseNbr:  house number given in input
        - lpostHouseNbr: "housenumber" provided by libpostal receiving concatenation of street and house number (from input)
        - lpostUnit: "unit"  provided by libpostal receiving concatenation of street and house number (from input)
- In 'long' mode, additional fields:
    - All columns present in input will appear in output
    - displayName
    - Check results indicators (if checkResult='yes'):
        - SIMStreetWhich
        - SIMStreet
        - SIMCity
        - SIMZip
        - SIMHouseNbr
    - osmAddrIn: what address (after possibly some sequence of transformations) is actually sent to Nominatim
    - retryOn26: If placeRank in match record is below 30 and housenumber (in input) contains other characters than digits, we retry to call Nominatim by only considering the first digits of housenumber: "30A","30.3", "30 bt 2", "30-32" become "30". If it gives a result with place_rank = 30, we keep it (in this case, a "cleansedHouseNbr" appears in the output, with "30" in this example), and this field is set to "True"


If "withRejected=yes", an additional field with all rejected records is added, with the same field selection as above, according to "mode", plus one additional fields, 'reject_reason'. Equal to:
- 'mismatch' if 'checkResult=yes', and this result is "too far away" from the original value
- 'tail' if it was just not the first record.


        """
        log("batch")


        mode = get_arg("mode", "short")
        if not mode in ["geo", "short", "long"]:
            return [{"error": f"Invalid mode {mode}"}], 400

        error_msg = "Invalid value for '%s'. Possible values are 'yes' or 'no'"
        with_rejected = get_yesno_arg("withRejected", "no")
        if with_rejected is None:
            return [{"error": error_msg%('withRejected')}], 400

        check_results = get_yesno_arg("checkResult", "no")
        if check_results is None:
            return [{"error": error_msg%('checkResult')}], 400

        osm_structured = get_yesno_arg("structOsm", "no")
        if osm_structured is None:
            return [{"error": error_msg%('structOsm')}], 400

        with_extra_house_number =  get_yesno_arg("extraHouseNbr", "yes")
        if with_extra_house_number is None:
            return [{"error": error_msg%('extraHouseNbr')}], 400

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


        log("Input: \n" + df.to_string(max_rows=10))

        mandatory_fields = [street_field, housenbr_field , postcode_field , city_field, addr_key_field, country_field]
        for field in mandatory_fields:
            if field not in df:
                return [{"error": f"Field '{field}' mandatory in file. All mandatory fields are {';'.join(mandatory_fields)}"}], 400

        if df[df[addr_key_field].duplicated()].shape[0]>0:
            return [{"error": f"Field '{addr_key_field}' cannot contain duplicated values!"}], 400

        res, rejected_addresses = process_addresses(df,
                                                    check_results=check_results,
                                                    osm_structured=osm_structured,
                                                    with_extra_house_number= with_extra_house_number and mode != "geo",
                                                    transformers_sequence=transformers_sequence)


        if isinstance(res, dict) :
            return {0:res}, 400


        if res is None or res.shape[0] == 0:
            return [], 204

        res["place_id"] = res["place_id"].astype(int)
        res["place_rank"] = res["place_rank"].astype(int)
        rejected_addresses["place_id"] = rejected_addresses["place_id"].astype(int)
        rejected_addresses["place_rank"] = rejected_addresses["place_rank"].astype(int)



        try:
            res = df.merge(res, how="left")

            if mode == "geo":
                fields= [addr_key_field,"lat", "lon", "place_rank", "method"]
                res = res[fields]
                rejected_addresses = rejected_addresses[ [ f for f in fields if f in rejected_addresses ] ]
            elif mode == "short":
                fields=[addr_key_field,
                           "lat", "lon", "place_rank", "method", "place_id",
                           "addr_out_street", "addr_out_number", "in_house_nbr", "lpost_house_nbr", "lpost_unit", "addr_out_postcode", "addr_out_city", "addr_out_country", "addr_out_other" ]

                res = df.merge(res)[fields]
                rejected_addresses = rejected_addresses[[ f for f in fields if f in rejected_addresses ]]
            elif mode == "long":
                pass


            if with_rejected:
                rejected_rec = rejected_addresses.groupby(addr_key_field).apply(lambda rec: remove_empty_values( rec.to_dict(orient="records")) ).rename("reject").reset_index()
                res = res.merge(rejected_rec, how="outer")
                res["reject"] = res["reject"].apply(lambda rec: rec if isinstance(rec, list) else [])

            res["lat"] = res["lat"].astype(float)
            res["lon"] = res["lon"].astype(float)
        except KeyError as ex:
            log(f"Error during column selection: {ex}")
            traceback.print_exc(file=sys.stdout)

        log("Output: \n"+res.iloc[:, 0:9].to_string(max_rows=9))

        res = res.to_dict(orient="records")


        if mode != "geo":
            for i in range(len(res)):
                res[i] = convert_street_components(res[i])

                if "reject" in res[i]:
                    for j in range(len(res[i]["reject"])):
                        res[i]["reject"][j] = convert_street_components(res[i]["reject"][j])

        res = to_camel_case(res)

        #log(res)

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
        - {'status': 'DEGRADED'}: Either Libpostal or Photon is down (or gives an unexpected answer). Geocoding is still possible as long as it does not requires one of those transformer
        - {'status': 'UP'}: Service works correctly

        """
        # Checking Nominatim

        try:
            osm = get_osm(city_test_from)
            if not city_test_to[0] in osm[0]["namedetails"]["name:fr"]:

                return {"status": "DOWN",
                        "details": {"errorMessage": "Nominatim server answers, but gives an unexpected answer",
                                "details": f"Nominatim answer: {osm}"}}, 503

        except Exception as exc:

            log("Nominatim not up & running")
            log(f"Nominatim host: {osm_host}")

            return {"status": "DOWN",
                    "details": {"errorMessage": "Nominatim server does not answer",
                                "details": f"{exc}"}}, 503

        # Checking Libpostal

        try:
            lpost = parse_address(city_test_from)
            if len(lpost) < 1 or len(lpost[0]) < 1 or lpost[0][0].lower() != city_test_to[0].lower():
                return {"status": "DEGRADED",
                        "details": {"errorMessage": "Libpostal server answers, but gives an unexpected answer",
                                    "details": f"Libpostal answer: {lpost}"}}, 200

        except Exception as exc:
            log("Libpostal not up & running ")

            return {"status": "DEGRADED",
                    "details": {"errorMessage": "Libpostal server does not answer",
                                "details": f"{exc}"}}, 200

        # Checking Photon

        try:
            phot=""
            phot = get_photon(city_test_from)
            if not city_test_to[0] in phot["features"][0]["properties"]["name"] and \
               not city_test_to[1] in phot["features"][0]["properties"]["name"]:
                return {"status": "DEGRADED",
                        "details": {"errorMessage": "Photon server answers, but gives an unexpected answer",
                                "details": f"Photon answer: {phot}"}}, 200
        except Exception as exc:
            return {"status": "DEGRADED",
                "details": {"errorMessage": "Photon server does not answer",
                        "details": f"{exc}"}}, 200

        return {"status": "UP"}, 200
