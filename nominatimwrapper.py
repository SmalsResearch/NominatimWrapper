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
# - according to 'mode', avoid to compute useless things



import os

import sys
import traceback

from datetime import datetime, timedelta
import time
import logging
import json


from flask import Flask,  request, url_for
from flask_restx import Api, Resource, reqparse, fields

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


from config import osm_host, libpostal_host, photon_host, default_transformers_sequence, city_test_from, city_test_to, similarity_threshold


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
    Get argument from request form. Sometimes get it from request.form (from payload),
    sometimes from request.args.get (from url args)

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

    if argname in request.form: # in payload
        return convert_bool(request.form[argname])

    return convert_bool(request.args.get(argname, def_val)) # in args


app = Flask(__name__)
api = Api(app,
          version='1.0.0',
          title='NominatimWrapper API',
          description="""A service that allows geocoding (postal address cleansing and conversion into geographical coordinates), based on Nominatim (OpenStreetMap).

          Source available on: https://github.com/SmalsResearch/NominatimWrapper/

          """,
          external_docs={"description": "ext docs", "url": "https://github.com/SmalsResearch/NominatimWrapper/"}, # does not work
          doc='/doc',
          prefix='/REST/nominatimWrapper/v1.0',
          contact='Vandy BERTEN',
          contact_email='vandy.berten@smals.be',
          contact_url='https://www.smalsresearch.be/author/berten/',
          #tags=['geocoding']
)

namespace = api.namespace(
    '',
    'Main namespace')

single_address = namespace.model("singleAddress",
                          {'addrKey': fields.String(example="1"),
                           'streetName':fields.String(example="Avenue Fonsny",
                                                     description="The name of a passage or way through from one location to another (cf. Fedvoc)."),
                           'houseNumber': fields.String(example="20",
                                                       description = "An official alphanumeric code assigned to building units, mooring places, stands or parcels (cf. Fedvoc)"),
                           'postCode': fields.String(example="1160",
                                                    description="The post code (a.k.a postal code, zip code etc.) (cf. Fedvoc)"),
                           'postName': fields.String(example="Bruxelles",
                                                    description="Name with which the geographical area that groups the addresses for postal purposes can be indicated, usually the city (cf. Fedvoc)."),
                           'countryName': fields.String(example="Belgium",
                                                    description="The country of the address, expressed in natural language, possibly with errors (cf. Fedvoc)"),
                          },
                         description ="Generic content for a single input address",
                         skip_none=True)

input_address   = namespace.model("inputAddress",   {"address": fields.Nested(single_address)}, skip_none=True)
input_addresses = namespace.model("inputAddresses", {"addresses": fields.List(fields.Nested(single_address))})

output_meta=namespace.model("outputMetadata", {
    "method":       fields.String(description="Which transformation methods were used before sending the address to Nominatim. If the address was found without any transformation, will be 'orig' (or 'fast') (mode:all)",
                                  example='libpostal+regex[lpost]'),
    "addrKey":      fields.String(description="Copied from input (mode:all)",
                                  example='1'),
    "placeRank":    fields.String(description="'placeRank' field from Nominatim. 26: street level, 30: building level (mode:all)",
                       example='30'),
    "placeId":      fields.String(description="Nominatim identifier (mode:short,full)",
                       example='182128'),
    "rejectReason": fields.String(description="'mismatch' (only if checkResult is set to 'true') or 'tail' (only in 'rejected' bloc) (mode:all)",
                                  example='tail'),
     "distToMatch": fields.Float(description="Distance (in kilometer) to the result given in 'match' (only in 'rejected' bloc) (mode:short,full)",
                                 example=0.1),

    "transformedAddress": fields.String(description="What address (after possibly some sequence of transformations) is actually sent to Nominatim (mode:full)",
                                       example="Avenue Fonsny, 20, 1060 Bruxelles"),
    "osmOrder":     fields.Integer(description="What was the rank of this result in Nominatim result (more useful in 'rejected' part) (mode:full)",
                                   example=0),
    "retryOn26":    fields.Boolean(description="If placeRank in match record is below 30 and housenumber (in input) contains other characters than digits, we retry to call Nominatim by only considering the first digits of housenumber: '30A','30.3', '30 bt 2', '30-32' become '30'. If it gives a result with place_rank = 30, we keep it (in this case, a 'cleansedHouseNumber' appears in the output, with '30' in this example), and this field is set to 'True') (mode:full)",
                                   example=True),
    "cleansedHouseNumber": fields.String(description="Cf retryOn26 (mode:full)",
                                         example="30")

}, skip_none=True)



output_output=namespace.model("outputOutput", {
    "streetName": fields.String(description= 'first non null value in ["road", "pedestrian","footway", "cycleway", "path", "address27", "construction", "hamlet", "park", "square"] from nominatim result. (mode:short,full)',
                                example="Avenue Fonsny - Fonsnylaan"),
    "houseNumber": fields.String(description= 'house_number from nominatim result (mode:short,full)',
                                 example='20'),
    "postCode":    fields.String(description= 'postcode from nominatim result (mode:short,full)',
                                 example='1060'),
    "postName":    fields.String(description= 'First non null value in ["town", "village", "city_district", "county", "city"] from nominatim result (mode:short,full)',
                                 example='Saint-Gilles - Sint-Gillis'),
    "countryName": fields.String(description= 'country from nominatim result. (mode:short,full)',
                                 example='België / Belgique / Belgien'),

    "displayName": fields.String(description= 'display_name nominatim result (mode:short,full)',
                                 example='20, Avenue Fonsny - Fonsnylaan, Saint-Gilles - Sint-Gillis, Brussel-Hoofdstad - Bruxelles-Capitale, Région de Bruxelles-Capitale - Brussels Hoofdstedelijk Gewest, 1060, België / Belgique / Belgien'),

    "other":       fields.String(description= 'Concatenate all values which were not picked by one of the above item (mode:short,full)',
                                 example=''),

    "lpostHouseNumber":  fields.String(description= '"housenumber" provided by libpostal receiving concatenation of street and house number (from input) (if extraHouseNumber = true ; mode:short,full)',
                                       example='20'),

    "lpostUnit":  fields.String(description= '"unit"  provided by libpostal receiving concatenation of street and house number (from input) (if extraHouseNumber = true ; mode:short,full)',
                                example='box 2'),



    "lat": fields.Float(description= 'Latitude, in EPSG:4326. Angular distance from some specified circle or plane of reference (mode:all)',
                       example=50.8358677),
    "lon": fields.Float(description= 'Longitude, in EPSG:4326. Angular distance measured on a great circle of reference from the intersection of the adopted zero meridian with this reference circle to the similar intersection of the meridian passing through the object (mode:all)',
                       example=4.3385087),


}, skip_none=True)



output_checks=namespace.model("outputChecks", {
    "simStreetWhich": fields.String(description= 'Which field in Nominatim was compared to input street name. Could be street_name (cf streetName output bloc), other (one element in other from output bloc), alt_names (name found in parent node), namedetails (usually a translation of the street name)',
                                    example="street_name"),
    "simStreet":      fields.Float(description= 'Comparison score between input street name and the best item in the field given by simStreetWhich. Score is the maximum between Levenshtein distance (after removing common street name words in french/dutch, such as avenue, steenweg...) and an inclusion test (check that a string s1 is equal to another string s2, except that s2 contains an additional substring)',
                                   example=0.8),
    "simCity":        fields.Float(description= 'Levenshtein distances between in/out city names',
                                   example=0.8),
    "simZip":         fields.Float(description= 'Similarity between in/out postcode: 1 if both equal, 0.5 if two first digits are equal, 0 otherwise',
                                   example=1.0),
    "simHouseNumber": fields.Float(description= 'Similarity between house numbers. 1 if perfect (non empty) match, 0.8 on range match (10 vs 8-10 or 10-12), 0.5 on number match (10A vs 10)',
                                   example=1.0),
}, skip_none=True)



output_address = namespace.model("cleansedAddress",
                            {
                               'metadata':   fields.Nested(output_meta,   skip_none=True, description= "Information describing the geocoding process"),
                               'output':     fields.Nested(output_output, skip_none=True, description= "Geocoding result (standardized address and geographical coordinates)"),
                               'check':      fields.Nested(output_checks, skip_none=True, mandatory=False, description=f"Similarity between input address and result (only if mode=long and checkResult=true). If checkResult=true, result are eliminated if (simZip = 0 and simCity < {similarity_threshold}) or (simStreet < {similarity_threshold})")
                            }, skip_none=True)

geocode_output = namespace.model("geocodeOutput",
                            {
                               'match':     fields.List(fields.Nested(output_address), description="A list with a single result"),
                               'rejected':    fields.List(fields.Nested(output_address), description="A list of rejected results (only if 'withRejected' is True)", mandatory=False)})

geocode_batch_output = namespace.model("geocodeBatchOutput",
                            {
                               'match':     fields.List(fields.Nested(output_address), description="A list with all results"),
                               'rejected':    fields.List(fields.Nested(output_address), description="A list of rejected results (only if 'withRejected' is True)", mandatory=False)})



with_https = os.getenv('HTTPS', "NO").upper().strip()

if with_https=="YES":
    # It runs behind a reverse proxy
    @property
    def specs_url(self):
        return url_for(self.endpoint('specs'), _external=True, _scheme='https')

    Api.specs_url = specs_url


single_parser = reqparse.RequestParser()

single_parser.add_argument('mode',
                          type=str,
                          choices=('geo', 'short', 'long'),
                          default='short',
                          help="""
Selection of columns in the ouput :

- geo: return mainly lat/long
- short: return lat/long, cleansed address (street, number, postcode, city, country)
- long: return all results from Nominatim""")


single_parser.add_argument('withRejected',
                           type=bool,
                           choices=(True, False),
                           default=False,
                           help='If "true", rejected results are returned')
single_parser.add_argument('checkResult',
                           type=bool,
                           choices=(True, False),
                           default=False,
                           help='If "true", will "double check" OSM results')
single_parser.add_argument('structOsm',
                           type=bool,
                           choices=(True, False),
                           default=False,
                           help='If "true", will call the structured version of OSM')
single_parser.add_argument('extraHouseNumber',
                           type=bool,
                           choices=(True, False),
                           default=True,
                           help='If "true", will call libpostal on all addresses to get the house number (cf lpostHouseNumber and lpostUnit fields in output bloc of output)')


@namespace.route('/geocode')
class Geocode(Resource):
    """ Single address geocoding"""

    @namespace.expect(single_parser, input_address)

    @namespace.response(400, 'Error in arguments')
    @namespace.response(500, 'Internal Server error')
    @namespace.response(204, 'No address found, even rejected')


    @namespace.marshal_with(geocode_output, description='Found a match for this address (or some rejected addresses)', skip_none=True)#, code=200)
    
    def post(self):
        """
Geocode (postal address cleansing and conversion into geographical coordinates) a single address.

Fields available in output will depends upon parameter "mode" (see 'mode:xx' in response model)
        """

        log("geocode")

        start_time = datetime.now()

        for k in timestats:
            timestats[k]=timedelta(0)

        try:
            if "address" not in namespace.payload:
                namespace.abort(400, "No data (address) was provided in payload")
        except Exception as e:
            log("exception")
            log(e)

            namespace.abort(400, "Cannot read request payload")

        data = namespace.payload["address"]

        used_fields = list(filter(lambda x: data[x]!="", data))

        vlog(f"used_fields: {used_fields}")

        mode = get_arg("mode", "short")
        if not mode in ["geo", "short", "long"]:
            namespace.abort(400, f"Invalid mode {mode}")

        error_msg = "Invalid value for '%s'. Possible values are 'true' or 'false' (received '%s')"
        with_rejected = get_arg("withRejected", False)
        if not with_rejected in [True, False]:
            namespace.abort(400, error_msg%('withRejected', with_rejected))

        check_results = get_arg("checkResult", False)
        if not check_results in [True, False]:
            namespace.abort(400, error_msg%('checkResult', check_results))

        osm_structured = get_arg("structOsm", False)
        if not osm_structured in [True, False]:
            namespace.abort(400, error_msg%('structOsm', osm_structured))

        with_extra_house_number =  get_arg("extraHouseNumber", True)
        if not with_extra_house_number in [True, False]:
            return namespace.abort(400, error_msg%('extraHouseNumber', with_extra_house_number))


        mandatory_fields = {street_field[1], housenbr_field[1], city_field[1], postcode_field[1], country_field[1]}
        if "fullAddress" in data:
            forbidden_fields = data.keys() & mandatory_fields

            if len(forbidden_fields) >0:
                namespace.abort(400, "Field 'fullAddress' cannot be used together with fields "+";".join(forbidden_fields))

            if osm_structured :
                namespace.abort(400, "Field 'fullAddress' cannot be used together with fields 'structOsm=true'")
            if check_results :
                namespace.abort(400, "Field 'fullAddress' cannot be used together with fields 'checkResult=true'")


            for f in mandatory_fields:
                data[f] = ""
            data[street_field[1]] = data["fullAddress"]
        else:

            missing_fields = mandatory_fields - data.keys()

            if len(missing_fields)>0:
                namespace.abort(400, f"Fields '{'; '.join(missing_fields)}' mandatory in 'address'.")




        _data = {}
        for f in data:
            _data[("input", f)] = data[f]
        data = _data

        if addr_key_field not in data:
            data[addr_key_field] = "-1"


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

        def filter_dict(dict_data, out_fields):
            return { f1: {f2:dict_data[f1][f2] for f2 in out_fields[f1] if f2 in dict_data[f1] } for f1 in out_fields if f1 in dict_data}
        try:

            if mode == "geo":
                out_fields={"output": ["lat", "lon", ], "metadata": ["method", "reject_reason", "place_rank", addr_key_field[1]]}

            elif mode == "short":
                out_fields={"metadata": ["method","reject_reason", "place_rank", "dist_to_match", addr_key_field[1]],
                        "output": res["match"][0]["output"]}

            elif mode == "long":
                out_fields = None

            if out_fields:
                res["match"] =    [filter_dict(res["match"][0], out_fields)]

            if not with_rejected and "rejected" in res:
                del res["rejected"]
            elif out_fields:
                res["rejected"] = [filter_dict(rec, out_fields) for rec in res["rejected"]]

            log(f"Result after selection: {res}")

        except KeyError as ex:
            log(f"Error during column selection: {ex}")


        update_timestats("global", start_time)

        if with_timing_info:
            res["timing"] = {k: v.total_seconds()*1000 for k, v in timestats.items()}


        if "error" in res:
            namespace.abort(500, res["error"])

        return_code = 200 if ("match" in res and len(res["match"])>0) or ("rejected" in res and len(res["rejected"])>0) else 204


        return to_camel_case(res), return_code




batch_parser = reqparse.RequestParser()

batch_parser.add_argument('mode',
                          type=str,
                          choices=('geo', 'short', 'long'),
                          default='short',
                          help="""
Selection of fields in the ouput :

- geo: only return lat/long
- short: return lat/long, cleansed address (street, number, zipcode, postname, country)
- long: return all results from Nominatim""")

batch_parser.add_argument('withRejected',
                          type=bool,
                          choices=(True, False),
                          default=False,
                          help='if "true", rejected results are returned')

batch_parser.add_argument('checkResult',
                          type=bool,
                          choices=(True, False),
                          default=False,
                          help='if "true", will "double check" OSM results')
batch_parser.add_argument('structOsm',
                          type=bool,
                          choices=(True, False),
                          default=False,
                          help='if "true", will call the structured version of OSM')
batch_parser.add_argument('extraHouseNumber',
                          type=bool,
                          choices=(True, False),
                          default=True,
                          help='if "true", will call libpostal on all addresses to get the house number')


@namespace.route('/batchGeocode', methods=['POST'])
class BatchGeocode(Resource):
    """Batch geocoding"""

    @namespace.expect(batch_parser, input_addresses)

    @namespace.response(400, 'Error in arguments')
    @namespace.response(500, 'Internal Server error')
    @namespace.response(204, 'No result at all')

    @namespace.marshal_with(geocode_batch_output, description='Found some results for at least one address', skip_none=True)#, code=200)

    def post(self):
        """
Geocode (postal address cleansing and conversion into geographical coordinates) all addresses in list.

Fields available in output will depends upon parameter "mode" (see 'mode:xx' in response model)


        """
        log("batch")

        mode = get_arg("mode", "short")
        if not mode in ["geo", "short", "long"]:
            namespace.abort(400, f"Invalid mode {mode}")

        error_msg = "Invalid value for '%s'. Possible values are 'true' or 'false' (received '%s')"
        with_rejected = get_arg("withRejected", False)
        if not with_rejected in [True, False]:
            namespace.abort(400, error_msg%('withRejected', with_rejected))

        check_results = get_arg("checkResult", False)
        if not check_results in [True, False]:
            namespace.abort(400, error_msg%('checkResult', check_results))

        osm_structured = get_arg("structOsm", False)
        if not osm_structured in [True, False]:
            namespace.abort(400, error_msg%('structOsm', osm_structured))

        with_extra_house_number =  get_arg("extraHouseNumber", True)
        if not with_extra_house_number in [True, False]:
            namespace.abort(400, error_msg%('extraHouseNumber', with_extra_house_number))



        try:
            if "addresses" not in namespace.payload:
                namespace.abort(400, "No data (address) was provided in payload")
        except Exception as e:
            log("exception")
            log(e)

            namespace.abort(400, "Cannot read request payload")

        data = namespace.payload["addresses"]

        if not isinstance(data, list):
            namespace.abort(400, f"Wrong format for 'addresses' in payload. Expecting a list of dicts (not a {type(data)})")

        if len(data)==0:
            namespace.abort(400, "List in 'addresses' in payload is empty")

        if not isinstance(data[0], dict):
            namespace.abort(400, "Wrong format for 'addresses' in payload. Expecting a list of dicts")

        try:
            df=pd.DataFrame(data)
        except ValueError as ve:
            namespace.abort(400, "Wrong format for 'addresses' in payload. Cannot build a dataframe from this data")

            log("Wrong format for 'addresses' in payload. Cannot build a dataframe from this data")
            log(data)
            log(ve)


        old_col_idx = df.columns.to_frame()
        old_col_idx.insert(0, 'L0', ["input"]*df.columns.shape[0])
        df.columns= pd.MultiIndex.from_frame(old_col_idx, names=["L0", "L1"])

        log("Input: \n" + df.to_string(max_rows=10))
        mandatory_fields = [street_field, housenbr_field , postcode_field , city_field, addr_key_field, country_field]
        for field in mandatory_fields:
            if field not in df:
                namespace.abort(400, f"Field '{field[1]}' mandatory in file. All mandatory fields are {';'.join([f[1] for f in mandatory_fields])}")

        if df[df[addr_key_field].duplicated()].shape[0]>0:
            namespace.abort(400, f"Field '{addr_key_field[1]}' cannot contain duplicated values!")

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
        # Field formatting
        for field, f_type in [(("metadata", "place_id"), int), (("metadata", "place_rank"), int), (("output", "lat"), float), (("output", "lon"), float)]:
            if field in res:
                res[field] = res[field].astype(f_type)
            if field in rejected_addresses:
                rejected_addresses[field] = rejected_addresses[field].astype(f_type)

        # Field selection
        try:
            if mode == "geo":
                out_fields= [("metadata", addr_key_field[1]), ("output", "lat"), ("output", "lon"), ("metadata", "place_rank"), ("metadata", "method"), ("metadata", "reject_reason")]
                res = res[[ f for f in out_fields if f in res ] ]
                rejected_addresses = rejected_addresses[ [ f for f in out_fields if f in rejected_addresses ] ]
            elif mode == "short":
                out_fields=[("metadata", addr_key_field[1]),
                           ("metadata", "place_rank"), ("metadata", "place_id"), ("metadata", "method"),("metadata", "dist_to_match"), ("metadata", "reject_reason")
                            ] + [("output", f) for f in res[("output")].columns]

                res = df.merge(res)
                res = res[[ f for f in out_fields if f in res ]]


                rejected_addresses = rejected_addresses[[ f for f in out_fields if f in rejected_addresses ]]
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
