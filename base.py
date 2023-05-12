# -*- coding: utf-8 -*-
"""
Basic functions

@author: Vandy Berten (vandy.berten@smals.be)
"""

import logging
import urllib
import json


from datetime import datetime, timedelta


import requests


import pandas as pd

from config import osm_host, photon_host, libpostal_host


#logging.basicConfig(format='[%(asctime)s]  %(message)s', stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def log(arg):
    """
    Print log message

    Parameters
    ----------
    arg : any type


    Returns
    -------
    None.

    """
    if isinstance(arg, (pd.core.frame.DataFrame, pd.core.frame.Series)):
        with pd.option_context("display.max_columns", None, 'display.width', 200):
            log("\n"+str(arg))

    else:
        logging.info(arg)


def vlog(arg):
    """
    Print (verbose) log message

    Parameters
    ----------
    arg : any type

    Returns
    -------
    None.

    """
    if isinstance(arg, (pd.core.frame.DataFrame, pd.core.frame.Series)):
        if logger.getEffectiveLevel() <= logging.DEBUG:
            with pd.option_context("display.max_columns", None, 'display.width', 200):
                log("\n"+str(arg))
    else:
        logging.debug(arg)




def get_osm(addr, accept_language = "", namedetails="1"):
    """
    Call OSM (Nominatim) webservice

    Parameters
    ----------
    addr : str
        address (in a single string) to geocode.
    accept_language : str, optional
        set to OSM webservice in the 'accept-language' parameter. The default is "".

    Returns
    -------
    Result of the call, as a list of dict. Only a selection of fields are kept
    (place_id, lat, lon, display_name, address, namedetails, place_rank,
     category, type)

    """
    params = urllib.parse.urlencode({"q": addr,
                                    "format":"jsonv2",
                                    "accept-language":accept_language,
                                    "addressdetails":"1",
                                    "namedetails" :  namedetails,
                                    "limit": "50"
                                    })

    url = f"http://{osm_host}/search.php?{params}"
    vlog(f"Call to OSM: {url}")
    try:
        with urllib.request.urlopen(url) as response:
            res = response.read()
            res = json.loads(res)
#             return res
            return [ {field: item[field] for field in ["place_id", "lat", "lon",
                                                       "display_name", "address",
                                                       "namedetails", "place_rank",
                                                       "category", "type"] if field in item} for item in res]
    except Exception as exc:
        raise Exception (f"Cannot get OSM results ({osm_host}): {exc}") from exc




def get_osm_struct(street, housenumber, postcode, city, country,
                   accept_language = "", namedetails="1"):
    """
    Call OSM (Nominatim) webservice, using the structured version, splitting
    appart street (including building number), city, postalcode and country

    Parameters
    ----------
    street : str
        Street. Sent (with housenumber) to "street" argument
    housenumber : str
        House number
    postcode : str
        postal code. Sent to "postalcode" argument
    city : str
        city name. Sent to "city" argument
    country : str
        country. Sent to "country" argument
    accept_language : str, optional
        Sent to "accept-language" argument. The default is "".

    Returns
    -------
    None.

    """
    params = urllib.parse.urlencode({"street": f"{street}, {housenumber}" \
                                                 if pd.notnull(street)
                                                     and len(str(street).strip())>0 \
                                                 else "" ,
                                     "city":city,
                                     "postalcode": postcode,
                                     "country": country,
                                    "format":"jsonv2",
                                    "accept-language":accept_language,
                                    "addressdetails": "1",
                                    "namedetails" : namedetails,
                                    "limit": "50"
                                    })

    url = f"http://{osm_host}/search.php?{params}"
    vlog(f"Call to OSM: {url}")
    try:
        with urllib.request.urlopen(url) as response:
            res = response.read()
            res = json.loads(res)
#             return res
            return [ {field: item[field] for field in ["place_id",
                                                       "lat", "lon",
                                                       "display_name",
                                                       "address",
                                                       "namedetails",
                                                       "place_rank",
                                                       "category",
                                                       "type"]if field in item} for item in res ]
    except Exception as exc:
        raise Exception (f"Cannot get OSM results ({osm_host}): {exc}") from exc




def get_osm_details(place_id):
    """
    Call OSM (Nominatim) "details" service, to get additional information
    about a specific address

    Parameters
    ----------
    place_id : int
        Nominatim place id.

    Returns
    -------
    dict
        Result of OSM (Nominatim) details call.

    """
    params = urllib.parse.urlencode({"place_id": place_id,
                                    "format":"json",
                                    })

    url = f"http://{osm_host}/details.php?{params}"
    try:
        with urllib.request.urlopen(url) as response:
            res = response.read()
            return json.loads(res)
    except Exception as exc:
        logger.warning("Problems with get_details for place_id {place_id} (url: %s)", url)
        print(exc)
        return {"category":"error", "names": []}
#         raise e


def get_photon(addr):
    """
    Call Photon web service

    Parameters
    ----------
    addr : Address
        Address to be sent to Photon.

    Raises
    ------
    Exception
        In case any problem occurs with call.

    Returns
    -------
    dict
        Photon result.

    """
    params = urllib.parse.urlencode({"q": addr})
    url = f"http://{photon_host}/api?{params}"

    try:
        with urllib.request.urlopen(url) as response:
            res = response.read()
            return json.loads(res)
    except Exception as exc:
        raise Exception (f"Cannot connect to Photon ({photon_host}):  {exc}") from exc


def parse_address(address):
    """
    Call Libpostal web service

    Parameters
    ----------
    address : str
        Address to be parsed by Libpostal.

    Raises
    ------
    Exception
        In case any problem occurs with call.

    Returns
    -------
    res : lst
        Libpostal result.

    """

    url = f"http://{libpostal_host}/parser"
    params = {"query": address}

    try:
        res = requests.post(url, json = params)
    except Exception as exc:
        raise Exception (f"Cannot connect to Libpostal ({libpostal_host}): {exc}") from exc

    res = json.loads(res.content.decode())

    return res


timestats = {}


# # Functions

# ## Global


def update_timestats(label, start_time):
    """
    Add delay since 'start_time' (datetime.now()-start_time) to timestats[label]

    Parameters
    ----------
    label : str
        DESCRIPTION.
    t : timestamp
        when activity to measure started.

    Returns
    -------
    None.

    """
    if not label in timestats:
        timestats[label] = timedelta(0)
    timestats[label] += datetime.now() - start_time
