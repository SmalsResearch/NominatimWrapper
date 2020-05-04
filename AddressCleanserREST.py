#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask,  request,jsonify

import pandas as pd

import os

from  importlib import reload
import AddressCleanserUtils
reload(AddressCleanserUtils)
from AddressCleanserUtils import *

from IPython.display import display

import sys, traceback

from datetime import datetime, timedelta


# In[ ]:


import config_REST
reload(config_REST)
from config_REST import *


# In[ ]:





# In[ ]:


# TO RUN : 
# jupyter nbconvert --to python AddressCleanserREST.ipynb
# export  FLASK_APP=AddressCleanserREST.py ; export  FLASK_ENV=development ;  flask   run  

# OR : 
# gunicorn -w 8 -b 127.0.0.1:5000 wsgi:app


# In[ ]:


# !jupyter nbconvert --to python AddressCleanserREST.ipynb


# In[1]:


# AddressCleanserUtils.pbar.unregister()

AddressCleanserUtils.with_dask         = False
AddressCleanserUtils.check_osm_results = True

AddressCleanserUtils.addr_key_field  = addr_key_field

AddressCleanserUtils.regex_replacements = regex_replacements

AddressCleanserUtils.use_osm_parent = use_osm_parent 


# In[2]:


"low".upper()


# In[6]:


import logging, os

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


def get_init_df(data):
    return pd.DataFrame([{addr_key_field : "1",
                          street_field:   data["street"],
                          housenbr_field: data["housenumber"],
                          postcode_field: data["postcode"],
                          city_field:     data["city"],
                          country_field:  data["country"]
                          }])


# In[ ]:


def get_row_dict(row, orig=False):
    if orig: 
        return row["osm_item_result"]
    else: 
        to_copy_field = ["osm_id", "lat","lon","display_name", "place_rank", "method", "extra_house_nbr"] + list(collapse_params.keys())  + list(filter(lambda x: x.startswith("SIM"), row.index))
        res =  {}

        for f in to_copy_field:
            if f in row: 
                res[f] = row[f]

        return res
    


# In[ ]:


def format_res(res):
    return list(res.fillna("").apply(lambda row: get_row_dict(row, False), axis=1))


# In[1]:


transformers_sequence = [ ["orig"],
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

def process_address(data):
    vlog(f"Will process {data}")
    to_process_addresses = get_init_df(data)
    
    vlog("Got dataframe")
    all_reject = pd.DataFrame()
    for transformers in transformers_sequence:
        vlog ("--------------------------")
        vlog("| Transformers : " + ";".join(transformers))
        vlog ("--------------------------")

        try :
            osm_results, rejected, step_stats = transform_and_process(to_process_addresses, transformers, addr_key_field, 
                                                                      street_field=street_field, housenbr_field=housenbr_field, 
                                                                      postcode_field=postcode_field, city_field=city_field,
                                                                      country_field=country_field)
        except Exception as e: 
            log(f"Error during processing : {e}")
            traceback.print_exc(file=sys.stdout)
            return {"error": str(e)}
        
        all_reject = all_reject.append(rejected, sort=False)
        
        vlog(step_stats)
        if osm_results.shape[0] > 0:
            osm_results = add_extra_house_number(osm_results, to_process_addresses, street_field=street_field, housenbr_field=housenbr_field)
            
            return {"match": format_res(osm_results), "rejected": format_res(all_reject)}
    
    return {"rejected": format_res(all_reject)}


# In[ ]:


def get_arg(argname, def_val):
    if argname in request.form: 
        return request.form[argname]
    return request.args.get(argname, def_val)


import sys

app = Flask(__name__)

@app.route('/search/', methods=['GET', 'POST'])
def search():
#     print("search!")

    for k in AddressCleanserUtils.timestats:
        AddressCleanserUtils.timestats[k]=timedelta(0)
        
    data= {"street"      : get_arg("street", ""),
           "housenumber" : get_arg("housenumber", ""),
           "city"        : get_arg("city", ""),
           "postcode"    : get_arg("postcode", ""),
           "country"     : get_arg("country", "")
          }
    res = process_address(data)
    log(f"Input: {data}")
    log(f"Result: {res}")
    
    if with_timing_info: 
        res["timing"] = {k: AddressCleanserUtils.timestats[k].total_seconds() for k in AddressCleanserUtils.timestats}
    
    return jsonify(res)



