#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import urllib

import numpy as np

import json

from tqdm.autonotebook import tqdm

#%matplotlib inline

tqdm.pandas()

import dask.dataframe as dd

from dask.multiprocessing import get
from dask.diagnostics import ProgressBar

from datetime import datetime
import matplotlib.pyplot as plt

from IPython.display import display


# In[2]:


import urllib3


# In[3]:


http = urllib3.PoolManager()


# In[4]:


from config_batch import * 


# # Functions

# In[5]:


ws_hostname = "127.0.1.1"
# ws_hostname = "192.168.1.3"


# In[6]:


def call_ws(addr_data, check_result=True, structured_osm=False): #lg = "en,fr,nl"
    t = datetime.now()
    
    params = urllib.parse.urlencode({"street": addr_data[street_field],
                                     "housenumber": addr_data[housenbr_field],
                                     "city": addr_data[city_field],
                                     "postcode": addr_data[postcode_field],
                                     "country": addr_data[country_field],
                                     "check_result" : "yes" if check_result else "no",
                                     "struct_osm" : "yes" if structured_osm else "no"
                                    })
    url = f"http://{ws_hostname}:5000/search/?{params}"
    
    try:
        with urllib.request.urlopen(url) as response:
            res = response.read()
            res = json.loads(res)
#             print(res)
            res["time"] = datetime.now() - t
            return res
    except Exception as e:
        return str(e)
    


# In[7]:


def call_ws_batch(addr_data, mode="geo", with_reject=False, check_result=True, structured_osm=False): #lg = "en,fr,nl"
#     print(addr_data)
#     print(addr_data.shape)
#     print()
    file_data = addr_data.rename(columns = {
        street_field : "street",
        housenbr_field: "housenumber",
        postcode_field: "postcode",
        city_field: "city",
        country_field: "country",
        addr_key_field : "addr_key"
    }).to_csv(index=False)
    
    r = http.request(
    'POST',
    f'http://{ws_hostname}:5000/batch',
    fields= { 
        'media': ('addresses.csv', file_data),
        'mode': mode,
        "with_rejected" : "yes" if with_reject else "no",
        "check_result" : "yes" if check_result else "no",
        "struct_osm" : "yes" if structured_osm else "no"
    })
    
    try:
        res = pd.DataFrame(json.loads(r.data.decode('utf-8')))
    except ValueError:
        print("Cannot decode result:")
        print(json.loads(r.data.decode('utf-8')))
        return 
#     display(res)
    return res


# In[8]:


def expand_json(addresses):
    addresses["status"]= addresses.json.apply(lambda d: "error" if "error" in d else "match" if "match" in d else "rejected")
    addresses["time"]  = addresses.json.apply(lambda d: d["time"])

    addresses["timing"]  = addresses.json.apply(lambda d: d["timing"] if "timing" in d else {})

    addresses["method"]= addresses.json.apply(lambda d: d["match"][0]["method"] if len(d)>0 and "match" in d else "none")
    
    for field in ["street", "number", "postcode", "city"]:
        addresses[field]= addresses.json.apply(lambda d: d["match"][0]["addr_out_"+field] if len(d)>0 and "match" in d else "")
    return 


# # Calls

# ## Single address calls

# In[11]:


call_ws({street_field:   "Av. Fonsny", 
         housenbr_field: "20",
         city_field:     "Saint-Gilles",
         postcode_field: "1060",
         country_field:  "Belgium"}, check_result=False, structured_osm=False)


# ## Batch calls (row by row)

# In[15]:


addresses = get_addresses("address.csv.gz")
addresses = addresses.sample(1000).copy()


# ### Simple way

# In[74]:


addresses["json"] = addresses.progress_apply(call_ws, check_result=True, structured_osm=False, axis=1)


# ### Using Dask

# In[17]:


dd_addresses = dd.from_pandas(addresses, npartitions=4)

dask_task = dd_addresses.apply(call_ws, meta=('x', 'str'), axis=1)

with ProgressBar(): 
    addresses["json"] = dask_task.compute()


# In[26]:


expand_json(addresses)


# In[27]:


addresses


# ## Batch calls (batch WS)

# ### Single block

# In[53]:


# Only geocoding
call_ws_batch(addresses, mode="geo", check_result=True)


# In[62]:


# Geocode + address
call_ws_batch(addresses, mode="short") 


# In[63]:


# Geocode + address, with rejected addresses
call_ws_batch(addresses, mode="long", with_reject=True) 


# ### Batch blocs

# In[22]:


chunk_size = 10
chunks = np.array_split(addresses, addresses.shape[0]//chunk_size)

res= [call_ws_batch(chunk, mode="long") for chunk in tqdm(chunks)]

## TODO : find a better way with dask? It seems that map_partitions does not support function returning dataframes. 
#50: 4:04
#100 : 2:30
#250 : 2:04
#1000 : 1:37


# In[23]:


df_res = pd.concat(res, sort=False)
df_res


# In[24]:


df_res.method.value_counts()


# In[75]:


# df_res


# ## Comparing options

# In[24]:


def call_ws_batch_chunks(addr_data, mode="geo", with_reject=False, check_result=True, structured_osm=False, chunk_size=50): 
    chunks = np.array_split(addr_data, addr_data.shape[0]//chunk_size)

    res= [call_ws_batch(chunk, mode=mode, 
                        check_result=check_result, 
                        structured_osm=structured_osm) for chunk in tqdm(chunks)]
    df_res = pd.concat(res, sort=False)
    return df_res


# In[13]:


results = {}


# In[25]:


results[("check", "struct")] = call_ws_batch_chunks(addresses, mode="short", check_result=True, structured_osm=True)


# In[26]:


results[("check", "unstruct")] = call_ws_batch_chunks(addresses, mode="short", check_result=True, structured_osm=False)


# In[27]:


results[("nocheck", "struct")] = call_ws_batch_chunks(addresses, mode="short", check_result=False, structured_osm=True)


# In[29]:


results[("nocheck", "unstruct")] = call_ws_batch_chunks(addresses, mode="short", check_result=False, structured_osm=False)


# In[30]:


for i in ["check","nocheck"]:
    for j in ["struct", "unstruct"]:
        print(i, j, results[(i,j)].shape)


# In[44]:


mg = results[("nocheck", "unstruct")].merge(results[("check", "unstruct")], how="outer", indicator=True, 
                                            on=["addr_key", "lat", "lon", "place_rank", 
                                                "addr_out_street", "addr_out_number", "extra_house_nbr", 
                                                "addr_out_postcode", "addr_out_city", "addr_out_country"])
mg


# In[45]:


mg[mg.addr_key.duplicated(keep=False)].sort_values("addr_key").iloc[0:60]

