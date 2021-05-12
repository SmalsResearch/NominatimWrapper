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


# In[38]:


def call_ws(addr_data, check_result=True): #lg = "en,fr,nl"
    t = datetime.now()
    
    params = urllib.parse.urlencode({"street": addr_data[street_field],
                                     "housenumber": addr_data[housenbr_field],
                                     "city": addr_data[city_field],
                                     "postcode": addr_data[postcode_field],
                                     "country": addr_data[country_field],
                                     "check_result" : "yes" if check_result else "no"
                                    })
    url = f"http://{ws_hostname}:5000/search/?{params}"
    
    print(url)
    try:
        with urllib.request.urlopen(url) as response:
            res = response.read()
            res = json.loads(res)
#             print(res)
            res["time"] = datetime.now() - t
            return res
    except Exception as e:
        return str(e)
    


# In[28]:


def call_ws_batch(addr_data, mode="geo", with_reject=False, no_check=False): #lg = "en,fr,nl"
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
        "no_check": 1 if no_check else 0
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

# In[43]:


call_ws({street_field:   "Av. Fonsny", 
         housenbr_field: "20",
         city_field:     "Saint-Gilles",
         postcode_field: "1060",
         country_field:  "Belgium"}, check_result=False)


# ## Batch calls (row by row)

# In[10]:


addresses = get_addresses("address.csv.gz")
addresses = addresses.sample(100).copy()


# ### Simple way

# In[15]:


addresses["json"] = addresses.progress_apply(call_ws, axis=1)


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

# In[20]:


# Only geocoding
call_ws_batch(addresses, mode="geo")


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


# In[25]:


df_res


# In[ ]:




