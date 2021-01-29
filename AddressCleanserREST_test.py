#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


import urllib3


# In[4]:


http = urllib3.PoolManager()


# In[5]:


from config_batch import * 


# # Functions

# In[6]:


ws_hostname = "127.0.1.1"
# ws_hostname = "192.168.1.3"


# In[7]:


def call_ws(addr_data): #lg = "en,fr,nl"
    t = datetime.now()
    
    params = urllib.parse.urlencode({"street": addr_data[street_field],
                                     "housenumber": addr_data[housenbr_field],
                                     "city": addr_data[city_field],
                                     "postcode": addr_data[postcode_field],
                                     "country": addr_data[country_field],
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
    


# In[30]:


def call_ws_batch(addr_data, mode="geo"): #lg = "en,fr,nl"
#     print(addr_data)
#     print(addr_data.shape)
#     print()
    file_data = addr_data.rename(columns = {
        street_field : "street",
        housenbr_field: "housenumber",
        postcode_field: "postcode",
        city_field: "city",
        country_field: "country",
        addr_key_field : "addr_key"}).to_csv(index=False)
    
    r = http.request(
    'POST',
    f'http://{ws_hostname}:5000/batch',
    fields= { 
        'media': ('addresses.csv', file_data),
        'mode': mode
    })
    
#     print(r.data.decode('utf-8'))
    res = pd.DataFrame(json.loads(r.data.decode('utf-8')))
#     display(res)
    return res


# In[9]:


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

# In[12]:


call_ws({street_field: "Av. Fonsny", 
          housenbr_field: "20",
          city_field: "Saint-Gilles",
          postcode_field:  "1060",
          country_field: "Belgium"})


# In[17]:





# ## Batch calls (row by row)

# In[31]:


addresses = get_addresses("address.csv.gz")
addresses = addresses.sample(10).copy()


# ### Simple way

# In[29]:


addresses["json"] = addresses.progress_apply(call_ws, axis=1)


# ### Using Dask

# In[17]:


dd_addresses = dd.from_pandas(addresses, npartitions=4)

dask_task = dd_addresses.apply(call_ws, meta=('x', 'str'), axis=1)

with ProgressBar(): 
    addresses["json"] = dask_task.compute()


# In[30]:


expand_json(addresses)


# ## Batch calls (batch WS)

# ### Single block

# In[15]:


# Only geocoding
call_ws_batch(addresses)


# In[27]:


# Geocode + address
res = call_ws_batch(addresses, mode="long") 
res


# In[29]:


res.method.value_counts()


# In[28]:


addresses[~addresses.EntityNumber.isin(res.addr_key)]


# ### Batch blocs

# In[29]:


chunk_size = 10
chunks = np.array_split(addresses, addresses.shape[0]//chunk_size)

res= [call_ws_batch(chunk, mode="long") for chunk in tqdm(chunks)]

## TODO : find a better way with dask? It seems that map_partitions does not support function returning dataframes. 
#50: 4:04
#100 : 2:30
#250 : 2:04
#1000 : 1:37


# In[30]:


df_res = pd.concat(res, sort=False)
df_res


# In[31]:


df_res.method.value_counts()


# In[ ]:





# In[ ]:




