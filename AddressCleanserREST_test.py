#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import urllib

import numpy as np

import json

from tqdm.autonotebook import  tqdm

#%matplotlib inline

tqdm.pandas(tqdm)

import dask.dataframe as dd

from dask.multiprocessing import get
from dask.diagnostics import ProgressBar

from datetime import datetime
import matplotlib.pyplot as plt

from IPython.display import display


# In[10]:


from config_batch import * 


# In[7]:


def call_ws(addr_data): #lg = "en,fr,nl"
    t = datetime.now()
    
    params = urllib.parse.urlencode({"street": addr_data[street_field],
                                     "housenumber": addr_data[housenbr_field],
                                     "city": addr_data[city_field],
                                     "postcode": addr_data[postcode_field],
                                     "country": addr_data[country_field],
                                    })
    url = "http://172.26.0.1:5000/search/?%s"%params
    
    
    try:
        with urllib.request.urlopen(url) as response:
            res = response.read()
            res = json.loads(res)
#             print(res)
            res["time"] = datetime.now() - t
            return res
    except Exception as e:
        return str(e)
    


# In[8]:


def get_plots(addresses, column):
    fig, axes = plt.subplots(ncols=1, nrows = addresses[column].nunique(),  sharex=True, figsize=[6, 3*addresses[column].nunique()])

    addresses.hist(column="_time", by=column,ax=axes)

    return fig


# In[82]:


addresses = get_addresses("address.csv.gz")


# In[83]:


display(addresses)


# In[92]:


addresses  =addresses.sample(50)


# In[94]:


with_dask = False
if with_dask : 
    dd_addresses = dd.from_pandas(addresses, npartitions=4)

    dask_task = dd_addresses.apply(call_ws, meta=('x', 'str'), axis=1)

    with ProgressBar(): 
        addresses["json"] = dask_task.compute()
else: 
    
    addresses["json"] = addresses.progress_apply(call_ws, axis=1)


# In[86]:


# Flask : 1:03
# gunicorn -w 1: 0:55
# gunicorn -w 2, npartition=2: 0:30
# gunicorn -w 2, npartition=4: 0:30
# gunicorn -w 4, npartition=4: 0:33

addresses


# In[87]:


addresses["status"]= addresses.json.apply(lambda d: "error" if "error" in d else "match" if "match" in d else "rejected")
addresses["time"]  = addresses.json.apply(lambda d: d["time"])

addresses["timing"]  = addresses.json.apply(lambda d: d["timing"] if "timing" in d else {})

addresses["method"]= addresses.json.apply(lambda d: d["match"][0]["method"] if len(d)>0 and "match" in d else "none")
addresses["street"]= addresses.json.apply(lambda d: d["match"][0]["addr_out_street"] if len(d)>0 and "match" in d else "")


display(addresses.drop("json", axis=1))


# In[88]:


addresses["timing"].apply(pd.Series)


# In[89]:


display(addresses.status.value_counts())


# In[90]:


addresses[addresses.status == "error"]


# In[80]:


# addresses[addresses.status == "error"].progress_apply(call_ws, axis=1)


# In[91]:


display(addresses.method.value_counts())


# In[63]:


addresses["_time"] = addresses.time.apply(lambda t: t.total_seconds())


# In[30]:


print("Method : mean")


# In[31]:


display(addresses.groupby("method")._time.mean())


# In[32]:


print("Method : std")


# In[33]:


display(addresses.groupby("method")._time.std())


# In[ ]:





# In[34]:


print("Status : mean")


# In[35]:


display(addresses.groupby("status")._time.mean())


# In[36]:


print("Status : std")


# In[37]:


display(addresses.groupby("status")._time.std())


# In[38]:


fig = get_plots(addresses, "status")


# In[39]:


fig.savefig("time_per_status.png")


# In[40]:


fig = get_plots(addresses, "method")


# In[41]:


fig.savefig("time_per_method.png")


# In[ ]:




