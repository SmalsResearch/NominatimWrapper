#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask,  request,jsonify

from postal.parser import parse_address


# In[3]:


# TO RUN : 
# jupyter nbconvert --to python LibpostalREST.ipynb
# export  FLASK_APP=LibpostalREST.py ; export  FLASK_ENV=development ;  flask run --port 8080

# OR : 
# gunicorn -w 2 -b 127.0.0.1:8080 wsgi_libpostal:app


# In[ ]:


def get_arg(argname, def_val):
    if request.json and argname in request.json: 
        return request.json[argname]
    
    return request.values.get(argname, def_val)
    

app = Flask(__name__)

@app.route('/parser', methods=['GET', 'POST'])
def parser():
    query = get_arg("query", "")
           
    res = parse_address(str(query))
    
    return jsonify(res)

