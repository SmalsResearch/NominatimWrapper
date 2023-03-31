#!/usr/bin/env python
# coding: utf-8

from flask import Flask,  request,jsonify

from postal.parser import parse_address


def get_arg(argname, def_val):
    """
    Get argument 'argname' from request, with defaulf value 'def_val'

    Parameters
    ----------
    argname : str
        argument namge.
    def_val : str
        default value.

    Returns
    -------
    str
        request argument.

    """
    if request.json and argname in request.json:
        return request.json[argname]

    return request.values.get(argname, def_val)


app = Flask(__name__)

@app.route('/parser', methods=['GET', 'POST'])
def parser():
    """

    Call libpostal service, with 'query' argument
    Returns
    -------
    dict
        libpostal result.

    """
    query = get_arg("query", "")

    res = parse_address(str(query))

    return jsonify(res)
