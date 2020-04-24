FROM centos:7

ARG photon_data

RUN yum -y update

# Install Python & Packages
RUN yum -y install python3 python3-devel java git wget pbzip2 vim gcc

# TODO : remove vim

RUN pip3 install Flask gunicorn pandas numpy matplotlib requests dask dask[dataframe] toolz jellyfish tqdm fsspec IPython openpyxl==3.0.1 psutil bokeh
 


# Install libpostal
# https://github.com/openvenues/libpostal 
RUN yum -y install curl autoconf automake libtool pkgconfig make

RUN mkdir /libpostal_code
RUN mkdir /libpostal_datadir

WORKDIR /libpostal_code
RUN git clone https://github.com/openvenues/libpostal
WORKDIR libpostal

RUN ./bootstrap.sh 
RUN ./configure --datadir=/libpostal_datadir
RUN make -j4
RUN make install
RUN ldconfig /usr/local/lib


# Pypostal : https://github.com/openvenues/pypostal
RUN pip3 install postal


# Install Photon
# TODO : try photo 0.3.2

RUN mkdir /photon
WORKDIR /photon
RUN wget --progress=dot:mega https://github.com/komoot/photon/releases/download/0.3.1/photon-0.3.1.jar
COPY ${photon_data} photon.tar.gz
RUN tar xzf photon.tar.gz

# To get the full datafile : 
# RUN wget -O - http://download1.graphhopper.com/public/photon-db-latest.tar.bz2 | pbzip2 -cd | tar x

# Install code
#

RUN mkdir /nominatim_wrapper
WORKDIR /nominatim_wrapper
COPY *.py ./

ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8     

WORKDIR /

COPY Docker/start.sh /start.sh
CMD ["./start.sh"]

EXPOSE 2322 8080 5000
