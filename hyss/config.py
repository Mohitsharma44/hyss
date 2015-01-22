#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# -- hold default paths and load a config file if it exists
HYSS_ENVIRON = {'HYSS_DPATH' : '.',
                'HYSS_DNAME' : '.',
                'HYSS_HPATH' : '.',
                'HYSS_HNAME' : '.',
                'HYSS_WRITE' : '.',
                'NOAA_DPATH' : '.',
                'HYSS_fac'   : '1'
                }



def load_config(infile):
    """
    Load a configuration file.

    The configuration file must be of the format,

    # Comments line 1
    # Comments line 2
    HYSS_DPATH : /path/to/data/file
    HYSS_DNAME : name_of_data_file
    HYSS_HPATH : /path/to/header/file
    HYSS_HNAME : name_of_header_file
    HYSS_WRITE : /path/to/write/output
    NOAA_DPATH : /path/to/noaa/data
    HYSS_fac   : 4

    Paramters
    ---------
    infile : str
        The name of the configuration file to load.
    """

    # -- Update the config dictionary
    for line in open(infile,'r'):
        if line[0]=='#':
            continue
        elif ':' in line:
            recs = line.split(':')
            HYSS_ENVIRON[recs[0].replace(" ","")] = \
                recs[1].replace("\n","").lstrip().rstrip()

    return




def default_environ():
    """
    Check for existing configuration file and load it.
    """

    config_files = [i for i in os.listdir('.') if i.endswith('.hcfg')]

    if len(config_files)==1:
        print('CONFIG: found configuration file {0}.'.format(config_files[0]))
        print('CONFIG: initializing configuration, check hyss.HYSS_ENVIRON ' 
              'for details.')
    load_config(config_files[0])

    return HYSS_ENVIRON
