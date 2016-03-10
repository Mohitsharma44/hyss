#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

if __name__=="__main__":

    smin  = -5000
    smax  = 5000
    svel  = 100.
    stime = (smax-smin)/svel
    fps   = 30.
    ncol  = fps*stime

    print("ncol = {0}".format(ncol))
