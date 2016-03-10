#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

if __name__=="__main__":

    # -- initialize input parser
    parser = argparse.ArgumentParser()

    # -- set the min and max scan range
    parser.add_argument("-smin", type=float, default=-5000)
    parser.add_argument("-smax", type=float, default=5000)

    # -- set the velocity
    parser.add_argument("-svel", type=float, default=100)

    # -- set the fps
    parser.add_argument("-fps", type=float, default=30)

    # -- parse arguments
    args = parser.parse_args()

    # -- calculate scan time and number of columns
    stime = int((args.smax-args.smin)/args.svel)
    ncol  = int(args.fps*stime)

    # -- print to screen
    print("")
    print("smin  : {0:8}".format(args.smin))
    print("smax  : {0:8}".format(args.smax))
    print("svel  : {0:8} s^-1".format(args.svel))
    print("fps   : {0:8}".format(args.fps))
    print("stime : {0:8} s".format(stime))
    print("ncol  : {0:8} pix".format(ncol))
    print("")
