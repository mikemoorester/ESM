#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import numpy as np
import re
import gzip
import calendar

from scipy.stats.stats import nanmean, nanmedian, nanstd

import gpsTime as gt
import datetime as dt

import os, sys

def parseSVNAV(svnavFile) :
    """

    svnav = parseSVNAVDAT(svnavFile)

    """

    svnav = []

    with open(svnavFile) as f:
        #print("Opend the file",svnavFile)
        for line in f:
            if len(line) > 3 and line[2] == ',' :
                record = {}
                record['prn']   = int(line[0:2])
                record['sv']    = int(line[4:6])
                record['blk']   = int(line[8:9])
                record['mass']  = float(line[11:20])
                record['bias']  = str(line[24:25])
                record['yrate'] = float(line[30:37])
                yr    = int(line[37:41])
                mo    = int(line[42:45])
                dy    = int(line[45:48])
                hr    = int(line[49:51])
                mn    = int(line[52:54])
                record['date']  = dt.datetime(yr,mo,dy,hr,mn)  
                record['dX']    = float(line[55:62])
                record['dY']    = float(line[63:70])
                record['dZ']    = float(line[72:79])
                svnav.append(record)
    return svnav

def findSV_DTO(svnav,prn,dto) :
    sv = -1
    lstdto = dt.datetime(1994,01,01)

    for record in svnav:
        if record['prn'] == prn and record['date'] < dto and record['date'] > lstdto :
            sv = record['sv']
            lstdto = record['date']
       
    return sv
#===========================================================================
if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from matplotlib import cm 

    import argparse

    parser = argparse.ArgumentParser(prog='autclnParse',description='Read in the autcln post fit summary file')

    parser.add_argument("-f", "--filename", dest="filename", help="Autcln post fit summary file")
    args = parser.parse_args()
    #===================================
   
    if args.filename :
        svnav = parseSVNAV(args.filename)
        #print("SVNAV:",svnav)
        dto = dt.datetime(2012,06,01)
        for prn in range(1,33):
            sv = findSV_DTO(svnav,prn,dto)
            print(prn,sv)
