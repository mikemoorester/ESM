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

    Read in a GAMIT undifferenced phase residual file.
    Return a DPH structure

    Will skip any lines in the file which contain a '*' 
    within any column 

    Checks there are no comments in the first column of the file
    Checks if the file is gzip'd or uncompressed

    """

    # look for the satellite nadir residuals
    # this will be in increment of 0.2 degrees
    nadirRGX = re.compile('NAMEAN G')

    svnav = {}
    #satNadir = np.zeros((32,70)) 
    #autcln['satNadir'] = satNadir

    with open(svnavFile) as f:
        print("Opend the file",svnavFile)
        for line in f:
            if len(line) > 3 and line[2] == ',' :
                prn   = int(line[0:2])
                sv    = int(line[4:6])
                blk   = int(line[8:9])
                mass  = float(line[11:20])
                bias  = str(line[24:25])
                yrate = float(line[30:37])
                yr    = int(line[37:41])
                mo    = int(line[42:45])
                dy    = int(line[45:48])
                hr    = int(line[49:51])
                mn    = int(line[52:54])
                dX    = float(line[55:62])
                dY    = float(line[63:70])
                dZ    = float(line[72:79])
                print(prn,sv,blk,mass,bias,yrate,yr,mo,dy,hr,mn,dX,dY,dZ)

    return svnav

#===========================================================================
if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from matplotlib import cm 

    #===================================
    # TODO Change this to argparse..
    #from optparse import OptionParser
    import argparse

    parser = argparse.ArgumentParser(prog='autclnParse',description='Read in the autcln post fit summary file')

    parser.add_argument("-f", "--filename", dest="filename", help="Autcln post fit summary file")
    args = parser.parse_args()
    #===================================
   
    if args.filename :
        svnav = parseSVNAV(args.filename)
        print("SVNAV:",svnav)

