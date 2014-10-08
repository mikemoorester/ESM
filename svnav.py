#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import numpy as np
import re
import gzip

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

def searchSNAV_timeSpan(svnav_data,sdto,edto):
    svs = []
    for record in svnav_data:
        if record['date'] < edto and record['date'] > sdto :
            svs.append(int(record['sv']))

    return np.unique(svs)

def findBLK_SV(svnav,sv):
    blk = -1
    sv = int(sv)
    for record in svnav:
        if record['sv'] == sv :
            blk = record['blk']
            #print("The SV",sv,"belongs to block:",blk)
    return blk 

def blockType(b):
    """
    BTString = blockType(blk)
    
    Given the block number return a string describing the block type of the satellite:

    BLK: 1=Blk I  2=Blk II    3=Blk IIA    4=Blk IIR-A   5=Blk IIR-B   6=Blk IIR-M   7=Blk IIF

    """
    blk = int(b)
    if(blk == 1):
        return "Block I"
    elif(blk == 2):
        return "Block II"
    elif(blk == 3):
        return "Block IIA"
    elif(blk == 4):
        return "Block IIR-A"
    elif(blk == 5):
        return "Block IIR-B"
    elif(blk == 6):
        return "Block IIR-M"
    elif(blk == 7):
        return "Block IIF"
    return

#===========================================================================
if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from matplotlib import cm 

    import argparse

    parser = argparse.ArgumentParser(prog='autclnParse',description='Read in the autcln post fit summary file')

    parser.add_argument("-f", "--filename", dest="filename", help="Autcln post fit summary file")

    parser.add_argument("--syyyy",dest="syyyy",type=int,default=1994,help="Start yyyy")
    parser.add_argument("--sdoy",dest="sdoy",type=int,default=0,help="Start doy")
    parser.add_argument("--eyyyy",dest="eyyyy",type=int,default=2014,help="End yyyyy")
    parser.add_argument("--edoy",dest="edoy",type=int,default=365,help="End doy")

    parser.add_argument("--search",dest="search",default=False,action='store_true',help="")

    args = parser.parse_args()
    #===================================
  
    if args.search and args.filename:
        svnav = parseSVNAV(args.filename)
        sdto = dt.datetime(int(args.syyyy),01,01) + dt.timedelta(days=int(args.sdoy)-1)
        edto = dt.datetime(int(args.eyyyy),01,01) + dt.timedelta(days=int(args.edoy)-1)
        svs = searchSNAV_timeSpan(svnav,sdto,edto)
        print("SVS:",svs)
    elif args.filename :
        svnav = parseSVNAV(args.filename)
        #print("SVNAV:",svnav)
        dto = dt.datetime(2012,06,01)
        for prn in range(1,33):
            sv = findSV_DTO(svnav,prn,dto)
            blk = findBLK_SV(svnav,sv)
            print(sv,blk)
            BT = blockType(blk)
            print(prn,sv,blk,BT)


