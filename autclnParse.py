#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import numpy as np
import re
import gzip
import calendar
import os, sys
import zipfile

from scipy.stats.stats import nanmean, nanmedian, nanstd

import gpsTime as gt
import datetime as dt

import esm
import svnav

def file_opener(filename):
    '''
    Decide what kind of file opener should be used to parse the data:
    # file signatures from: http://www.garykessler.net/library/file_sigs.html
    '''

    # A Dictionary of some file signatures,
    # Note the opener statements are not correct for bzip2 and zip
    openers = {
        "\x1f\x8b\x08": gzip.open,
        "\x42\x5a\x68": open,      # bz2 file signature
        "\x50\x4b\x03\x04": open   # zip file signature
    }

    max_len = max(len(x) for x in openers)
    with open(filename) as f:
        file_start = f.read(max_len)
        for signature, filetype in openers.items():
            if file_start.startswith(signature):
                return filetype
    return open

def file_len(fname):
    """
    file_len : work out how many lines are in a file
    
    Usage: file_len(fname)

    Input: fname - filename, can take a txt file or a gzipp'd file

    Output: i - number of lines in the file

    """
    file_open = file_opener(fname)
    with file_open(fname) as f:
        i=-1 # account for 0-length files
        for i, l in enumerate(f):
            pass
    return i + 1

def reject_outliers_arg(data,nSigma):
    """
    Do a simple outlier removal at 3 sigma, with two passes over the data
    """
    criterion = ( (data[:] < (data[:].mean() + data[:].std() * nSigma)) &
                  (data[:] > (data[:].mean() - data[:].std() * nSigma)) )
    ind = np.array(np.where(criterion))[0]

    return ind

def reject_outliers_byelevation_arg(data,nSigma,zenSpacing=0.5):
    zen = np.linspace(0,90,int(90./zenSpacing)+1)
    tmp = []
    #tmp = np.array(tmp)
    for z in zen:
        criterion = ( (data[:,2] < (z + zenSpacing/2.)) &
                      (data[:,2] > (z - zenSpacing/2.)) )
        ind = np.array(np.where(criterion))[0]
        rout = reject_outliers_arg(data[ind,3],nSigma)
        tmp.append(rout.tolist())
        #tmp = np.concatenate([tmp,rout])
    return tmp

def reject_outliers_elevation(data,nSigma,zenSpacing=0.5):
    zen = np.linspace(0,90,int(90./zenSpacing)+1)
    init = 0
    for z in zen:
        criterion = ( (data[:,2] < (z + zenSpacing/2.)) &
                      (data[:,2] > (z - zenSpacing/2.)) )
        ind = np.array(np.where(criterion))[0]

        tdata = np.zeros((np.size(ind),3))
        tdata = data[ind,:]

        criterion = ( (data[ind,3] < (data[ind,3].mean() + data[ind,3].std() * nSigma)) &
                  (data[ind,3] > (data[ind,3].mean() - data[ind,3].std() * nSigma)) )
        rout = np.array(np.where(criterion))[0]

        # if its the first interation initialise tmp
        if init == 0 and np.size(rout) > 0:
            tmp = tdata[rout,:]
            init = 1
        elif np.size(rout) > 0:
            tmp = np.vstack((tmp,tdata[rout,:]))

    return tmp

def parseAUTCLN(data):
    asterixRGX = re.compile('\*')

    # look for the satellite nadir residuals
    # this will be in increment of 0.2 degrees
    nadirRGX = re.compile('NAMEAN G')
    cfilesRGX = re.compile('INPUT CFILES:')
    siteEleRGX = re.compile('ATELV')
    siteEleHdrRGX = re.compile('ATELV Site')

    autcln = {}

    satNadir = np.zeros((32,70)) 
    autcln['satNadir'] = satNadir
    sites = {}
    autcln['sites'] = sites

    for line in data:
        if nadirRGX.search(line):
            prn = int(line[8:11])
            offset = 12
            resid = np.zeros(70)
            for i in range(0,70):
                start = offset + i*6
                end = offset + i*6 + 5
                resid[i] = float(line[start:end].strip())
            satNadir[prn-1,:] = resid
        elif cfilesRGX.search(line):
            year  = int(line[21:25])
            month = int(line[26:28])
            day   = int(line[29:31])
            autcln['date'] = dt.datetime(year,month,day)
        elif siteEleRGX.search(line):
            # skip the header
            if siteEleHdrRGX.search(line):
                #print("header:",line)
                continue
            # save off the A anb B terms to charcaterise the residuals:
            # RMS^2 = A^2 + B^2/(sin(elv))^2
            site = str(line[6:10])
            A    = float(line[12:16])
            B    = float(line[18:22])
            sites[site] = {}
            sites[site]['A'] = A
            sites[site]['B'] = B

    return autcln 

def parseAUTCLNSUM(autclnFile) :
    """
    autcln = parseDPH(dphFile)

    Reads in an autcln postfit summary file to store some statistics from the processing:
    1) Residuals of the satellites by Nadir angle
    2) Time stamp of data processed
    3) The A and B terms used to charcaterise the phase residuals at each station
    """

    asterixRGX = re.compile('\*')

    # look for the satellite nadir residuals
    # this will be in increment of 0.2 degrees
    nadirRGX = re.compile('NAMEAN G')
    cfilesRGX = re.compile('INPUT CFILES:')
    siteEleRGX = re.compile('ATELV')
    siteEleHdrRGX = re.compile('ATELV Site')

    autcln = {}
    satNadir = np.zeros((32,70)) 
    autcln['satNadir'] = satNadir
    autcln['network'] = autclnFile[-17] 
    print("LOOKING at network :",autcln['network'])
    sites = {}
    autcln['sites'] = sites

    # work out if the file is compressed or not,
    # and then get the correct file opener.
    file_open = file_opener(autclnFile)

    with file_open(autclnFile) as f:

        #print("Opend the file",autclnFile)
        for line in f:
            if nadirRGX.search(line):
                prn = int(line[8:11])
                offset = 12
                resid = np.zeros(70)
                for i in range(0,70):
                    start = offset + i*6
                    end = offset + i*6 + 5
                    resid[i] = float(line[start:end].strip())
                satNadir[prn-1,:] = resid
            elif cfilesRGX.search(line):
                year  = int(line[21:25])
                month = int(line[26:28])
                day   = int(line[29:31])
                autcln['date'] = dt.datetime(year,month,day)
            elif siteEleRGX.search(line):
                # skip the header
                if siteEleHdrRGX.search(line):
                    #print("header:",line)
                    continue
                # save off the A anb B terms to charcaterise the residuals:
                # RMS^2 = A^2 + B^2/(sin(elv))^2
                site = str(line[6:10])
                A    = float(line[12:16])
                B    = float(line[18:22])
                sites[site] = {}
                sites[site]['A'] = A
                sites[site]['B'] = B

    return autcln 

def consolidateNadir(svdat,autcln,outfile='SV_RESIDUALS.ND3') :
    '''
            timestamp az zen lc(mm) prn

    Input: 
        dphs a parsed dph structe obtained from resiudals.parseDPH(file)
        startDT a datetime object specify the start time of the first residual at epoch 1

    Output:
        filename if it ends in gz it will be automatically compressed
    '''
    lines = ''
    sep = ' '
    dto = autcln['date']
    with open(outfile,'a') as f:

        for i in range(0,32):
            prn = i+1
            sv = svnav.findSV_DTO(svdat,prn,dto)
            ctr = 0
            time = dto.strftime("%Y %j")
            res_line = ''

            for res in autcln['satNadir'][i]:
                if res < 99.9:
                    res_line = res_line + ' ' +  str(res)
                else :
                    res_line = res_line + ' nan '
            print(time,sv, res_line,file=f)

    return lines

def searchAUTCLN(args):
    """
    autclns = searchAUTCLN(args)

    search for autcln post ft summary files from a specified path

    return an array of autcln data structures

    """
    autclns = []
    autclnFile = 'autcln.post.sum'

    for root, dirs, files in os.walk(args.search):
        path = root.split('/')
        if autclnFile in files:
            print("Found a file in :",root,path[-1])
            aFile = root + '/' + autclnFile
            autcln = parseAUTCLNSUM(aFile)
            autclns.append(autcln)

    return autclns


#===========================================================================
if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from matplotlib import cm 

    import argparse

    #===================================
    parser = argparse.ArgumentParser(prog='autclnParse',description='Read in the autcln post fit summary file')

    parser.add_argument("-f", "--filename", dest="filename", help="Autcln post fit summary file")
    parser.add_argument("-e", "--elevation", dest="elevationPlot",action='store_true',default=False,
                        help="Plot Residuals vs Elevation Angle")
    parser.add_argument("--search",dest="search",help="Search for autcln post summary files from the provided path")
    parser.add_argument('--network',dest='network',default='yyyy_dddnN',choices=['yyyy_dddnN','ddd'],
                                        help="Format of gps subnetworks")
    parser.add_argument('--ns', dest="nadirDump", default=False, action='store_true', help='Stack the nadir residuals')
    parser.add_argument('--sv', dest="svnavFile", help="Location of GAMIT svnav.dat")
    parser.add_argument('--zip', dest="zip", help="search zip archive")
    args = parser.parse_args()
    #===================================
   
    if args.filename :
        autcln = parseAUTCLNSUM(args.filename)
        print("Autcln:",autcln)
    if args.search:
        autclns = searchAUTCLN(args)
        print("autclns",autclns)
        if args.nadirDump:
            print("Dumping the nadir residuals to a file")
            svdat = svnav.parseSVNAV(args.svnavFile)
            for autcln in autclns:
                consolidateNadir(svdat,autcln)
