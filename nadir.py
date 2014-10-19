#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import re
import gzip
import calendar
import os, sys
import datetime as dt

from scipy import interpolate
from scipy.stats.stats import nanmean, nanmedian, nanstd
from scipy import sparse
from scipy import stats

#from multiprocessing import Process, Queue, current_process, freeze_support
import multiprocessing

import antenna as ant
import residuals as res
import gpsTime as gt
import GamitStationFile as gsf
import time
import svnav


def satelliteModel(antenna,nadirData):
    #assuming a 14 model at 1 deg intervals
    ctr = 0
    newNoAzi = []
    # from the Nadir model force the value at 13.8 to be equal to 14.0
    for val in antenna['noazi'] :
        if ctr == 13:
            antenna['noazi'][ctr] = (val + nadirData[ctr*5 -1])
        elif ctr > 13:
            antenna['noazi'][ctr] = val 
        else:
            antenna['noazi'][ctr] = val + nadirData[ctr*5]
        ctr +=1

    return antenna 

def calcNadirAngle(ele):
    """
        Calculate the NADIR angle based on the station's elevation angle

    """

    nadeg = np.arcsin(6378.0/26378.0 * np.cos(ele/180.*np.pi)) * 180./np.pi

    return nadeg

def pwl(site_residuals, svs, nadSpacing=0.1,):
    """
    PWL piece-wise-linear interpolation fit of phase residuals
    -construct a PWL fit for each azimuth bin, and then paste them all together to get 
     the full model
    -inversion is doen within each bin

    cdata -> compressed data
    """
    #print("rejecting any residuals greater than 100mm",np.shape(site_residuals))
    tdata = res.reject_absVal(site_residuals,100.)
    del site_residuals 
    #print("rejecting any residuals greater than 5 sigma",np.shape(tdata))
    data = res.reject_outliers_elevation(tdata,5,0.5)
    #print("finished outlier detection",np.shape(data))
    del tdata

    numd = np.shape(data)[0]
    # add one to make sure we have a linspace which includes 0.0 and 14.0
    # add another parameter for the zenith PCO estimate
    numNADS = int(14.0/nadSpacing) + 1 
    PCOEstimates = 1
    # 0 => 140 PCV, 141 PCO
    # 142 => 283 PCV, 284 PCO
    numSVS = np.size(svs)
    numParamsPerSat = numNADS + PCOEstimates
    numParams = numSVS * (numParamsPerSat)

    for i in range(0,numd):
        # work out the nadir angle
        nadir = calcNadirAngle(data[i,2])
        niz = np.floor(nadir/nadSpacing)
        # work out the svn number
        svndto =  gt.unix2dt(data[i,0])
        svn = svnav.findSV_DTO(svdat,data[i,4],svndto)

        svn_search = 'G{:03d}'.format(svn) 
        ctr = 0
        for sv in svs:
            if sv == svn_search:
                ind = ctr
                break
            ctr+=1

        iz = int(numParamsPerSat * ctr + niz)
        pco_iz = int(numParamsPerSat *ctr + numParamsPerSat -1)

        if iz >= numParams or pco_iz > numParams:
            print("prn,svn_search,svn,ctr,size(svs),niz,iz,nadir,numParams:",data[i,4],svn_search,svn,ctr,np.size(svs),niz,iz,nadir,numParams)
            print(svs)

        Apart_1 = (1.-(nadir-niz*nadSpacing)/nadSpacing)
        Apart_2 = (nadir-niz*nadSpacing)/nadSpacing
        # Now  add in the PCO offsest into the Neq
        Apart_3 = 1./np.sin(np.radians(nadir)) 

        w = 1. #np.sin(data[i,2]/180.*np.pi)
        
        Neq[iz,iz]         += (Apart_1*Apart_1) * 1./w**2
        Neq[iz,iz+1]       += (Apart_1*Apart_2) * 1./w**2
        Neq[iz,pco_iz]     += (Apart_1*Apart_3) * 1./w**2
        Neq[iz+1,iz]       += (Apart_2*Apart_1) * 1./w**2
        Neq[iz+1,iz+1]     += (Apart_2*Apart_2) * 1./w**2
        Neq[iz+1,pco_iz]   += (Apart_2*Apart_3) * 1./w**2
        Neq[pco_iz,iz]     += (Apart_3*Apart_1) * 1./w**2
        Neq[pco_iz,iz+1]   += (Apart_3*Apart_2) * 1./w**2
        Neq[pco_iz,pco_iz] += (Apart_3*Apart_3) * 1./w**2

        AtWb[iz]     += Apart_1 * data[i,3] * 1./w**2
        AtWb[iz+1]   += Apart_2 * data[i,3] * 1./w**2
        AtWb[pco_iz] += Apart_3 * data[i,3] * 1./w**2

    return Neq, AtWb

def pwlNadirSite(site_residuals, svs, params, nadSpacing=0.1,zenSpacing=0.5):
    """
    Create a model for the satellites and sites at the same time.
    PWL piece-wise-linear interpolation fit of phase residuals
    -construct a PWL fit for each azimuth bin, and then paste them all together to get 
     the full model
    -inversion is done within each bin

    cdata -> compressed data
    """

    # add one to make sure we have a linspace which includes 0.0 and 14.0
    # add another parameter for the zenith PCO estimate
    numNADS = int(14.0/nadSpacing) + 1 
    PCOEstimates = 1
    numSVS = np.size(svs)
    numParamsPerSat = numNADS + PCOEstimates
    numParamsPerSite = int(90.0/zenSpacing) + 1
    numParams = numSVS * (numParamsPerSat) + numParamsPerSite*params['numModels']

    print("\t Have:",numParams,"parameters to solve for",params['site'],params['numModels'])
    Neq = np.zeros((numParams,numParams))
    AtWl = np.zeros(numParams)
    change = params['changes']

    for m in range(0,int(params['numModels'])):
        print("\t\tCreating model",m+1,"of",params['numModels'])

        # start_yyyy and start_ddd should always be defind, however stop_dd may be absent
        #ie no changes have ocured since the last setup
        minVal_dt = gt.ydhms2dt(change['start_yyyy'][m],change['start_ddd'][m],0,0,0)

        if np.size(change['stop_ddd']) > m  :
            print("stop_ddd is in change")
            maxVal_dt = gt.ydhms2dt(change['stop_yyyy'][m],change['stop_ddd'][m],0,0,0)

            criterion = ( ( site_residuals[:,0] >= calendar.timegm(minVal_dt.utctimetuple()) ) &
                    ( site_residuals[:,0] < calendar.timegm(maxVal_dt.utctimetuple()) ) )
        else:
            criterion = ( site_residuals[:,0] >= calendar.timegm(minVal_dt.utctimetuple()) ) 

        mind = np.array(np.where(criterion))[0]
        print("MIND:",np.size(mind))
        #print("rejecting any residuals greater than 100mm",np.shape(site_residuals))
        tdata = res.reject_absVal(site_residuals[mind,0:],100.)
        print("rejecting any residuals greater than 5 sigma",np.shape(tdata))
        data = res.reject_outliers_elevation(tdata,5,0.5)
        print("finished outlier detection",np.shape(data))
        del tdata

        # Get the total number of observations for this site
        numd = np.shape(data)[0]
        print("Have:",numd,"observations")
        for i in range(0,numd):
            # work out the nadir angle
            nadir = calcNadirAngle(data[i,2])
            niz = int(np.floor(nadir/nadSpacing))

            nsiz = int(np.floor(data[i,2]/zenSpacing))
            siz = int( numParamsPerSat*numSVS +  m*numParamsPerSite + nsiz)

            # work out the svn number
            svndto =  gt.unix2dt(data[i,0])
            svn = svnav.findSV_DTO(svdat,data[i,4],svndto)
            svn_search = 'G{:03d}'.format(svn) 
            ctr = 0
            for sv in svs:
                if sv == svn_search:
                    ind = ctr
                    break
                ctr+=1

            w = 1.#np.sin(data[i,2]/180.*np.pi)
            iz = int(numParamsPerSat * ctr + m * numParamsPerSite +niz)
            pco_iz = int(numParamsPerSat *ctr + numParamsPerSat -1)

            # Nadir partials..
            Apart_1 = (1.-(nadir-niz*nadSpacing)/nadSpacing)
            Apart_2 = (nadir-niz*nadSpacing)/nadSpacing
            # PCO partial ...
            Apart_3 = 1./np.sin(np.radians(nadir)) 
            # Site partials
            Apart_4 = (1.-(data[i,2]-nsiz*zenSpacing)/zenSpacing)
            Apart_5 = (data[i,2]-nsiz*zenSpacing)/zenSpacing

            AtWb[iz]     = AtWb[iz]     + Apart_1 * data[i,3] * 1./w**2
            AtWb[iz+1]   = AtWb[iz+1]   + Apart_2 * data[i,3] * 1./w**2
            AtWb[pco_iz] = AtWb[pco_iz] + Apart_3 * data[i,3] * 1./w**2
            AtWb[siz]    = AtWb[siz]    + Apart_4 * data[i,3] * 1./w**2
            AtWb[siz+1]  = AtWb[siz+1]  + Apart_5 * data[i,3] * 1./w**2

            Neq[iz,iz]     = Neq[iz,iz]     + Apart_1 * Apart_1 * 1./w**2
            Neq[iz,iz+1]   = Neq[iz,iz+1]   + Apart_1 * Apart_2 * 1./w**2
            Neq[iz,pco_iz] = Neq[iz,pco_iz] + Apart_1 * Apart_3 * 1./w**2
            Neq[iz,siz]    = Neq[iz,siz]    + Apart_1 * Apart_4 * 1./w**2
            Neq[iz,siz+1]  = Neq[iz,siz+1]  + Apart_1 * Apart_5 * 1./w**2

            Neq[iz+1,iz]     = Neq[iz+1,iz]     + Apart_2 * Apart_1 * 1./w**2
            Neq[iz+1,iz+1]   = Neq[iz+1,iz+1]   + Apart_2 * Apart_2 * 1./w**2
            Neq[iz+1,pco_iz] = Neq[iz+1,pco_iz] + Apart_2 * Apart_3 * 1./w**2
            Neq[iz+1,siz]    = Neq[iz+1,siz]    + Apart_2 * Apart_4 * 1./w**2
            Neq[iz+1,siz+1]  = Neq[iz+1,siz+1]  + Apart_2 * Apart_5 * 1./w**2

            Neq[pco_iz,iz]     = Neq[pco_iz,iz]     + Apart_3 * Apart_1 * 1./w**2
            Neq[pco_iz,iz+1]   = Neq[pco_iz,iz+1]   + Apart_3 * Apart_2 * 1./w**2
            Neq[pco_iz,pco_iz] = Neq[pco_iz,pco_iz] + Apart_3 * Apart_3 * 1./w**2
            Neq[pco_iz,siz]    = Neq[pco_iz,siz]    + Apart_3 * Apart_4 * 1./w**2
            Neq[pco_iz,siz+1]  = Neq[pco_iz,siz+1]  + Apart_3 * Apart_5 * 1./w**2

            Neq[siz,iz]     = Neq[siz,iz]     + Apart_4 * Apart_1 * 1./w**2
            Neq[siz,iz+1]   = Neq[siz,iz+1]   + Apart_4 * Apart_2 * 1./w**2
            Neq[siz,pco_iz] = Neq[siz,pco_iz] + Apart_4 * Apart_3 * 1./w**2
            Neq[siz,siz]    = Neq[siz,siz]    + Apart_4 * Apart_4 * 1./w**2
            Neq[siz,siz+1]  = Neq[siz,siz+1]  + Apart_4 * Apart_5 * 1./w**2

            Neq[siz+1,iz]     = Neq[siz+1,iz]     + Apart_5 * Apart_1 * 1./w**2
            Neq[siz+1,iz+1]   = Neq[siz+1,iz+1]   + Apart_5 * Apart_2 * 1./w**2
            Neq[siz+1,pco_iz] = Neq[siz+1,pco_iz] + Apart_5 * Apart_3 * 1./w**2
            Neq[siz+1,siz]    = Neq[siz+1,siz]    + Apart_5 * Apart_4 * 1./w**2
            Neq[siz+1,siz+1]  = Neq[siz+1,siz+1]  + Apart_5 * Apart_5 * 1./w**2

        
    return Neq, AtWb

def neqBySite(params,svs,args):
    print("\t Reading in file:",params['filename'])
    #site_residuals = res.parseConsolidatedNumpy(filename,dt_start,dt_stop)
    site_residuals = res.parseConsolidatedNumpy(params['filename'])
    if args.model == 'pwl':
        Neq_tmp,AtWb_tmp = pwl(site_residuals,svs,args.nadir_grid)
    elif args.model == 'pwlSite':
        Neq_tmp,AtWb_tmp = pwlNadirSite(site_residuals,svs,params,args.nadir_grid,0.5)

    print("Returned Neq, AtWb:",np.shape(Neq_tmp),np.shape(AtWb_tmp))
            
    sf = filename+".npz"
    np.savez(sf,neq=Neq_tmp,atwb=AtWb_tmp,svs=svs)

    return sf 

#def setUpTasks(cl3files,svs,dt_start,dt_stop,opts):
def setUpTasks(cl3files,svs,opts,params):
    print('cpu_count() = {:d}\n'.format(multiprocessing.cpu_count()))
    NUMBER_OF_PROCESSES = multiprocessing.cpu_count()

    if opts.cpu < NUMBER_OF_PROCESSES:
        NUMBER_OF_PROCESSES = int(opts.cpu)

    print("Creating a pool of {:d} processes".format(NUMBER_OF_PROCESSES))

    pool = multiprocessing.Pool(NUMBER_OF_PROCESSES)
    #print("pool = {:s}".format(pool))
    # Submit the tasks
    results = []
    for i in range(0,np.size(cl3files)) :
        print("Submitting job:",params[i]['site'])
        results.append(pool.apply_async(neqBySite,(params[i],svs,opts)))
        #results.append(pool.apply_async(neqBySite,(cl3files[i],svs,opts)))
        #results.append(pool.apply_async(neqBySite,(cl3files[i],svs,dt_start,dt_stop,opts)))

    # Wait for all of them to finish before moving on
    for r in results:
        print("\t Waiting:",r.wait())
        #r.wait()


#=====================================
#
# TODO: time filter residuals
#       check station constraints --> no variance in areas of no observations
#       run through station file to check how many sitemodels need to be created
#       put in elevation dependent weighting
#
#=====================================
if __name__ == "__main__":
#    import warnings
#    warnings.filterwarnings("ignore")

    import argparse

    parser = argparse.ArgumentParser(prog='nadir',description='Create an Empirical Nadir Model from one-way GAMIT phase residuals',
            formatter_class=argparse.RawTextHelpFormatter,
            epilog='''\
    Example:

    To create a consolidated phase residual file:
    > python ~/gg/com/nadir.py --model -f ./t/YAR2.2012.CL3
                   ''')

    #===================================================================
    # Station meta data options
    parser.add_argument('-a', '--antex', dest='antex', default="~/gg/tables/antmod.dat",help="Location of ANTEX file (default = ~/gg/tables/antmod.dat)")
    parser.add_argument('--sv','--svnav', dest="svnavFile",default="~/gg/tables/svnav.dat", help="Location of GAMIT svnav.dat")
    parser.add_argument('--sf','--station_file', dest="station_file",default="~/gg/tables/station.info", help="Location of GAMIT station.info")

    
    parser.add_argument('--nadir_grid', dest='nadir_grid', default=0.1, type=float,help="Grid spacing to model NADIR corrections (default = 0.1 degrees)")
    parser.add_argument('--zenith_grid', dest='zen', default=0.5, type=float,help="Grid spacing to model Site corrections (default = 0.5 degrees)")
    parser.add_argument('-f', dest='resfile', default='',help="Consolidated one-way LC phase residuals")
    parser.add_argument('-p','--path',dest='path',help="Search for all CL3 files in the directory path") 

    parser.add_argument('-m','--model',dest='model',choices=['pwl','pwlSite','placeHolder'], help="Create a ESM for satellites only, or for satellites and sites")
    parser.add_argument('--save',dest='save_file',default=False, action='store_true',help="Save the Neq and Atwl matrices into numpy compressed format (npz)")
    parser.add_argument('-l','--load',dest='load_file',help="Load stored NEQ and AtWl matrices from a file")
    parser.add_argument('--lpath',dest='load_path',help="Path to search for .npz files")
   
    parser.add_argument('--ls','--load_site',dest='load_site',help="Load in the Neq and AtWl matrices for a specfic station")
    parser.add_argument('--cpu',dest='cpu',type=int,default=4,help="Maximum number of cpus to use")
    #===================================================================

    parser.add_argument("--syyyy",dest="syyyy",type=int,help="Start yyyy")
    parser.add_argument("--sdoy","--sddd",dest="sdoy",type=int,default=0,help="Start doy")
    parser.add_argument("--eyyyy",dest="eyyyy",type=int,help="End yyyyy")
    parser.add_argument("--edoy","--eddd",dest="edoy",type=int,default=365,help="End doy")

    #===================================================================
    # Plot options
    parser.add_argument('--plot',dest='plotNadir', default=False, action='store_true', help="Produce an elevation dependent plot of ESM phase residuals")
    
    #===================================================================
    # Debug function, not needed
    args = parser.parse_args()

    # expand any home directory paths (~) to the full path, otherwise python won't find the file
    if args.resfile : args.resfile = os.path.expanduser(args.resfile)
    args.antex = os.path.expanduser(args.antex)
    args.svnavFile = os.path.expanduser(args.svnavFile)

    svdat = []
    nadirData = {}
    cl3files = []
    npzfiles = []
    totalSites = 1

    if args.model: 
        #===================================================================
        # get the antenna information from an antex file
        antennas = ant.parseANTEX(args.antex)

        if args.resfile :
            cl3files.append(args.resfile)
        elif args.path:
            print("Checking {:s} for CL3 files".format(args.path))
            phsRGX = re.compile('.CL3')
            for root, dirs, files in os.walk(args.path):
                path = root.split('/')
                for lfile in files:
                    if phsRGX.search(lfile):
                        print("Found:",args.path + "/" + lfile)
                        cl3files.append(args.path + "/"+ lfile)
            cl3files = np.sort(cl3files)
            totalSites = np.size(cl3files)

        elif args.load_file:
            print("")
            print("Reading in the Neq and AtWb matrices from:",args.load_file)
            print("")

            npzfile = np.load(args.load_file)
            Neq  = npzfile['neq']
            AtWb = npzfile['atwb']
            svs  = np.sort(npzfile['svs'])
        elif args.load_path:
            phsRGX = re.compile('.npz')
            for root, dirs, files in os.walk(args.load_path):
                path = root.split('/')
                for lfile in files:
                    if phsRGX.search(lfile):
                        print("Found:",args.load_path + "/" + lfile)
                        npzfiles.append(args.load_path + "/"+ lfile)
            nctr = 0
            for nfile in (npzfiles):
                npzfile = np.load(nfile)
                Neq_tmp  = npzfile['neq']
                AtWb_tmp = npzfile['atwb']
                svs_tmp  = npzfile['svs']
                if nctr == 0:
                    Neq = Neq_tmp
                    AtWb = AtWb_tmp
                    svs = np.sort(svs_tmp)
                else:
                    Neq  = np.add(Neq,Neq_tmp)
                    AtWb = np.add(AtWb,AtWb_tmp)

                nctr += 1
            if args.save_file:
                np.savez('consolidated.npz',neq=Neq,atwb=AtWb,svs=svs)
        
        if not args.load_file and not args.load_path:
            # read in the consolidated LC residuals
            if args.syyyy and args.eyyyy:
                dt_start = dt.datetime(int(args.syyyy),01,01) + dt.timedelta(days=int(args.sdoy))
                dt_stop  = dt.datetime(int(args.eyyyy),01,01) + dt.timedelta(days=int(args.edoy))
            else:
                print("")
                print("Warning:")
                print("\tusing:",cl3files[0],"to work out the time period to determine how many satellites were operating.")
                print("")
                site_residuals = res.parseConsolidatedNumpy(cl3files[0])
                dt_start = gt.unix2dt(site_residuals[0,0])
                res_start = int(dt_start.strftime("%Y") + dt_start.strftime("%j"))
                dt_stop = gt.unix2dt(site_residuals[-1,0])
                res_stop = int(dt_stop.strftime("%Y") + dt_stop.strftime("%j"))
                print("\tResiduals run from:",res_start,"to:",res_stop)
                del site_residuals

            #=====================================================================
            # Work out how many satellites we need to solve for
            #=====================================================================
            svdat = svnav.parseSVNAV(args.svnavFile)
            svs = ant.satSearch(antennas,dt_start,dt_stop)
            svs = np.sort(svs)

            #=====================================================================
            # Work out how many station models need to be created 
            #=====================================================================
            numModels = 0
            params = []

            for f in range(0,np.size(cl3files)):
                filename = os.path.basename(cl3files[f])
                siteID = filename[0:4]
                sdata = gsf.parseSite(args.station_file,siteID.upper())
                changes = gsf.determineESMChanges(dt_start,dt_stop,sdata)
                numModels = numModels + np.size(changes['ind']) + 1
                info = {}
                info['filename']  = cl3files[f]
                info['basename']  = filename
                info['site']      = siteID
                info['numModels'] = np.size(changes['ind']) + 1 
                info['changes']   = changes
                params.append(info)

            #=====================================================================
            # add one to make sure we have a linspace which includes 0.0 and 14.0
            # add another parameter for the zenith PCO estimate
            #=====================================================================
            numNADS = int(14.0/args.nadir_grid) + 1 
            PCOEstimates = 1
            numSVS = np.size(svs)
            numParamsPerSat = numNADS + PCOEstimates
            tSat = numParamsPerSat * numSVS
            numSites = numModels # np.size(cl3files)
            tSite = 0
            numParamsPerSite = 0

            if args.model == 'pwl':
                numParams = numSVS * (numParamsPerSat)
            elif args.model == 'pwlSite':
                numParamsPerSite = int(90./args.zen) + 1 
                numParams = numSVS * (numParamsPerSat) + numParamsPerSite * numSites
                tSite = numParamsPerSite * numSites

            print("Total satellite parameters:",tSat)
            print("Total site parameters     :",tSite)
            print("\t Have:",numParams,"parameters to solve for")

            #========================================================================
            # Adding Constraints to the satellite parameters,
            # keep the site model free ~ 10mm  0.01 => 1/sqrt(0.01) = 10 (mm)
            # Adding 1 mm constraint to satellites
            #========================================================================
            sPCV_constraint = 0.01
            sPCV_window = 1.0     # assume the PCV variation is correlated at this degree level
            site_constraint = 10.0
            site_window = 1.5

            C = np.eye(numParams,dtype=float) * sPCV_constraint
            if args.model == 'pwlSite' :
                for sitr in range(0,tSite):
                    spar = tSat + sitr
                    C[spar,spar] = site_constraint 

                # Now add in the off digonal commponents
                sPCV_corr = np.linspace(sPCV_constraint, 0., int(sPCV_window/args.nadir_grid))
                site_corr = np.linspace(site_constraint, 0., int(site_window/args.zen))

                # Add in the correlation constraints for the satellite PCVs
                for s in range(0,numSVS):
                    for ind in range(0,numNADS ):
                        start = (s * numParamsPerSat) + ind
                        if ind > (numNADS - np.size(sPCV_corr)):
                            end = start + (numNADS - ind) 
                        else:
                            end = start + np.size(sPCV_corr)
                        
                        C[start,start:end] = sPCV_corr[0:(end - start)] 
                        C[start:end,start] = sPCV_corr[0:(end - start)] 

                for s in range(0,numSites):
                    for ind in range(0,numParamsPerSite-np.size(site_corr) ):
                        start = tSat + (s * numParamsPerSite) + ind
                        if ind > (numParamsPerSite - np.size(site_corr)):
                            end = start + (numParamsPerSite - ind) 
                        else:
                            end = start + np.size(site_corr)
                        
                        C[start,start:end] = site_corr[0:(end - start)] 
                        C[start:end,start] = site_corr[0:(end - start)] 

            C_inv = np.linalg.inv(C)
            del C

            Neq = np.zeros((numParams,numParams))
            AtWb = np.zeros(numParams)

            Neq = np.add(Neq,C_inv)

            print("Will have to solve for ",np.size(svs),"sats",svs)
            print("\t Creating a PWL linear model for Nadir satelites for SVS:\n")

            multiprocessing.freeze_support()
            setUpTasks(cl3files,svs,args,params)

            #=====================================================================
            # Now read in all of the numpy compressed files
            #=====================================================================
            npyRGX = re.compile('.npz')
            for root, dirs, files in os.walk(args.path):
                path = root.split('/')
                for lfile in files:
                    if npyRGX.search(lfile):
                        print("Found:",args.path + "/" + lfile)
                        npzfiles.append(args.path + "/"+ lfile)
            # Start stacking the normal equations together
            nctr = 0
            for nfile in (npzfiles):
                npzfile = np.load(nfile)
                Neq_tmp  = npzfile['neq']
                AtWb_tmp = npzfile['atwb']

                # only need one copy of the svs array, they should be eactly the same
                if nctr == 0:
                    svs_tmp  = npzfile['svs']
                    svs = np.sort(svs_tmp)
                    del svs_tmp

                # Add the svn component to the Neq
                Neq[0:tSat-1,0:tSat-1] = Neq[0:tSat -1,0:tSat-1] + Neq_tmp[0:tSat-1,0:tSat-1]
                AtWb[0:tSat-1] = AtWb[0:tSat-1] + AtWb_tmp[0:tSat-1]

                # Add in the station dependent models
                start = tSat + nctr * numParamsPerSite 
                end = tSat + (nctr+1) * numParamsPerSite

                AtWb[start:end] = AtWb[start:end] + AtWb_tmp[tSat:(tSat+numParamsPerSite)]
                #print(np.shape(AtWb))
                #   ------------------------------------------------
                #  | SVN         | svn + site | svn + site2 | ....
                #  | svn + site  | site       | 0           | ....
                #  | svn + site2 | 0          | site2       | ....
                #

                # Add in the site block 
                Neq[start:end,start:end] = Neq[start:end,start:end]+ Neq_tmp[tSat:(tSat+numParamsPerSite),tSat:(tSat+numParamsPerSite)]

                # Adding in the correlation with the SVN and site
                Neq[0:tSat-1,start:end] = Neq[0:tSat-1,start:end] + Neq_tmp[0:tSat-1,tSat:(tSat+numParamsPerSite)]
                Neq[start:end,0:tSat-1] = Neq[start:end,0:tSat-1] + Neq_tmp[tSat:(tSat+numParamsPerSite),0:tSat-1]
                Send = tSat+numParamsPerSite
                #print("Neq_tmp:",np.shape(Neq_tmp),tSat,Send)
                #print("Neq:",np.shape(Neq),end)
                nctr += 1

            if args.save_file:
                np.savez('consolidated.npz',neq=Neq,atwb=AtWb,svs=svs)
        
            #=====================================================================
            # End of if not load_file or not load_path
            #=====================================================================
            print("FINISHED MP processing, now need to workout stacking:...\n") 
        if args.load_site:
            sitefile = np.load(args.load_site)
            S_Neq  = sitefile['neq']
            S_AtWb = sitefile['atwb']
            print("Neq:",np.shape(Neq),"AtWl:",np.shape(AtWb))
            print("S_Neq:",np.shape(S_Neq),"S_AtWl:",np.shape(S_AtWb))
            #AtWb = np.vstack((AtWb,S_AtWb))
            AtWb = np.concatenate((AtWb,S_AtWb))
            print("Neq:",np.shape(Neq),"AtWl:",np.shape(AtWb))
            from scipy.linalg import block_diag
            Neq = block_diag(Neq,S_Neq)
            print("Neq:",np.shape(Neq),"AtWl:",np.shape(AtWb))

        if not args.save_file:
            print("Now trying an inverse")
            Cov = np.linalg.pinv(Neq)
        
            print("Now computing the solution")
            Sol = np.dot(Cov,AtWb)
            print("The solution is :",np.shape(Sol))

            #f = loglikelihood(np.array(meas_complete),np.array(model_complete))
            #numd = np.size(meas_complete)
            #dof = numd - np.shape(Sol_complete)[0]
            #aic = calcAIC(f,dof)
            #bic = calcBIC(f,dof,numd)
            #prechi = np.dot(data[:,3].T,data[:,3])
            #prechi = np.dot(np.array(meas_complete).T,np.array(meas_complete))
            #postchi = prechi - np.dot(np.array(Bvec_complete).T,np.array(Sol_complete))
            #print("My loglikelihood:",f,aic,bic,dof,numd)
            #print("STATS:",numd,np.sqrt(prechi/numd),np.sqrt(postchi/numd),np.sqrt((prechi-postchi)/numd),aic,bic)
            #print("MaX PCO:",max_pco_iz)
        if args.plotNadir:
            nad = np.linspace(0,14, int(14./args.nadir_grid)+1 )
            numParamsPerSat = int(14.0/args.nadir_grid) + 2 

            variances = np.diag(Neq)
            print("Variance:",np.shape(variances))
            del Neq, AtWb

            ctr = 0
            for svn in svs:
                fig = plt.figure(figsize=(3.62, 2.76))
                fig.canvas.set_window_title("SVN_"+svn+"_nadirCorrectionModel.png")
                ax = fig.add_subplot(111)

                siz = numParamsPerSat * ctr 
                eiz = numParamsPerSat *ctr + numNADS 
                print("SVN:",svn,siz,eiz,numParamsPerSat,tSat)
                #ax.plot(nad,Sol[siz:eiz],'r-',linewidth=2)
                ax.errorbar(nad,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,fmt='o')

                ax.set_xlabel('Nadir Angle (degrees)',fontsize=8)
                ax.set_ylabel('Phase Residuals (mm)',fontsize=8)
                #ax.set_xlim([0, 14])

                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                           ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(8)

                plt.tight_layout()
                ctr += 1
                
                if ctr > 10:
                    break

            #==================================================
            fig = plt.figure(figsize=(3.62, 2.76))
            fig.canvas.set_window_title("PCO_correction.png")
            ax = fig.add_subplot(111)
            ctr = 0
            numSVS = np.size(svs)
            numNADS = int(14.0/args.nadir_grid) + 1 
            numParamsPerSat = numNADS + PCOEstimates
            print("Number of Params per Sat:",numParamsPerSat,"numNads",numNADS,"Sol",np.shape(Sol))
            for svn in svs:
                eiz = numParamsPerSat *ctr + numParamsPerSat -1 
                #print(ctr,"PCO:",eiz)
                ax.plot(ctr,Sol[eiz],'k.',linewidth=2)
                #ax.errorbar(ctr,Sol[eiz],yerr=np.sqrt(variances[eiz])/2.,fmt='o')
                ctr += 1

            ax.set_xlabel('SVN',fontsize=8)
            ax.set_ylabel('Adjustment to PCO (mm)',fontsize=8)
            #ax.set_xlim([0, 14])

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                       ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(8)

            plt.tight_layout()

            #==================================================
            if args.model == 'pwlSite':
                ctr = 0
                numSVS = np.size(svs)
                numNADS = int(14.0/args.nadir_grid) + 1 
                numParamsPerSat = numNADS + PCOEstimates
                print("Number of Params per Sat:",numParamsPerSat,"numNads",numNADS,"Sol",np.shape(Sol))
                numParams = numSVS * (numParamsPerSat) + numParamsPerSite * totalSites 
                for snum in range(0,totalSites):
                    fig = plt.figure(figsize=(3.62, 2.76))
                    fig.canvas.set_window_title(cl3files[snum]+"_elevation_model.png")
                    ax = fig.add_subplot(111)
                    siz = numParamsPerSat*numSVS + snum * numParamsPerSite 
                    eiz = siz + numParamsPerSite 
                    ele = np.linspace(0,90,numParamsPerSite)
                    print("Sol",np.shape(Sol),"siz  ",siz,eiz)
                    #ax.plot(ele,Sol[siz:eiz],'k.',linewidth=2)
                    ax.errorbar(ele,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,fmt='o')

                    ax.set_xlabel('Elevation Angle',fontsize=8)
                    ax.set_ylabel('Adjustment to PCO (mm)',fontsize=8)
                    #ax.set_xlim([0, 14])

                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(8)

                    plt.tight_layout()

            plt.show()

    print("FINISHED")
