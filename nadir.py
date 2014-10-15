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

#import statsmodels.api as sm
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

def pwl(site_residuals, svs, Neq, AtWb,nadSpacing=0.1,):
    """
    PWL piece-wise-linear interpolation fit of phase residuals
    -construct a PWL fit for each azimuth bin, and then paste them all together to get 
     the full model
    -inversion is doen within each bin

    cdata -> compressed data
    """
    print("rejecting any residuals greater than 100mm",np.shape(site_residuals))
    tdata = res.reject_absVal(site_residuals,100.)
    del site_residuals 
    print("rejecting any residuals greater than 5 sigma",np.shape(tdata))
    data = res.reject_outliers_elevation(tdata,5,0.5)
    print("finished outlier detection",np.shape(data))
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
    #print("\t Have:",numParams,"parameters to solve for")

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
        #Apart[i,iz] = (1.-(nadir-iz*nadSpacing)/nadSpacing)
        Apart_1 = (1.-(nadir-niz*nadSpacing)/nadSpacing)
        #Apart[i,iz+1] = (nadir-iz*nadSpacing)/nadSpacing
        Apart_2 = (nadir-niz*nadSpacing)/nadSpacing

        # Now  add in the PCO offsest into the Neq
        Apart_3 = 1./np.sin(np.radians(nadir)) 

        #print("i:{:d}, nadir {:.3f}, iz {:.3f}, nadSpacing {:.3f}, Apart_1 {:.3f}".format(i,nadir,niz,nadSpacing,Apart_1))
        w = 1. #np.sin(data[i,2]/180.*np.pi)
        
        Neq[iz,iz] += (Apart_1*Apart_1) * 1./w**2
        Neq[iz,iz+1] += (Apart_2*Apart_1) * 1./w**2
        Neq[iz+1,iz] += (Apart_2*Apart_1) * 1./w**2
        Neq[iz+1,iz+1] += (Apart_2*Apart_2) * 1./w**2
        Neq[iz,pco_iz] += (Apart_1*Apart_3) * 1./w**2
        Neq[iz+1,pco_iz] += (Apart_2*Apart_3) * 1./w**2
        Neq[pco_iz,iz] += (Apart_1*Apart_3) * 1./w**2
        Neq[pco_iz,iz+1] += (Apart_2*Apart_3) * 1./w**2
        Neq[pco_iz,pco_iz] += (Apart_3 * Apart_3) * 1./w**2

        AtWb[iz] += Apart_1 * data[i,3] * 1./w**2
        AtWb[iz+1] += Apart_2 * data[i,3] * 1./w**2
        AtWb[pco_iz] += Apart_3 * data[i,3] * 1./w**2
        #print("nadir {:.2f}, iz {:d}, pco_iz {:d}, el {:.2f}, w {:.2f}, Apart_1 {:2f}, data {:.2f}, AtWb {:.3f}".format(nadir,iz,pco_iz,data[i,2],w,Apart_1,data[i,3],AtWb[iz]))

        # PCO partial ...
        #Apart_3 = 1./np.cos(np.radians(nadir)) 
        #print("nadir {:.2f}, iz{:d}, pco_iz{:d}, Apart_3 {:.3f}".format(nadir,iz,pco_iz,Apart_3))

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
    return Neq, AtWb

def pwlNadirSite(site_residuals, svs, Neq, AtWb,nadSpacing=0.1,zenSpacing=0.5,totalSites=1,siteNumber=0):
    """
    Create a model for the satellites and sites at the same time.
    PWL piece-wise-linear interpolation fit of phase residuals
    -construct a PWL fit for each azimuth bin, and then paste them all together to get 
     the full model
    -inversion is done within each bin

    cdata -> compressed data
    """
    print("rejecting any residuals greater than 100mm",np.shape(site_residuals))
    tdata = res.reject_absVal(site_residuals,100.)
    del site_residuals 
    print("rejecting any residuals greater than 5 sigma",np.shape(tdata))
    data = res.reject_outliers_elevation(tdata,5,0.5)
    print("finished outlier detection",np.shape(data))
    del tdata

    print("Creating a model for the Satellites and Station Simultaneously for site number",siteNumber+1,"of",totalSites)
    
    # Get the total number of observations for this site
    numd = np.shape(data)[0]
    # add one to make sure we have a linspace which includes 0.0 and 14.0
    # add another parameter for the zenith PCO estimate
    numNADS = int(14.0/nadSpacing) + 1 
    PCOEstimates = 1
    numSVS = np.size(svs)
    numParamsPerSat = numNADS + PCOEstimates

    numParamsPerSite = int(90.0/zenSpacing) + 1
    numParams = numSVS * (numParamsPerSat) + numParamsPerSite*totalSites
    print("\t Have:",numParams,"parameters to solve for")

    for i in range(0,numd):
        # work out the nadir angle
        nadir = calcNadirAngle(data[i,2])
        niz = int(np.floor(nadir/nadSpacing))

        nsiz = int(np.floor(data[i,2]/zenSpacing))
        siz = int( numParamsPerSat*numSVS +  siteNumber*numParamsPerSite + nsiz)

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
        iz = int(numParamsPerSat * ctr + niz)
        pco_iz = int(numParamsPerSat *ctr + numParamsPerSat -1)

        #if iz >= numParams or pco_iz > numParams:
        #    print("prn,svn_search,svn,ctr,size(svs),niz,iz,nadir,numParams:",data[i,4],svn_search,svn,ctr,np.size(svs),niz,iz,nadir,numParams)
        #    print(svs)

        # Nadir partials..
        Apart_1 = (1.-(nadir-niz*nadSpacing)/nadSpacing)
        Apart_2 = (nadir-niz*nadSpacing)/nadSpacing
        # PCO partial ...
        Apart_3 = 1./np.sin(np.radians(nadir)) 
        # Site partials
        Apart_4 = (1.-(data[i,2]-nsiz*zenSpacing)/zenSpacing)
        #print("Apart_4 {:.2f}, ELE: {:.2f}, nsiz {:d}, siz {:d}".format(Apart_4,data[i,2],nsiz,siz))
        Apart_5 = (data[i,2]-nsiz*zenSpacing)/zenSpacing
        #print("Apart_5 {:.2f}, ELE: {:.2f}, nsiz {:d}, siz {:d}".format(Apart_5,data[i,2],nsiz,siz))

        AtWb[iz]     = AtWb[iz] + Apart_1 * data[i,3] * 1./w**2
        AtWb[iz+1]   = AtWb[iz+1] + Apart_2 * data[i,3] * 1./w**2
        AtWb[pco_iz] = AtWb[pco_iz] + Apart_3 * data[i,3] * 1./w**2
        AtWb[siz]    = AtWb[siz] + Apart_4 * data[i,3] * 1./w**2
        AtWb[siz+1]  = AtWb[siz+1] + Apart_5 * data[i,3] * 1./w**2

        Neq[iz,iz]     = Neq[iz,iz] + Apart_1 * Apart_1 * 1./w**2
        Neq[iz,iz+1]   = Neq[iz,iz+1] + Apart_1 * Apart_2 * 1./w**2
        Neq[iz,pco_iz] = Neq[iz,pco_iz] + Apart_1 * Apart_3 * 1./w**2
        Neq[iz,siz]    = Neq[iz,siz] + Apart_1 * Apart_4 * 1./w**2
        Neq[iz,siz+1]  = Neq[iz,siz+1] + Apart_1 * Apart_5 * 1./w**2

        Neq[iz+1,iz]     = Neq[iz+1,iz] + Apart_2 * Apart_1 * 1./w**2
        Neq[iz+1,iz+1]   = Neq[iz+1,iz+1] + Apart_2 * Apart_2 * 1./w**2
        Neq[iz+1,pco_iz] = Neq[iz+1,pco_iz] + Apart_2 * Apart_3 * 1./w**2
        Neq[iz+1,siz]    = Neq[iz+1,siz] + Apart_2 * Apart_4 * 1./w**2
        Neq[iz+1,siz+1]  = Neq[iz+1,siz+1] + Apart_2 * Apart_5 * 1./w**2

        Neq[pco_iz,iz]     = Neq[pco_iz,iz] + Apart_3 * Apart_1 * 1./w**2
        Neq[pco_iz,iz+1]   = Neq[pco_iz,iz+1] + Apart_3 * Apart_2 * 1./w**2
        Neq[pco_iz,pco_iz] = Neq[pco_iz,pco_iz] + Apart_3 * Apart_3 * 1./w**2
        Neq[pco_iz,siz]    = Neq[pco_iz,siz] + Apart_3 * Apart_4 * 1./w**2
        Neq[pco_iz,siz+1]  = Neq[pco_iz,siz+1] + Apart_3 * Apart_5 * 1./w**2

        Neq[siz,iz]     = Neq[siz,iz] + Apart_4 * Apart_1 * 1./w**2
        Neq[siz,iz+1]   = Neq[siz,iz+1] + Apart_4 * Apart_2 * 1./w**2
        Neq[siz,pco_iz] = Neq[siz,pco_iz] + Apart_4 * Apart_3 * 1./w**2
        Neq[siz,siz]    = Neq[siz,siz] + Apart_4 * Apart_4 * 1./w**2
        Neq[siz,siz+1]  = Neq[siz,siz+1] + Apart_4 * Apart_5 * 1./w**2

        Neq[siz+1,iz]     = Neq[siz+1,iz] + Apart_5 * Apart_1 * 1./w**2
        Neq[siz+1,iz+1]   = Neq[siz+1,iz+1] + Apart_5 * Apart_2 * 1./w**2
        Neq[siz+1,pco_iz] = Neq[siz+1,pco_iz] + Apart_5 * Apart_3 * 1./w**2
        Neq[siz+1,siz]    = Neq[siz+1,siz] + Apart_5 * Apart_4 * 1./w**2
        Neq[siz+1,siz+1]  = Neq[siz+1,siz+1] + Apart_5 * Apart_5 * 1./w**2

        #print("nadir {:.2f}, iz{:d}, pco_iz{:d}, Apart_3 {:.3f}".format(nadir,iz,pco_iz,Apart_3))

        
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
    return Neq, AtWb

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

    
    parser.add_argument('--nadir_grid', dest='nadir_grid', default=0.1, type=float,help="Grid spacing to model NADIR corrections (default = 0.1 degrees)")
    parser.add_argument('--zenith_grid', dest='zen', default=0.5, type=float,help="Grid spacing to model Site corrections (default = 0.5 degrees)")
    parser.add_argument('-f', dest='resfile', default='',help="Consolidated one-way LC phase residuals")
    parser.add_argument('-p','--path',dest='path',help="Search for all CL3 files in the directory path") 

    parser.add_argument('-m','--model',dest='model',choices=['pwl','pwlSite','placeHolder'], help="Create a ESM for satellites only, or for satellites and sites")
    parser.add_argument('--save',dest='save_file',default=False, action='store_true',help="Save the Neq and Atwl matrices into numpy compressed format (npz)")
    parser.add_argument('-l','--load',dest='load_file',help="Load stored NEQ and AtWl matrices from a file")
    parser.add_argument('--lpath',dest='load_path',help="Path to search for .npz files")
   
    parser.add_argument('--ls','--load_site',dest='load_site',help="Load in the Neq and AtWl matrices for a specfic station")
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
            #print("Neq shape:",np.shape(Neq))
            #for i in range(0,np.shape(Neq)[0]):
            #    print("Neq[0,",i,"]",Neq[200,i])
            #ctr = 0
            #for val in AtWb:
            #    print("AtWb", ctr, val)
            #    ctr += 1
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
            print("")
            print("Reading in the consolidated phase residuals from:",args.resfile)
            print("")
            site_residuals = res.parseConsolidatedNumpy(cl3files[0])
            #site_residuals = res.parseConsolidatedNumpy(args.resfile)

            if args.syyyy and args.eyyyy:
                dt_start = dt.datetime(int(args.syyyy),01,01) + dt.timedelta(days=int(args.sdoy))
                dt_stop  = dt.datetime(int(args.eyyyy),01,01) + dt.timedelta(days=int(args.edoy))
            else:
                print("")
                print("Warning:")
                print("Using:",args.resfile,"to work out the time period to deterimine how man satellites weere operating")
                print("")
                dt_start = gt.unix2dt(site_residuals[0,0])
                res_start = int(dt_start.strftime("%Y") + dt_start.strftime("%j"))
                dt_stop = gt.unix2dt(site_residuals[-1,0])
                res_stop = int(dt_stop.strftime("%Y") + dt_stop.strftime("%j"))
                print("\tResiduals run from:",res_start,"to:",res_stop)

            svdat = svnav.parseSVNAV(args.svnavFile)
            svs = ant.satSearch(antennas,dt_start,dt_stop)
            svs = np.sort(svs)
            #=====================================================================
            # add one to make sure we have a linspace which includes 0.0 and 14.0
            # add another parameter for the zenith PCO estimate

            numNADS = int(14.0/args.nadir_grid) + 1 
            PCOEstimates = 1
            # 0 => 140 PCV, 141 PCO
            # 142 => 283 PCV, 284 PCO
            numSVS = np.size(svs)
            numParamsPerSat = numNADS + PCOEstimates
            if args.model == 'pwl':
                numParams = numSVS * (numParamsPerSat)
            elif args.model == 'pwlSite':
                numParamsPerSite = int(90./args.zen) + 1 
                numParams = numSVS * (numParamsPerSat) + numParamsPerSite * np.size(cl3files)
            print("\t Have:",numParams,"parameters to solve for")

            Neq = np.zeros((numParams,numParams)) #dtype=float) * 0.001
            AtWb = np.zeros(numParams)

            print("Will have to solve for ",np.size(svs),"sats",svs)
            print("\t Creating a PWL linear model for Nadir satelites for SVS:\n")
            print("\t Reading in file:",args.resfile)

            for i in range(0,np.size(cl3files)) :
                # we don't need to read the residuals in for the first iteration
                # this has already been done previously to scan for start and stop times
                if i > 0:
                    site_residuals = res.parseConsolidatedNumpy(cl3files[i])

                if args.model == 'pwl':
                    Neq_tmp,AtWb_tmp = pwl(site_residuals,svs,Neq,AtWb,args.nadir_grid)
                elif args.model == 'pwlSite':
                    Neq_tmp,AtWb_tmp = pwlNadirSite(site_residuals,svs,Neq,AtWb,args.nadir_grid,0.5,np.size(cl3files),i)

                Neq = np.add(Neq,Neq_tmp)
                AtWb = np.add(AtWb,AtWb_tmp)

                del site_residuals,Neq_tmp,AtWb_tmp
            
                print("Returned Neq, AtWb:",np.shape(Neq),np.shape(AtWb))
            
                if args.save_file:
                    print("Saving the Neq and AtWb arrays")
                    sf = cl3files[i]+".npz"
                    np.savez(sf,neq=Neq,atwb=AtWb,svs=svs)
            
            #for i in range(0,np.shape(Neq)[0]):
            #    print("Neq[0,",i,"]",Neq[200,i])
            #=====================================================================
            # End of if not load_file or not load_path
            #=====================================================================
            
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

        if args.plotNadir:
            nad = np.linspace(0,14, int(14./args.nadir_grid)+1 )
            num_nad = int(14./args.nadir_grid) + 1
            ctr = 0
            numParamsPerSat = int(14.0/args.nadir_grid) + 1 

            del Neq, AtWb

            for svn in svs:
                fig = plt.figure(figsize=(3.62, 2.76))
                fig.canvas.set_window_title("SVN_"+svn+"_nadirCorrectionModel.png")
                ax = fig.add_subplot(111)

                siz = numParamsPerSat * ctr 
                eiz = numParamsPerSat *ctr + numParamsPerSat 
                ax.plot(nad,Sol[siz:eiz],'r-',linewidth=2)

                ax.set_xlabel('Nadir Angle (degrees)',fontsize=8)
                ax.set_ylabel('Phase Residuals (mm)',fontsize=8)
                #ax.set_xlim([0, 14])

                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                           ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(8)

                plt.tight_layout()
                ctr += 1
                
                if ctr > 5:
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
                print(ctr,"PCO:",eiz)
                ax.plot(ctr,Sol[eiz],'k.',linewidth=2)
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
                    fig.canvas.set_window_title("Site_"+str(snum)+"_elevation_model.png")
                    ax = fig.add_subplot(111)
                    siz = numParamsPerSat*numSVS + snum * numParamsPerSite 
                    eiz = siz + numParamsPerSite 
                    ele = np.linspace(0,90,numParamsPerSite)
                    print("Sol",np.shape(Sol),"siz  ",siz,eiz)
                    ax.plot(ele,Sol[siz:eiz],'k.',linewidth=2)

                    ax.set_xlabel('Elevation Angle',fontsize=8)
                    ax.set_ylabel('Adjustment to PCO (mm)',fontsize=8)
                    #ax.set_xlim([0, 14])

                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(8)

                    plt.tight_layout()


            plt.show()

    print("FINISHED")
