#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import matplotlib
matplotlib.use('Agg')

import numpy as np
import re
import calendar
import os, sys
import datetime as dt

import multiprocessing

import antenna as ant
import residuals as res
import gpsTime as gt
import GamitStationFile as gsf
import svnav

def satelliteModel(antenna,nadirData):
    #assuming a 14 model at 1 deg intervals
    ctr = 0
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
    tSat = numParamsPerSat * numSVS

    numParamsPerSite = int(90.0/zenSpacing) + 1
    tSite = numParamsPerSite*params['numModels']
    #numParams = numSVS * (numParamsPerSat) + numParamsPerSite*params['numModels']
    numParams = tSat + tSite 

    #print("\t Have:",numParams,"parameters to solve for",params['site'],"Number of Models:",params['numModels'])
    print("------------------------------------------------")
    print("Processing Site:                        ",params['site'])
    print("------------------------------------------------")
    print("Sat Params:----------------",numParamsPerSat)
    print("Number of Sats:------------",np.size(svs))
    print("Total satellite parameters:-------------",tSat)
    print("Site Params:---------------",numParamsPerSite)
    print("Number of Models:----------",params['numModels'])
    print("Total Site Params:----------------------",tSite)
    print("------------------------------------------------")
    print("Total Params:---------------------------",numParams)
    print("------------------------------------------------")

    # Creating matrices
    Neq = np.zeros((numParams,numParams))
    AtWb = np.zeros(numParams)
    change = params['changes']

    for m in range(0,int(params['numModels'])):
        print(params['site'],"----> creating model",m+1,"of",params['numModels'])

        # start_yyyy and start_ddd should always be defind, however stop_dd may be absent
        #ie no changes have ocured since the last setup
        minVal_dt = gt.ydhms2dt(change['start_yyyy'][m],change['start_ddd'][m],0,0,0)

        if np.size(change['stop_ddd']) > m  :
            maxVal_dt = gt.ydhms2dt(change['stop_yyyy'][m],change['stop_ddd'][m],23,59,59)
            print("Min:",minVal_dt,"Max:",maxVal_dt,m,np.size(change['stop_ddd']))
            criterion = ( ( site_residuals[:,0] >= calendar.timegm(minVal_dt.utctimetuple()) ) &
                    ( site_residuals[:,0] < calendar.timegm(maxVal_dt.utctimetuple()) ) )
        else:
            criterion = ( site_residuals[:,0] >= calendar.timegm(minVal_dt.utctimetuple()) ) 

        mind = np.array(np.where(criterion))[0]
        #print("rejecting any residuals greater than 100mm",np.shape(site_residuals))
        tdata = res.reject_absVal(site_residuals[mind,:],100.)
        if m >= (int(params['numModels']) -1 ):
            del site_residuals

        #print("rejecting any residuals greater than 5 sigma",np.shape(tdata))
        data = res.reject_outliers_elevation(tdata,5,0.5)
        #print("finished outlier detection",np.shape(data))
        del tdata

        # Get the total number of observations for this site
        numd = np.shape(data)[0]
        #print("Have:",numd,"observations")
        for i in range(0,numd):
            # work out the nadir angle
            nadir = calcNadirAngle(data[i,2])
            niz = int(np.floor(nadir/nadSpacing))

            nsiz = int(np.floor(data[i,2]/zenSpacing))
            siz = int( tSat +  m*numParamsPerSite + nsiz)

            # work out the svn number
            svndto =  gt.unix2dt(data[i,0])
            svn = svnav.findSV_DTO(svdat,data[i,4],svndto)
            svn_search = 'G{:03d}'.format(svn) 
            ctr = 0
            for sv in svs:
                if sv == svn_search:
                    break
                ctr+=1

            w = 1. #np.sin(data[i,2]/180.*np.pi)
            iz = int(numParamsPerSat * ctr + niz)
            pco_iz = int(numParamsPerSat *ctr + numNADS )

            #print("Indices m,iz,pco_iz,siz:",m,iz,pco_iz,siz,i,numd)
            # Nadir partials..
            Apart_1 = (1.-(nadir-niz*nadSpacing)/nadSpacing)
            Apart_2 = (nadir-niz*nadSpacing)/nadSpacing
            # PCO partial ...
            Apart_3 = 1./np.sin(np.radians(nadir)) 
            # Site partials
            Apart_4 = (1.-(data[i,2]-nsiz*zenSpacing)/zenSpacing)
            Apart_5 = (data[i,2]-nsiz*zenSpacing)/zenSpacing
            #print("Finished forming Design matrix")

            #print("Starting AtWb",np.shape(AtWb),iz,pco_iz,siz)
            AtWb[iz]     = AtWb[iz]     + Apart_1 * data[i,3] * 1./w**2
            AtWb[iz+1]   = AtWb[iz+1]   + Apart_2 * data[i,3] * 1./w**2
            AtWb[pco_iz] = AtWb[pco_iz] + Apart_3 * data[i,3] * 1./w**2
            AtWb[siz]    = AtWb[siz]    + Apart_4 * data[i,3] * 1./w**2
            AtWb[siz+1]  = AtWb[siz+1]  + Apart_5 * data[i,3] * 1./w**2
            #print("Finished forming b vector")

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
            #print("Finished NEQ Nadir estimates")
            
            Neq[pco_iz,iz]     = Neq[pco_iz,iz]     + Apart_3 * Apart_1 * 1./w**2
            Neq[pco_iz,iz+1]   = Neq[pco_iz,iz+1]   + Apart_3 * Apart_2 * 1./w**2
            Neq[pco_iz,pco_iz] = Neq[pco_iz,pco_iz] + Apart_3 * Apart_3 * 1./w**2
            Neq[pco_iz,siz]    = Neq[pco_iz,siz]    + Apart_3 * Apart_4 * 1./w**2
            Neq[pco_iz,siz+1]  = Neq[pco_iz,siz+1]  + Apart_3 * Apart_5 * 1./w**2
            #print("Finished NEQ PCO estimates")

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
            #print("Finished NEQ Site estimates")

    prechi = np.dot(data[:,3].T,data[:,3])
    print("Normal finish of pwlNadirSite",prechi,numd)
    return Neq, AtWb, prechi, numd

def neqBySite(params,svs,args):
    print("\t Reading in file:",params['filename'])
    #site_residuals = res.parseConsolidatedNumpy(filename,dt_start,dt_stop)
    site_residuals = res.parseConsolidatedNumpy(params['filename'])
    if args.model == 'pwl':
        Neq_tmp,AtWb_tmp = pwl(site_residuals,svs,args.nadir_grid)
    elif args.model == 'pwlSite':
        Neq_tmp,AtWb_tmp,prechi_tmp,numd_tmp = pwlNadirSite(site_residuals,svs,params,args.nadir_grid,0.5)

    print("Returned Neq, AtWb:",np.shape(Neq_tmp),np.shape(AtWb_tmp))
            
    sf = params['filename']+".npz"
    non_zero = 0
    for i in range(0,np.shape(Neq_tmp)[0]):
        criterion = (np.abs(Neq_tmp[:,i]) > 0.00001)
        non_zero = non_zero + np.size(np.array(np.where(criterion)))
    Msize = np.size(Neq_tmp)
    density = np.float(non_zero)/np.float(Msize)
    print("Density of Neq:{:.3f} {:d} {:d}".format(density,non_zero,Msize))
    np.savez_compressed(sf,neq=Neq_tmp,atwb=AtWb_tmp,svs=svs,prechi=[prechi],numd=[numd])

    return prechi_tmp, numd_tmp 

def setUpTasks(cl3files,svs,opts,params):
    prechi = 0
    numd = 0
    print('cpu_count() = {:d}\n'.format(multiprocessing.cpu_count()))
    NUMBER_OF_PROCESSES = multiprocessing.cpu_count()

    if opts.cpu < NUMBER_OF_PROCESSES:
        NUMBER_OF_PROCESSES = int(opts.cpu)

    #print("Creating a pool of {:d} processes".format(NUMBER_OF_PROCESSES))

    pool = multiprocessing.Pool(NUMBER_OF_PROCESSES)

    # Submit the tasks
    results = []
    for i in range(0,np.size(cl3files)) :
        print("Submitting job:",params[i]['site'])
        results.append(pool.apply_async(neqBySite,(params[i],svs,opts)))

    # Wait for all of them to finish before moving on
    for r in results:
        #print("\t Waiting:",r.wait())
        r.wait()
        prechi_tmp, numd_tmp = r.get()
        prechi = prechi + prechi_tmp
        numd   = numd + numd_tmp
        print("RGET:", prechi,numd)

    return prechi,numd
   

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

    parser.add_argument('--parse_only',dest='parse_only',action='store_true',default=False,help="parse the cl3 file and save the normal equations to a file (*.npz)") 
    parser.add_argument('--nadir_grid', dest='nadir_grid', default=0.1, type=float,help="Grid spacing to model NADIR corrections (default = 0.1 degrees)")
    parser.add_argument('--zenith_grid', dest='zen', default=0.5, type=float,help="Grid spacing to model Site corrections (default = 0.5 degrees)")
    parser.add_argument('-f', dest='resfile', default='',help="Consolidated one-way LC phase residuals")
    parser.add_argument('--conf', dest='config_file', default='',help="Get options from a configuration file")

    parser.add_argument('-p','--path',dest='path',help="Search for all CL3 files in the directory path") 

    parser.add_argument('-m','--model',dest='model',choices=['pwl','pwlSite','placeHolder'], help="Create a ESM for satellites only, or for satellites and sites")
    parser.add_argument('--save',dest='save_file',default=False, action='store_true',help="Save the Neq and Atwl matrices into numpy compressed format (npz)")
    parser.add_argument('-l','--load',dest='load_file',help="Load stored NEQ and AtWl matrices from a file")
    parser.add_argument('--lpath',dest='load_path',help="Path to search for .npz files")
   
    parser.add_argument('--cpu',dest='cpu',type=int,default=4,help="Maximum number of cpus to use")
    #===================================================================

    parser.add_argument("--syyyy",dest="syyyy",type=int,help="Start yyyy")
    parser.add_argument("--sdoy","--sddd",dest="sdoy",type=int,default=0,help="Start doy")
    parser.add_argument("--eyyyy",dest="eyyyy",type=int,help="End yyyyy")
    parser.add_argument("--edoy","--eddd",dest="edoy",type=int,default=365,help="End doy")

    #===================================================================
    # Plot options
    parser.add_argument('--plot',dest='plotNadir', default=False, action='store_true', help="Produce an elevation dependent plot of ESM phase residuals")
    parser.add_argument('--ps','--plot_save',dest='savePlots',default=False,action='store_true', help="Save the plots in png format")
    
    #===================================================================
    # Debug function, not needed
    args = parser.parse_args()

    #import matplotlib
    #if args.savePlots:
    #matplotlib.use('Agg')
    #    print("Only saving plots")

    import matplotlib.pyplot as plt

    # expand any home directory paths (~) to the full path, otherwise python won't find the file
    if args.resfile : args.resfile = os.path.expanduser(args.resfile)
    args.antex = os.path.expanduser(args.antex)
    args.svnavFile = os.path.expanduser(args.svnavFile)
    args.station_file = os.path.expanduser(args.station_file)
    
    svdat = []
    nadirData = {}
    cl3files = []
    npzfiles = []
    totalSites = 1
    totalSiteModels = 0
    siteIDList = []
    prechis = []
    numds = []

    prechi = 0
    numd = 0

    if args.model: 
        #===================================================================
        # get the antenna information from an antex file
        antennas = ant.parseANTEX(args.antex)

        if args.resfile :
            cl3files.append(args.resfile)
            siteIDList.append(os.path.basename(args.resfile)[0:4]+"_model_1")
        elif args.path:
            print("Checking {:s} for CL3 files".format(args.path))
            phsRGX = re.compile('.CL3$')
            for root, dirs, files in os.walk(args.path):
                path = root.split('/')
                for lfile in files:
                    if phsRGX.search(lfile):
                        print("Found:",args.path + "/" + lfile)
                        cl3files.append(args.path + "/"+ lfile)
            cl3files = np.sort(cl3files)
            totalSites = np.size(cl3files)

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

            # Read in the residual files and create the normal equations
            multiprocessing.freeze_support()
            prechi, numd = setUpTasks(cl3files,svs,args,params)
            print("Prechi",prechi,np.sqrt(prechi/numd))
            prechis.append(prechi)
            numds.append(numd)
            
            if args.parse_only:
                sys.exit(0)
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

            Neq = np.zeros((numParams,numParams))
            AtWb = np.zeros(numParams)

            #=====================================================================
            # Now read in all of the numpy compressed files
            #=====================================================================
            nctr = 0

            for f in range(0,np.size(params)):
                filename = os.path.basename(cl3files[f])
                params[f]['npzfile'] = params[f]['filename']+'.npz'
                nfile = params[f]['npzfile'] 

                npzfile = np.load(nfile)
                Neq_tmp  = npzfile['neq']
                AtWb_tmp = npzfile['atwb']

                #non_zero = 0
                ## Check how many elements have values..
                #for i in range(0,np.shape(Neq_tmp)[0]):
                #    criterion = (np.abs(Neq_tmp[:,i]) > 0.00001)
                #    non_zero = non_zero + np.size(np.array(np.where(criterion)))

                #Msize = np.size(Neq_tmp)
                #density = np.float(non_zero)/np.float(Msize)
                #print("Density of Neq:{:s} {:.3f} {:d} {:d}".format(filename,density,non_zero,Msize))

                # only need one copy of the svs array, they should be eactly the same
                if nctr == 0:
                    svs_tmp  = npzfile['svs']
                    svs = np.sort(svs_tmp)
                    del svs_tmp

                # Add the svn component to the Neq
                Neq[0:tSat-1,0:tSat-1] = Neq[0:tSat -1,0:tSat-1] + Neq_tmp[0:tSat-1,0:tSat-1]
                AtWb[0:tSat-1] = AtWb[0:tSat-1] + AtWb_tmp[0:tSat-1]

                #===================================
                # Loop over each model 
                #===================================
                for m in range(0,params[f]['numModels']) :
                    # Add in the station dependent models
                    start = tSat + nctr * numParamsPerSite 
                    end = tSat + (nctr+1) * numParamsPerSite

                    tmp_start = tSat + numParamsPerSite * m 
                    tmp_end   = tSat + numParamsPerSite * m + numParamsPerSite

                    AtWb[start:end] = AtWb[start:end] + AtWb_tmp[tSat:(tSat+numParamsPerSite)]
                    #
                    #   ------------------------------------------------
                    #  | SVN         | svn + site | svn + site2 | ....
                    #  | svn + site  | site       | 0           | ....
                    #  | svn + site2 | 0          | site2       | ....
                    #

                    # Add in the site block 
                    Neq[start:end,start:end] = Neq[start:end,start:end] + Neq_tmp[tmp_start:tmp_end,tmp_start:tmp_end]

                    # Adding in the correlation with the SVN and site
                    Neq[0:tSat-1,start:end] = Neq[0:tSat-1,start:end] + Neq_tmp[0:tSat-1,tmp_start:tmp_end]
                    Neq[start:end,0:tSat-1] = Neq[start:end,0:tSat-1] + Neq_tmp[tmp_start:tmp_end,0:tSat-1]
                    nctr += 1
                    totalSiteModels = totalSiteModels + 1
                    siteIDList.append(params[f]['site'])

            if args.save_file:
                np.savez_compressed('consolidated.npz',neq=Neq,atwb=AtWb,svs=svs)
            #=====================================================================
            # End of if not load_file or not load_path
            #=====================================================================
            print("FINISHED MP processing, now need to workout stacking:...\n") 

        if args.load_file or args.load_path:
            npzfiles = []
            # Number of Parameters
            numNADS = int(14.0/args.nadir_grid) + 1 
            PCOEstimates = 1
            numParamsPerSat = numNADS + PCOEstimates

            # Should do a quick loop through and check that all of the svs in each file
            # are of the same dimension
            numParamsPerSite = int(90./args.zen) + 1 

            if args.load_file:
                npzfiles.append(args.load_file)
            else:
                npyRGX = re.compile('.npz')
                for root, dirs, files in os.walk(args.load_path):
                    path = root.split('/')
                    for lfile in files:
                        if npyRGX.search(lfile):
                            print("Found:",args.load_path + "/" + lfile)
                            npzfiles.append(args.load_path + "/"+ lfile)

            npzfiles = np.sort(npzfiles)

            # model counter
            mctr = 0
            meta = []
            prechi = 0
            numd = 0
            #=====================================================================
            # first thing, work out how many models and parameters we will need to account for
            #=====================================================================
            for n in range(0,np.size(npzfiles)):
                info = {}
                filename = os.path.basename(npzfiles[n])
                info['basename'] = filename
                info['site'] = filename[0:4]

                npzfile = np.load(npzfiles[n])
                AtWb = npzfile['atwb']
                info['num_svs'] = np.size(npzfile['svs'])
                tSat = numParamsPerSat * np.size(npzfile['svs']) 
                info['numModels'] = int((np.size(AtWb) - tSat)/numParamsPerSite)
                mctr = mctr + info['numModels']
                meta.append(info)
                print(npzfile['prechi']) 
                prechis.append(npzfile['prechi'])
                numds.append(npzfile['numd'])

                for s in range(0,info['numModels']):
                    siteIDList.append(info['site']+"_model_"+str(s+1))
                del AtWb, npzfile

            totalSiteModels = mctr
            numSVS = meta[0]['num_svs']
            numSites = int(mctr) 
            tSite = numParamsPerSite * numSites
            numParams = tSat + tSite

            print("Total number of site models :",mctr, "Total number of paramters to solve for:",numParams)
            Neq = np.zeros((numParams,numParams))
            AtWb = np.zeros(numParams)

            mdlCtr = 0
            for n in range(0,np.size(npzfiles)):
                npzfile = np.load(npzfiles[n])
                Neq_tmp  = npzfile['neq']
                AtWb_tmp = npzfile['atwb']
                svs_tmp  = npzfile['svs']
                if n == 0:
                    svs = np.sort(svs_tmp)

                # Add the svn component to the Neq
                Neq[0:tSat-1,0:tSat-1] = Neq[0:tSat -1,0:tSat-1] + Neq_tmp[0:tSat-1,0:tSat-1]
                AtWb[0:tSat-1] = AtWb[0:tSat-1] + AtWb_tmp[0:tSat-1]

                for m in range(0,meta[n]['numModels']) :
                    # Add in the station dependent models
                    start = tSat + mdlCtr * numParamsPerSite 
                    end = tSat + (mdlCtr+1) * numParamsPerSite

                    tmp_start = tSat + numParamsPerSite * m 
                    tmp_end   = tSat + numParamsPerSite * m + numParamsPerSite

                    AtWb[start:end] = AtWb[start:end] + AtWb_tmp[tSat:(tSat+numParamsPerSite)]
                    Neq[start:end,start:end] = Neq[start:end,start:end] + Neq_tmp[tmp_start:tmp_end,tmp_start:tmp_end]

                    # Adding in the correlation with the SVN and site
                    Neq[0:tSat-1,start:end] = Neq[0:tSat-1,start:end] + Neq_tmp[0:tSat-1,tmp_start:tmp_end]
                    Neq[start:end,0:tSat-1] = Neq[start:end,0:tSat-1] + Neq_tmp[tmp_start:tmp_end,0:tSat-1]
                    mdlCtr = mdlCtr + 1

                non_zero = 0
                # Check how many elements have values..
                for i in range(0,np.shape(Neq_tmp)[0]):
                    criterion = (np.abs(Neq_tmp[:,i]) > 0.00001)
                    non_zero = non_zero + np.size(np.array(np.where(criterion)))
                Msize = np.size(Neq_tmp)
                density = np.float(non_zero)/np.float(Msize)
                print("Density of Neq:{:s} {:.3f} {:d} {:d}".format(npzfiles[n],density,non_zero,Msize))
            #sys.exit(0) 

        if args.save_file:
            np.savez_compressed('stacked.npz',neq=Neq,atwb=AtWb,svs=svs,prechi=np.sum(prechi),numd=np.sum(numd))
        
    if not args.save_file:
        #========================================================================
        # Adding Constraints to the satellite parameters,
        # keep the site model free ~ 10mm  0.01 => 1/sqrt(0.01) = 10 (mm)
        # Adding 1 mm constraint to satellites
        #========================================================================
        #sPCV_constraint = 0.1
        sPCV_constraint = 0.01
        sPCV_window = 1.0     # assume the PCV variation is correlated at this degree level
        #sPCV_window = 1.0     # assume the PCV variation is correlated at this degree level
        site_constraint = 10.0
        #site_window = 15.5
        site_window = 1.0

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

            # Add in the correlation constraints for the sites PCVs
            for s in range(0,numSites):
                for ind in range(0,numParamsPerSite-np.size(site_corr) ):
                    start = tSat + (s * numParamsPerSite) + ind
                    if ind > (numParamsPerSite - np.size(site_corr)):
                        end = start + (numParamsPerSite - ind) 
                    else:
                        end = start + np.size(site_corr)
                        
                    C[start,start:end] = site_corr[0:(end - start)] 
                    C[start:end,start] = site_corr[0:(end - start)] 

        C_inv = np.linalg.pinv(C)
        del C
        
        if args.plotNadir or args.savePlots:
            #============================================
            # Plot the sparsity of the matrix Neq before constraints are added
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.spy(Neq, precision=1e-3, marker='.', markersize=5)
            if args.savePlots:
                plt.savefig("NeqMatrix.png")
            plt.tight_layout()

        # Add the parameter constraints to the Neq
        Neq = np.add(Neq,C_inv)

        print("Now trying an inverse")
        #Cov = np.linalg.pinv(Neq)
        Cho = np.linalg.cholesky(Neq)
        Cho_inv = np.linalg.pinv(Cho)
        Cov = np.dot(Cho_inv.T,Cho_inv)

        print("Now computing the solution")
        Sol = np.dot(Cov,AtWb)
        print("The solution is :",np.shape(Sol))

        prechi = np.sum(prechis)
        numd = np.sum(numd)
        print("Prechi, numd",prechi,numd)
             
        postchi = prechi - np.dot(np.array(AtWb).T,np.array(Sol))
        print("STATS:",numd,np.sqrt(prechi/numd),np.sqrt(postchi/numd),np.sqrt((prechi-postchi)/numd))#,aic,bic)

    if args.plotNadir or args.savePlots:
        nad = np.linspace(0,14, int(14./args.nadir_grid)+1 )
        numParamsPerSat = int(14.0/args.nadir_grid) + 2 

        variances = np.diag(Neq)
        print("Variance:",np.shape(variances))

        #============================================
        # Plot the sparsity of the matrix Neq
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spy(Neq, precision=1e-3, marker='.', markersize=5)

        ctr = 0
        xlabels = []
        xticks = []
        for svn in svs:
            siz = numParamsPerSat * ctr 
            eiz = numParamsPerSat *ctr + numNADS 
            ctr = ctr + 1
            xlabels.append(svn)
            tick = int((eiz-siz)/2)+siz
            xticks.append(tick)

        
        for snum in range(0,totalSiteModels):
            siz = numParamsPerSat*numSVS + snum * numParamsPerSite 
            eiz = siz + numParamsPerSite 
            xlabels.append(siteIDList[snum])
            tick = int((eiz-siz)/2)+siz
            xticks.append(tick)

        print("XTICKS:",xticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels,rotation='vertical')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                       ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(4)

        plt.tight_layout()
        if args.savePlots:
            plt.savefig("NeqMatrix_with_constraints.eps")

        del Neq, AtWb

        #============================================
        # Plot the SVN stacked residuals/correction
        #============================================
        ctr = 0
        for svn in svs:
            #fig = plt.figure(figsize=(3.62, 2.76))
            fig = plt.figure()
            fig.canvas.set_window_title("SVN_"+svn+"_nadirCorrectionModel.png")
            ax = fig.add_subplot(212)
            ax1 = fig.add_subplot(211)

            siz = numParamsPerSat * ctr 
            eiz = numParamsPerSat *ctr + numNADS 
            
            #print("SVN:",svn,siz,eiz,numParamsPerSat,tSat)
            ax1.plot(nad,Sol[siz:eiz],'r-',linewidth=2)
            ax.errorbar(nad,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,fmt='o')

            #print(svn,Sol[siz:eiz],np.sqrt(variances[siz:eiz])/2.)
            ax.set_xlabel('Nadir Angle (degrees)',fontsize=8)
            ax.set_ylabel('Phase Residuals (mm)',fontsize=8)
            #ax.set_xlim([0, 14])

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                       ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(8)

            plt.tight_layout()

            for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                       ax1.get_xticklabels() + ax1.get_yticklabels()):
                item.set_fontsize(8)

            plt.tight_layout()
            if args.savePlots:
                plt.savefig("SVN_"+svn+"_nadirCorrectionModel.png")
            ctr += 1
                
            if ctr > 2:
                break

        #==================================================
        #fig = plt.figure(figsize=(3.62, 2.76))
        fig = plt.figure()
        fig.canvas.set_window_title("PCO_correction.png")
        #ax = fig.add_subplot(111)
        ax = fig.add_subplot(212)
        ax1 = fig.add_subplot(211)
        ctr = 0
        numSVS = np.size(svs)
        numNADS = int(14.0/args.nadir_grid) + 1 
        numParamsPerSat = numNADS + PCOEstimates
        print("Number of Params per Sat:",numParamsPerSat,"numNads",numNADS,"Sol",np.shape(Sol))
        for svn in svs:
            eiz = numParamsPerSat *ctr + numParamsPerSat -1 
            #print(ctr,"PCO:",eiz)
            ax1.plot(ctr,Sol[eiz],'k.',linewidth=2)
            #print(svn,Sol[eiz],np.sqrt(variances[eiz])/2.)
            #ax.subplot(212)
            ax.errorbar(ctr,Sol[eiz],yerr=np.sqrt(variances[eiz])/2.,fmt='o')
            ctr += 1

        ax.set_xlabel('SVN',fontsize=8)
        ax.set_ylabel('Adjustment to PCO (mm)',fontsize=8)
        #ax.set_xlim([0, 14])

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                   ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)
        plt.tight_layout()

        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                   ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(8)
        plt.tight_layout()

        if args.savePlots:
            plt.savefig("PCO_correction.png")

        #==================================================
        if args.model == 'pwlSite':
            ctr = 0
            numSVS = np.size(svs)
            numNADS = int(14.0/args.nadir_grid) + 1 
            numParamsPerSat = numNADS + PCOEstimates
            print("Number of Params per Sat:",numParamsPerSat,"numNads",numNADS,"Sol",np.shape(Sol),"TotalSites:",totalSiteModels)
            numParams = numSVS * (numParamsPerSat) + numParamsPerSite * totalSiteModels 
            for snum in range(0,totalSiteModels):
                #fig = plt.figure(figsize=(3.62, 2.76))
                fig = plt.figure()
                fig.canvas.set_window_title(siteIDList[snum]+"_elevation_model.png")
                ax = fig.add_subplot(212)
                ax1 = fig.add_subplot(211)
                siz = numParamsPerSat*numSVS + snum * numParamsPerSite 
                eiz = siz + numParamsPerSite 
                ele = np.linspace(0,90,numParamsPerSite)
                #print("Sol",np.shape(Sol),"siz  ",siz,eiz)
                ax1.plot(ele,Sol[siz:eiz],'k.',linewidth=2)
                ax.errorbar(ele,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,fmt='o')
                #print(svn,Sol[siz:eiz],np.sqrt(variances[siz:eiz])/2.)

                ax.set_xlabel('Elevation Angle',fontsize=8)
                ax.set_ylabel('Adjustment to PCO (mm)',fontsize=8)
                #ax.set_xlim([0, 14])

                for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                                ax1.get_xticklabels() + ax1.get_yticklabels()):
                    item.set_fontsize(8)
                plt.tight_layout()

                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(8)

                plt.tight_layout()
                if args.savePlots:
                    plt.savefig(siteIDList[snum]+"_elevation_model.png")
                
        if args.plotNadir:
            plt.show()

    print("FINISHED")
