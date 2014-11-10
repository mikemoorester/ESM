#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import calendar
import datetime as dt
import os
import re
import sys

import pickle

import antenna as ant
import residuals as res
import gpsTime as gt

import svnav

import Navigation as rnxN

import nadir as NADIR
import GamitStationFile as gsf
import GamitAprioriFile as gapr

def calcPostFitBySite(cl3file,svs,Sol,params,args,modelNum):
    """
    calcPostFitBySite()

    """
    # add one to make sure we have a linspace which includes 0.0 and 14.0
    # add another parameter for the zenith PCO estimate
    nadSpacing = args.nadir_grid
    numNADS = int(14.0/nadSpacing) + 1 
    PCOEstimates = 1
    numSVS = np.size(svs)
    numParamsPerSat = numNADS + PCOEstimates
    tSat = numParamsPerSat * numSVS

    zenSpacing = args.zen
    numParamsPerSite = int(90.0/zenSpacing) + 1
    tSite = numParamsPerSite*params['numModels']
    numParams = tSat + tSite 
   
    brdc_dir = args.brdc_dir

    postfit = 0.0
    postfit_sums = np.zeros(numParams)

    change = params['changes']

    site_residuals = res.parseConsolidatedNumpy(cl3file)

    for m in range(0,int(params['numModels'])):
        # start_yyyy and start_ddd should always be defind, however stop_dd may be absent
        # ie no changes have ocured since the last setup
        minVal_dt = gt.ydhms2dt(change['start_yyyy'][m],change['start_ddd'][m],0,0,0)

        if np.size(change['stop_ddd']) > m  :
            maxVal_dt = gt.ydhms2dt(change['stop_yyyy'][m],change['stop_ddd'][m],23,59,59)
            #print("Min:",minVal_dt,"Max:",maxVal_dt,m,np.size(change['stop_ddd']))
            criterion = ( ( site_residuals[:,0] >= calendar.timegm(minVal_dt.utctimetuple()) ) &
                    ( site_residuals[:,0] < calendar.timegm(maxVal_dt.utctimetuple()) ) )
        else:
            criterion = ( site_residuals[:,0] >= calendar.timegm(minVal_dt.utctimetuple()) ) 
            maxVal_dt = gt.unix2dt(site_residuals[-1,0])

        # get the residuals for this model time period
        mind = np.array(np.where(criterion))[0]
        model_residuals = site_residuals[mind,:]
        diff_dt = maxVal_dt - minVal_dt
        numDays = diff_dt.days + 1
        print("Have a total of",numDays,"days")

        # set up a lookup dictionary
        lookup_svs = {}
        lctr = 0
        for sv in svs:
            lookup_svs[str(sv)] = lctr
            lctr+=1

        site_geocentric_distance = np.linalg.norm(params['sitepos'])

        for d in range(0,numDays):
            minDTO = minVal_dt + dt.timedelta(days = d)
            maxDTO = minVal_dt + dt.timedelta(days = d+1)
            #print(d,"Stacking residuals on:",minDTO,maxDTO)
            criterion = ( ( model_residuals[:,0] >= calendar.timegm(minDTO.utctimetuple()) ) &
                          ( model_residuals[:,0] < calendar.timegm(maxDTO.utctimetuple()) ) )
            tind = np.array(np.where(criterion))[0]

            # if there are less than 300 obs, then skip to the next day
            if np.size(tind) < 300:
                continue

            #print("rejecting any residuals greater than 100mm",np.shape(site_residuals))
            tdata = res.reject_absVal(model_residuals[tind,:],100.)

            #print("rejecting any residuals greater than 5 sigma",np.shape(tdata))
            data = res.reject_outliers_elevation(tdata,5,0.5)
            del tdata

            # parse the broadcast navigation file for this day to get an accurate
            # nadir angle
            yy = minDTO.strftime("%y") 
            doy = minDTO.strftime("%j") 
            navfile = brdc_dir + 'brdc'+ doy +'0.'+ yy +'n'
            #print("Will read in the broadcast navigation file:",navfile)
            nav = rnxN.parseFile(navfile)

            # Get the total number of observations for this site
            numd = np.shape(data)[0]
            #print("Have:",numd,"observations")
            for i in range(0,numd):
                # work out the svn number
                svndto =  gt.unix2dt(data[i,0])
                svn = svnav.findSV_DTO(svdat,data[i,4],svndto)
                svn_search = 'G{:03d}'.format(svn) 
                ctr = lookup_svs[str(svn_search)]

                # get the satellite position
                svnpos = rnxN.satpos(data[i,4],svndto,nav)
                satnorm = np.linalg.norm(svnpos[0])

                # work out the nadir angle
                nadir = NADIR.calcNadirAngle(90.-data[i,2],site_geocentric_distance,satnorm)

                niz = int(np.floor(nadir/nadSpacing))
                iz = int((numParamsPerSat * ctr) + niz)
                pco_iz = numParamsPerSat * (ctr+1) - 1 

                nsiz = int(np.floor(data[i,2]/zenSpacing))
                siz = int( tSat +  m*numParamsPerSite + nsiz)

                sol_site = int( tSat +  (m+modelNum)*numParamsPerSite + nsiz)
                # check that the indices are not overlapping
                if iz+1 >= pco_iz:
                    continue
                
                # Nadir partials..
                Apart_1 = (1.-(nadir-niz*nadSpacing)/nadSpacing)
                Apart_2 = (nadir-niz*nadSpacing)/nadSpacing
                # PCO partial ...
                Apart_3 = -np.sin(np.radians(nadir)) 
                # Site partials
                Apart_4 = (1.-(data[i,2]-nsiz*zenSpacing)/zenSpacing)
                Apart_5 = (data[i,2]-nsiz*zenSpacing)/zenSpacing

                postfit = postfit + ((data[i,3] - Apart_1 * Sol[iz])/1000.)**2
                postfit_sums[iz] = postfit_sums[iz] + ((data[i,3] - Apart_1 * Sol[iz])/1000.)**2

                postfit = postfit + ((data[i,3] - Apart_2 * Sol[iz+1])/1000.)**2
                postfit_sums[iz+1] = postfit_sums[iz+1] + ((data[i,3] - Apart_2 * Sol[iz+1])/1000.)**2

                postfit = postfit + ((data[i,3] - Apart_3 * Sol[pco_iz])/1000.)**2
                postfit_sums[pco_iz] = postfit_sums[pco_iz] + ((data[i,3] - Apart_3 * Sol[pco_iz])/1000.)**2

                postfit = postfit + ((data[i,3] - Apart_4 * Sol[siz])/1000.)**2
                postfit_sums[siz] = postfit_sums[siz] + ((data[i,3] - Apart_4 * Sol[sol_site])/1000.)**2

                postfit = postfit + ((data[i,3] - Apart_5 * Sol[siz+1])/1000.)**2
                postfit_sums[siz+1] = postfit_sums[siz+1] + ((data[i,3] - Apart_5 * Sol[sol_site+1])/1000.)**2

    return postfit, postfit_sums, params, modelNum

def prepareSites(cl3files,dt_start,dt_end,args):
    #=====================================================================
    # Work out how many station models need to be created 
    #=====================================================================
    numModels = 0
    params = []

    for f in range(0,np.size(cl3files)):
        filename    = os.path.basename(cl3files[f])
        siteID      = filename[0:4]
        sdata       = gsf.parseSite(args.station_file,siteID.upper())
        changes     = gsf.determineESMChanges(dt_start,dt_stop,sdata)
        sitepos     = gapr.getStationPos(args.apr_file,siteID)
        numModels   = numModels + np.size(changes['ind']) + 1
        info = {}
        info['filename']  = cl3files[f]
        info['basename']  = filename
        info['site']      = siteID
        info['numModels'] = np.size(changes['ind']) + 1 
        info['changes']   = changes
        info['sitepos']   = sitepos
        params.append(info)

    return params, numModels
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
    parser.add_argument('--brdc',dest='brdc_dir',default="~/gg/brdc/",help="Location of broadcast navigation files")
    parser.add_argument('--apr',dest='apr_file',default="~/gg/tables/itrf08_comb.apr", help="Location of Apriori File containing the stations position")

    parser.add_argument('-f', dest='resfile', default='',help="Consolidated one-way LC phase residuals")

    parser.add_argument('--sf1',dest='solutionfile1',help="Pickle Solution file")
    parser.add_argument('--sf2',dest='solutionfile2',help="Numpy Solution file")

    parser.add_argument('--sum',dest='sum',help="Sum up postfit residual files in a directory")
    
    #===================================================================
    # Output options
    #===================================================================
    
    parser.add_argument('--save',dest='save_file',default=False, action='store_true',help="Save the Neq and Atwl matrices into numpy compressed format (npz)")
    parser.add_argument('--save_solution','--ss',dest='solution',default='solution.pkl',help="Save the Solution vector and meta data as a pickle object, needs save_file flag to be selected")#,META="Pickle filename")

    #===================================================================
    # Processing options
    #===================================================================
    parser.add_argument('--nadir_grid', dest='nadir_grid', default=0.1, type=float,help="Grid spacing to model NADIR corrections (default = 0.1 degrees)")
    parser.add_argument('--zenith_grid', dest='zen', default=0.5, type=float,help="Grid spacing to model Site corrections (default = 0.5 degrees)")
    parser.add_argument('-m','--model',dest='model',choices=['pwl','pwlSite','pwlSiteDaily'], help="Create a ESM for satellites only, or for satellites and sites")
    #===================================================================
    # Time period to check for satellite parameters
    parser.add_argument("--syyyy",dest="syyyy",type=int,help="Start yyyy")
    parser.add_argument("--sdoy","--sddd",dest="sdoy",type=int,default=1,help="Start doy")
    parser.add_argument("--eyyyy",dest="eyyyy",type=int,help="End yyyyy")
    parser.add_argument("--edoy","--eddd",dest="edoy",type=int,default=365,help="End doy")

    #===================================================================
    # Debug function, not needed
    args = parser.parse_args()
    
    if args.sum:
        ctr = 0
        pftRGX = re.compile('.pft.npz')
        for root, dirs, files in os.walk(args.sum):
            path = root.split('/')
            for lfile in files:
                if pftRGX.search(lfile):
                    npzfile = np.load(lfile)
                    if ctr == 0:
                        postfit_sums = npzfile['postfitsum']
                        postfit = npzfile['postfit'][0]
                    else:
                        postfit_sums = np.add(postfit_sums,npzfile['postfitsum'])
                        postfit += npzfile['postfit'][0]
                    ctr += 1
        print("postfit is",postfit)    
        sys.exit(0)                    
                    
    # expand any home directory paths (~) to the full path, otherwise python won't find the file
    if args.resfile : args.resfile = os.path.expanduser(args.resfile)
    args.antex        = os.path.expanduser(args.antex)
    args.svnavFile    = os.path.expanduser(args.svnavFile)
    args.station_file = os.path.expanduser(args.station_file)
    args.brdc_dir     = os.path.expanduser(args.brdc_dir) 
    args.apr_file     = os.path.expanduser(args.apr_file) 

    svdat = []
    nadirData = {}
    cl3files = []
    npzfiles = []
    totalSites = 1
    totalSiteModels = 0
    siteIDList = []
    prechis = []
    numds = []
    params = []
    numParams = 0
    prechi = 0
    numd = 0

    cl3files.append(args.resfile)
    siteIDSRCH = os.path.basename(args.resfile)[0:4]#+"_model_1"
    
    #==========================================================================
    # Read in the solution
    #==========================================================================
    npzfile = np.load(args.solutionfile2)
    Sol  = npzfile['sol']
    Cov  = npzfile['cov']
    nadir_freq = npzfile['nadirfreq']
    prefit_sums = npzfile['prefitsums'] 

    # Now read the pickle file
    with open(args.solutionfile1,'rb') as pklID:
        meta = pickle.load(pklID)
    pklID.close()

    args.model          = meta['model'] 
    args.nadir_grid     = meta['nadir_grid'] 
    args.antex          = meta['antex_file'] 
    args.svnavFile      = meta['svnav'] 
    args.station_file   = meta['station_info'] 
    args.zen            = meta['zenith_grid'] 

    svs         = meta['svs'] 
    numSites    = meta['numSiteModels'] 
    siteIDList  = meta['siteIDList'] 
    prechi      = meta['prechi']   #= np.sqrt(prechi/numd)
    postchi     = meta['postchi'] # = np.sqrt(postchi/numd)
    numd        = meta['numd']
    chi_inc     = meta['chi_inc'] # = np.sqrt((prechi-postchi)/numd)
  
    prefit      = meta['prefit']
    
    #===================================================================
    # Work out the time scale of observations, and number of parameters
    # that will be solved for. 
    #===================================================================
    print(args.syyyy, args.sdoy, args.eyyyy, args.edoy)
    dt_start = dt.datetime(int(args.syyyy),01,01) + dt.timedelta(days=int(args.sdoy))
    dt_stop  = dt.datetime(int(args.eyyyy),01,01) + dt.timedelta(days=int(args.edoy))

    # already have the parameters defined if we are loading in the npzfiles..
    params, numModels = prepareSites(cl3files,dt_start, dt_stop, args)

    antennas = ant.parseANTEX(args.antex)
    svdat    = svnav.parseSVNAV(args.svnavFile)
    svs      = ant.satSearch(antennas,dt_start,dt_stop)
    svs      = np.sort(np.unique(svs))

    # Number of Parameters
    numNADS             = int(14.0/args.nadir_grid) + 1 
    PCOEstimates        = 1
    numSVS              = np.size(svs)
    numParamsPerSat     = numNADS + PCOEstimates
    tSat                = numParamsPerSat * numSVS    
    numParamsPerSite    = int(90.0/args.zen)+1
    numParams           = np.size(Sol)
    
    print(siteIDList)
    IDX = siteIDList.index(siteIDSRCH)
    
    import collections
    counter = collections.Counter(siteIDList)
    print(counter)
    # Counter({1: 4, 2: 4, 3: 2, 5: 2, 4: 1})
    models = counter[siteIDSRCH]
    mdlCtr = IDX + models -1
   
    print("Model Counter",mdlCtr)
    # get ready to start caclulation
    postfit = 0.0
    postfit_sums = np.zeros(numParams)

    postfit_tmp, postfit_sums_tmp, info, mdlCtr = calcPostFitBySite(cl3files[0],svs,Sol,params[0],args,mdlCtr)
   
    postfit = postfit + postfit_tmp
    postfit_sums[0:tSat] = postfit_sums[0:tSat] + postfit_sums_tmp[0:tSat]

    
    ctr = 0
    for m in range(mdlCtr,(info['numModels']+mdlCtr)) :
        # Add in the station dependent models
        start = tSat + m * numParamsPerSite  
        end   = tSat + (m+1) * numParamsPerSite 

        tmp_start = tSat + numParamsPerSite * ctr 
        tmp_end   = tSat + numParamsPerSite * (ctr+1) # + numParamsPerSite 
        postfit_sums[start:end] = postfit_sums[start:end] + postfit_sums_tmp[tmp_start:tmp_end]

        ctr += 1    
        
    print("Prefit, Postfit, Postfit/Prefit",prefit,postfit,postfit/prefit)
    prefit_svs = np.sum(prefit_sums[0:tSat])
    postfit_svs = np.sum(postfit_sums[0:tSat])
   
    print("SVS Prefit, Postfit, Postfit/Prefit",prefit_svs,postfit_svs,postfit_svs/prefit_svs)
    postfitA = [postfit]
    np.savez_compressed(siteIDSRCH+".pft", postfit=postfitA, postfitsum=postfit_sums)
    #=======================================================================================================
    #
    #       Save the solution to a pickle data structure
    #
    #=======================================================================================================
    if args.save_file:
        with open(args.solution,'wb') as pklID:
            meta = {}
            meta['model'] = args.model
            meta['nadir_grid'] = args.nadir_grid
            meta['antex_file'] = args.antex
            meta['svnav'] = args.svnavFile
            meta['station_info'] = args.station_file
            meta['zenith_grid'] = args.zen
            meta['syyyy'] = args.syyyy
            meta['sddd']  = args.sdoy
            meta['eyyyy'] = args.eyyyy
            meta['eddd']  = args.edoy
            if args.stacked_file:
                meta['datafiles'] = args.stacked_file 
            else:
                meta['datafiles'] = npzfiles
            meta['svs'] = svs
            meta['numSiteModels'] = numSites 
            meta['siteIDList']  = siteIDList
            meta['prechi']   = np.sqrt(prechi/numd)
            meta['postchi']  = np.sqrt(postchi/numd)
            meta['numd']     = numd
            meta['chi_inc']  = np.sqrt((prechi-postchi)/numd)
            meta['apply_constraints'] = args.apply_constraints
            if args.apply_constraints:
                meta['constraint_SATPCV']  = args.constraint_SATPCV # 0.5 
                meta['constraint_SATPCO']  = args.constraint_SATPCO # 1.5
                meta['constraint_SATWIN']  = args.constraint_SATWIN # 0.5
                meta['constraint_SITEPCV'] = args.constraint_SITEPCV #10.0
                meta['constraint_SITEWIN'] = args.constraint_SITEWIN #1.5
            meta['saved_file'] = args.solution + ".sol"
            meta['prefit'] = prefit 
            if args.postfit:
                meta['postfit'] = postfit
            pickle.dump(meta,pklID,2)
            pklID.close()            

            #np.savez_compressed(args.solution+".sol",sol=Sol,cov=Cov,nadirfreq=nadir_freq)
            if args.postfit:
                np.savez_compressed(args.solution+".sol",sol=Sol,cov=Cov,nadirfreq=nadir_freq,
                                prefitsums=prefit_sums,postfitsum=postfit_sums)
            else:
                np.savez_compressed(args.solution+".sol",sol=Sol,cov=Cov,nadirfreq=nadir_freq,
                                prefitsums=prefit_sums)

   
    print("FINISHED")
