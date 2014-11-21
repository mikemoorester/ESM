#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import re
import calendar
import os, sys
import datetime as dt

import pickle
import multiprocessing

import antenna as ant
import residuals as res
import gpsTime as gt
import GamitStationFile as gsf
import GamitAprioriFile as gapr
import svnav
import broadcastNavigation as brdc
import Navigation as rnxN

import scipy as sp

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

def calcNadirAngle(zen,R=6378.0,r=26378.0):
    """
        Calculate the NADIR angle based on the station's elevation angle

        nadiar_angle = calNadirAngle(elevation,R,r)

        zen = zenith angle of satellite being observed
        R   = geocentric disatnce of station (default = 6378.0)
        r   = geocentric distance of satellite (default = 26378.0)

    """
    #nadeg = np.arcsin(6378.0/26378.0 * np.cos(ele/180.*np.pi)) * 180./np.pi
    #nadeg = np.degrees(np.arcsin(R/r * np.sin(np.radians(90.-zen)))) # * 180./np.pi
    nadeg = np.degrees(np.arcsin(R/r * np.sin(np.radians(zen)))) # * 180./np.pi
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
    #data = res.reject_outliers_elevation(tdata,5,0.5)
    data = tdata
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

        a,b = res.gamitWeight(data)
        print("GAMIT:",a,b)
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

            #w = np.sin(np.radians(data[i,2]))
            w = a**2 + b**2/np.sin(np.radians(data[i,2]))**2
            iz = int(numParamsPerSat * ctr + niz)
            pco_iz = int(numParamsPerSat *ctr + numNADS )

            #print("Indices m,iz,pco_iz,siz:",m,iz,pco_iz,siz,i,numd)
            # Nadir partials..
            #Apart_1 = (1.-(nadir-niz*nadSpacing)/nadSpacing)
            #Apart_2 = (nadir-niz*nadSpacing)/nadSpacing
            # PCO partial ...
            #Apart_3 = 1./np.sin(np.radians(nadir)) 
            # Site partials
            #Apart_4 = (1.-(data[i,2]-nsiz*zenSpacing)/zenSpacing)
            #Apart_5 = (data[i,2]-nsiz*zenSpacing)/zenSpacing
            #print("Finished forming Design matrix")
            Apart_1 = - np.sin(nadir)
            Apart_3 = 1. -np.sin(nadir)
            Apart_4 = 1.
            #print("Starting AtWb",np.shape(AtWb),iz,pco_iz,siz)
            AtWb[iz]     = AtWb[iz]     + Apart_1 * data[i,3] * 1./w**2
            #AtWb[iz+1]   = AtWb[iz+1]   + Apart_2 * data[i,3] * 1./w**2
            AtWb[pco_iz] = AtWb[pco_iz] + Apart_3 * data[i,3] * 1./w**2
            AtWb[siz]    = AtWb[siz]    + Apart_4 * data[i,3] * 1./w**2
            #AtWb[siz+1]  = AtWb[siz+1]  + Apart_5 * data[i,3] * 1./w**2
            #print("Finished forming b vector")

            Neq[iz,iz]     = Neq[iz,iz]     + Apart_1 * Apart_1 * 1./w**2
            #Neq[iz,iz+1]   = Neq[iz,iz+1]   + Apart_1 * Apart_2 * 1./w**2
            Neq[iz,pco_iz] = Neq[iz,pco_iz] + Apart_1 * Apart_3 * 1./w**2
            Neq[iz,siz]    = Neq[iz,siz]    + Apart_1 * Apart_4 * 1./w**2
            #Neq[iz,siz+1]  = Neq[iz,siz+1]  + Apart_1 * Apart_5 * 1./w**2

            #Neq[iz+1,iz]     = Neq[iz+1,iz]     + Apart_2 * Apart_1 * 1./w**2
            #Neq[iz+1,iz+1]   = Neq[iz+1,iz+1]   + Apart_2 * Apart_2 * 1./w**2
            #Neq[iz+1,pco_iz] = Neq[iz+1,pco_iz] + Apart_2 * Apart_3 * 1./w**2
            #Neq[iz+1,siz]    = Neq[iz+1,siz]    + Apart_2 * Apart_4 * 1./w**2
            #Neq[iz+1,siz+1]  = Neq[iz+1,siz+1]  + Apart_2 * Apart_5 * 1./w**2
            #print("Finished NEQ Nadir estimates")
            
            Neq[pco_iz,iz]     = Neq[pco_iz,iz]     + Apart_3 * Apart_1 * 1./w**2
            #Neq[pco_iz,iz+1]   = Neq[pco_iz,iz+1]   + Apart_3 * Apart_2 * 1./w**2
            Neq[pco_iz,pco_iz] = Neq[pco_iz,pco_iz] + Apart_3 * Apart_3 * 1./w**2
            Neq[pco_iz,siz]    = Neq[pco_iz,siz]    + Apart_3 * Apart_4 * 1./w**2
            #Neq[pco_iz,siz+1]  = Neq[pco_iz,siz+1]  + Apart_3 * Apart_5 * 1./w**2
            #print("Finished NEQ PCO estimates")

            Neq[siz,iz]     = Neq[siz,iz]     + Apart_4 * Apart_1 * 1./w**2
            #Neq[siz,iz+1]   = Neq[siz,iz+1]   + Apart_4 * Apart_2 * 1./w**2
            Neq[siz,pco_iz] = Neq[siz,pco_iz] + Apart_4 * Apart_3 * 1./w**2
            Neq[siz,siz]    = Neq[siz,siz]    + Apart_4 * Apart_4 * 1./w**2
            #Neq[siz,siz+1]  = Neq[siz,siz+1]  + Apart_4 * Apart_5 * 1./w**2

            #Neq[siz+1,iz]     = Neq[siz+1,iz]     + Apart_5 * Apart_1 * 1./w**2
            #Neq[siz+1,iz+1]   = Neq[siz+1,iz+1]   + Apart_5 * Apart_2 * 1./w**2
            #Neq[siz+1,pco_iz] = Neq[siz+1,pco_iz] + Apart_5 * Apart_3 * 1./w**2
            #Neq[siz+1,siz]    = Neq[siz+1,siz]    + Apart_5 * Apart_4 * 1./w**2
            #Neq[siz+1,siz+1]  = Neq[siz+1,siz+1]  + Apart_5 * Apart_5 * 1./w**2
            #print("Finished NEQ Site estimates")

    prechi = np.dot(data[:,3].T,data[:,3])
    print("Normal finish of pwlNadirSite",prechi,numd)
    return Neq, AtWb, prechi, numd

def pwlNadirSiteDailyStack(site_residuals, svs, params, nadSpacing=0.1,zenSpacing=0.5,brdc_dir="./"):
    """
    Create a model for the satellites and sites at the same time.
    PWL piece-wise-linear interpolation fit of phase residuals
    -construct a PWL fit for each azimuth bin, and then paste them all together to get 
     the full model
    -inversion is done within each bin

    cdata -> compressed data
    """
    prechi = 0
    NUMD   = 0
    # add one to make sure we have a linspace which includes 0.0 and 14.0
    # add another parameter for the zenith PCO estimate
    numNADS = int(14.0/nadSpacing) + 1 
    PCOEstimates = 1
    numSVS = np.size(svs)
    numParamsPerSat = numNADS + PCOEstimates
    tSat = numParamsPerSat * numSVS

    numParamsPerSite = int(90.0/zenSpacing) + 1
    tSite = numParamsPerSite*params['numModels']
    numParams = tSat + tSite 

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
    print("Changes for site",params['site'],change)
    # keep track of how may observations are in each bin
    NadirFreq = np.zeros((numSVS,numNADS))

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
            #print("finished outlier detection",np.shape(data))
            del tdata

            # determine the elevation dependent weighting
            a,b = res.gamitWeight(data)
            #print("Gamit Weighting:",minDTO,a,b)

            # parse the broadcast navigation file for this day to get an accurate
            # nadir angle
            year = minDTO.strftime("%Y") 
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
                #print("Looking for:",svn_search,lookup_svs)
                ctr = lookup_svs[str(svn_search)]
                #print("Position is CTR:",ctr,data[i,4])
                try:
                    # get the satellite position
                    svnpos = rnxN.satpos(data[i,4],svndto,nav)
                    #print("SVNPOS:",svnpos[0])
                    satnorm = np.linalg.norm(svnpos[0])
                    #print("NORM:",np.linalg.norm(svnpos[0]))
                except:
                    print("Error calculation satelite position for",svndto,data[i,:])
                    continue

                # work out the nadir angle
                #oldnadir = calcNadirAngle(data[i,2])
                nadir = calcNadirAngle(data[i,2],site_geocentric_distance,satnorm)
                #print("Ele {:.2f} Old: {:.2f} New:{:.2f}".format(data[i,2],oldnadir,nadir))
                #print("Ele {:.2f} New:{:.2f}".format(data[i,2],nadir))
                w = a**2 + b**2/np.sin(np.radians(90.-data[i,2]))**2
                w = 1./w

                niz = int(np.floor(nadir/nadSpacing))
                iz = int((numParamsPerSat * ctr) + niz)
                pco_iz = numParamsPerSat * (ctr+1) - 1 

                nsiz = int(np.floor(data[i,2]/zenSpacing))
                siz = int( tSat +  m*numParamsPerSite + nsiz)

                # check that the indices are not overlapping
                if iz+1 >= pco_iz or iz >= pco_iz:
                    #print("WARNING in indices iz+1 = pco_iz skipping obs",nadir,iz,pco_iz)
                    continue

                NadirFreq[ctr,niz] = NadirFreq[ctr,niz] +1
                # Nadir partials..
                Apart_1 = (1.-(nadir-niz*nadSpacing)/nadSpacing)
                Apart_2 = (nadir-niz*nadSpacing)/nadSpacing
                #
                # PCO partial ...
                #Apart_3 = -np.sin(np.radians(nadir)) 
                Apart_3 = np.cos(np.radians(nadir)) 

                # Site partials
                Apart_4 = (1.-(data[i,2]-nsiz*zenSpacing)/zenSpacing)
                Apart_5 = (data[i,2]-nsiz*zenSpacing)/zenSpacing
                #print("Finished forming Design matrix")
                #
                # R = SITE_PCV_ERR + SAT_PCV_ERR + SAT_PCO_ERR * cos(nadir)
                #
                # dR/dSITE_PCV_ERR = 1
                # dR/dSAT_PCV_ERR  = 1 
                # dR/dSAT_PCO_ERR  = cos(nadir)
                #
                # nice partial derivative tool:
                #  http://www.symbolab.com/solver/partial-derivative-calculator
                #

                #print("Starting AtWb",np.shape(AtWb),iz,pco_iz,siz)
                AtWb[iz]     = AtWb[iz]     + Apart_1 * data[i,3] * w
                AtWb[iz+1]   = AtWb[iz+1]   + Apart_2 * data[i,3] * w
                AtWb[pco_iz] = AtWb[pco_iz] + Apart_3 * data[i,3] * w
                AtWb[siz]    = AtWb[siz]    + Apart_4 * data[i,3] * w
                AtWb[siz+1]  = AtWb[siz+1]  + Apart_5 * data[i,3] * w
                #print("Finished forming b vector")

                Neq[iz,iz]     = Neq[iz,iz]     + (Apart_1 * Apart_1 * w)
                Neq[iz,iz+1]   = Neq[iz,iz+1]   + (Apart_1 * Apart_2 * w)
                Neq[iz,pco_iz] = Neq[iz,pco_iz] + (Apart_1 * Apart_3 * w)
                Neq[iz,siz]    = Neq[iz,siz]    + (Apart_1 * Apart_4 * w)
                Neq[iz,siz+1]  = Neq[iz,siz+1]  + (Apart_1 * Apart_5 * w)

                Neq[iz+1,iz]     = Neq[iz+1,iz]     + (Apart_2 * Apart_1 * w)
                Neq[iz+1,iz+1]   = Neq[iz+1,iz+1]   + (Apart_2 * Apart_2 * w)
                Neq[iz+1,pco_iz] = Neq[iz+1,pco_iz] + (Apart_2 * Apart_3 * w)
                Neq[iz+1,siz]    = Neq[iz+1,siz]    + (Apart_2 * Apart_4 * w)
                Neq[iz+1,siz+1]  = Neq[iz+1,siz+1]  + (Apart_2 * Apart_5 * w)
                #print("Finished NEQ Nadir estimates")
            
                Neq[pco_iz,iz]     = Neq[pco_iz,iz]     + (Apart_3 * Apart_1 * w)
                Neq[pco_iz,iz+1]   = Neq[pco_iz,iz+1]   + (Apart_3 * Apart_2 * w)
                Neq[pco_iz,pco_iz] = Neq[pco_iz,pco_iz] + (Apart_3 * Apart_3 * w)
                Neq[pco_iz,siz]    = Neq[pco_iz,siz]    + (Apart_3 * Apart_4 * w)
                Neq[pco_iz,siz+1]  = Neq[pco_iz,siz+1]  + (Apart_3 * Apart_5 * w)
                #print("Finished NEQ PCO estimates")

                Neq[siz,iz]     = Neq[siz,iz]     + (Apart_4 * Apart_1 * w)
                Neq[siz,iz+1]   = Neq[siz,iz+1]   + (Apart_4 * Apart_2 * w)
                Neq[siz,pco_iz] = Neq[siz,pco_iz] + (Apart_4 * Apart_3 * w)
                Neq[siz,siz]    = Neq[siz,siz]    + (Apart_4 * Apart_4 * w)
                Neq[siz,siz+1]  = Neq[siz,siz+1]  + (Apart_4 * Apart_5 * w)

                Neq[siz+1,iz]     = Neq[siz+1,iz]     + (Apart_5 * Apart_1 * w)
                Neq[siz+1,iz+1]   = Neq[siz+1,iz+1]   + (Apart_5 * Apart_2 * w)
                Neq[siz+1,pco_iz] = Neq[siz+1,pco_iz] + (Apart_5 * Apart_3 * w)
                Neq[siz+1,siz]    = Neq[siz+1,siz]    + (Apart_5 * Apart_4 * w)
                Neq[siz+1,siz+1]  = Neq[siz+1,siz+1]  + (Apart_5 * Apart_5 * w)
                #print("Finished NEQ Site estimates")
                
                if siz == pco_iz:
                    print("ERROR in indices siz = pco_iz")

        prechi = prechi + np.dot(data[:,3].T,data[:,3])
        NUMD = NUMD + numd
    print("Normal finish of pwlNadirSiteDailyStack",prechi,NUMD)
    #return Neq, AtWb, prechi, NUMD, NadirFreq, prefit, prefit_sums
    return Neq, AtWb, prechi, NUMD, NadirFreq

def neqBySite(params,svs,args):
    print("\t Reading in file:",params['filename'])
    site_residuals = res.parseConsolidatedNumpy(params['filename'])
    if args.model == 'pwl':
        Neq_tmp,AtWb_tmp = pwl(site_residuals,svs,args.nadir_grid)
    elif args.model == 'pwlSite':
        Neq_tmp,AtWb_tmp,prechi_tmp,numd_tmp = pwlNadirSite(site_residuals,svs,params,args.nadir_grid,0.5)
    elif args.model == 'pwlSiteDaily':
        print("Attempting a stack on each day")
        #Neq_tmp,AtWb_tmp,prechi_tmp,numd_tmp, nadir_freq, prefit_sum, prefit_sums = pwlNadirSiteDailyStack(site_residuals,svs,params,args.nadir_grid,0.5,args.brdc_dir)
        Neq_tmp,AtWb_tmp,prechi_tmp,numd_tmp, nadir_freq = pwlNadirSiteDailyStack(site_residuals,svs,params,args.nadir_grid,0.5,args.brdc_dir)

    print("Returned Neq, AtWb:",np.shape(Neq_tmp),np.shape(AtWb_tmp),prechi_tmp,numd_tmp,np.shape(nadir_freq))
            
    sf = params['filename']+".npz"
    prechis = [prechi_tmp]
    numds = [numd_tmp]
    
    np.savez_compressed(sf,neq=Neq_tmp,atwb=AtWb_tmp,svs=svs,prechi=prechis,numd=numds,nadirfreq=nadir_freq)

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
   
#def compressNeq(Neq,AtWb,svs,numParamsPerSat,nadir_freq,prefit_sums):
def compressNeq(Neq,AtWb,svs,numParamsPerSat,nadir_freq):
    # check for any rows/ columns without any observations, if they are empty remove the parameters
    satCtr = 0
    starts = []
    ends   = []
    remove = []

    for sv in svs:
        non_zero = 0
        start = satCtr * numParamsPerSat 
        end   = (satCtr+1) * numParamsPerSat
        ## Check how many elements have values..
        for d in range(start,end):
            criterion = (np.abs(Neq[d,:]) > 0.00001)
            non_zero = non_zero + np.size(np.array(np.where(criterion))[0])
            #print(sv,"Non_zero:",non_zero,d)

        if non_zero < 1 :
            print("No observations for:",sv)
            starts.append(start)
            ends.append(end)
            remove.append(satCtr)

        satCtr = satCtr + 1

    # remove the satellites without any observations from the svs array
    # Need to do it in reverse order
    remove = np.array(remove[::-1])
    for i in remove:
        svs = np.delete(svs,i)
        nadir_freq = np.delete(nadir_freq,i,0)

    ends = np.array(ends[::-1])
    starts = np.array(starts[::-1])

    print("BEFORE Neq shape:",np.shape(Neq),np.shape(nadir_freq))
    # Axis 0 ---> (row)
    # Axis 1 |    (column)
    #        |
    #        v
    #
    for i in range(0,np.size(ends)):
        del_ind = range(starts[i],ends[i])
        Neq = np.delete(Neq,del_ind,0)
        Neq = np.delete(Neq,del_ind,1)
        AtWb = np.delete(AtWb,del_ind)
        #prefit_sums = np.delete(prefit_sums,del_ind)

    print("AFTER Neq shape:",np.shape(Neq),np.shape(nadir_freq))
    #return Neq, AtWb, svs, nadir_freq, prefit_sums
    return Neq, AtWb, svs, nadir_freq

def calcPostFitBySite(cl3file,svs,Sol,params,svdat,args,modelNum):
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
  
    res_all = []

    brdc_dir = args.brdc_dir

    postfit = 0.0
    postfit_sums = np.zeros(numParams)
    postfit_res = np.zeros(numParams)

    prefit = 0.0
    prefit_sums = np.zeros(numParams)
    prefit_res = np.zeros(numParams)
    
    prefit_rms = 0.0
    postfit_rms = 0.0
    mod_rms = 0.0
    numObs = 0
    numObs_sums = np.zeros(numParams)

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
        print("Have a total of",numDays,"days in",cl3file)

        # set up a lookup dictionary
        lookup_svs = {}
        lctr = 0
        for sv in svs:
            lookup_svs[str(sv)] = lctr
            lctr+=1
        #print("The lookup_svs is:",lookup_svs)

        # get the distance from the centre of earth for nadir calculation
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

            #print("rejecting any residuals greater than 3 sigma",np.shape(tdata))
            data = res.reject_outliers_elevation(tdata,3,0.5)
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

            for i in range(0,numd):
                # work out the svn number
                svndto =  gt.unix2dt(data[i,0])
                svn = svnav.findSV_DTO(svdat,data[i,4],svndto)
                svn_search = 'G{:03d}'.format(svn) 
                ctr = lookup_svs[str(svn_search)]

                # get the satellite position
                try:
                    svnpos = rnxN.satpos(data[i,4],svndto,nav)
                    satnorm = np.linalg.norm(svnpos[0])
                except:
                    print("Error caclulating sat pos for:",svndto,data[i,:])
                    continue

                # work out the nadir angle
                nadir = calcNadirAngle(data[i,2],site_geocentric_distance,satnorm)

                niz = int(np.floor(nadir/nadSpacing))
                iz = int((numParamsPerSat * ctr) + niz)
                pco_iz = numParamsPerSat * (ctr+1) - 1 

                nsiz = int(np.floor(data[i,2]/zenSpacing))
                siz = int( tSat +  m*numParamsPerSite + nsiz)

                sol_site = int( tSat + (m+modelNum)*numParamsPerSite + nsiz)
                # check that the indices are not overlapping
                if iz+1 >= pco_iz or iz >= pco_iz:
                    continue
                
                if sol_site+1 >= np.size(Sol) :
                    continue

                if nsiz+1 >= numParamsPerSite :
                    print("nsiz+1 >= numParamsPerSite",nsiz+1,numParamsPerSite)
                    continue

                factor = (nadir/args.nadir_grid-(np.floor(nadir/args.nadir_grid)))
                dNad = Sol[iz] + (Sol[iz+1] - Sol[iz]) * factor  
                dPCO = np.cos(np.radians(nadir))*Sol[pco_iz] 

                factor = (data[i,2]/args.zen-(np.floor(data[i,2]/args.zen)))
                dSit = Sol[sol_site] + (Sol[sol_site+1] - Sol[sol_site]) * factor  

                prefit_tmp = data[i,3]**2 
                prefit     = prefit + prefit_tmp

                postfit_tmp = (data[i,3] - dNad-dPCO-dSit)**2 
                postfit    = postfit + postfit_tmp
                #postfit_all[iz] = data[i,3] - dNad+dPCO-dSit

                mod_rms    += (dNad+dPCO+dSit)**2

                post_res = data[i,3] - dNad-dPCO-dSit # 1.02
                pre_res = data[i,3]
                numObs += 1

                #print("Obs pre post:",i,numObs, np.sqrt(data[i,3]**2), np.sqrt(postfit_rms))

                postfit_sums[iz]     = postfit_sums[iz]     + postfit_tmp
                postfit_sums[iz+1]   = postfit_sums[iz+1]   + postfit_tmp
                postfit_sums[pco_iz] = postfit_sums[pco_iz] + postfit_tmp
                postfit_sums[siz]    = postfit_sums[siz]    + postfit_tmp
                postfit_sums[siz+1]  = postfit_sums[siz+1]  + postfit_tmp

                postfit_res[iz]     = postfit_res[iz]     + post_res
                postfit_res[iz+1]   = postfit_res[iz+1]   + post_res
                postfit_res[pco_iz] = postfit_res[pco_iz] + post_res
                postfit_res[siz]    = postfit_res[siz]    + post_res
                postfit_res[siz+1]  = postfit_res[siz+1]  + post_res

                prefit_sums[iz]     = prefit_sums[iz]     + prefit_tmp
                prefit_sums[iz+1]   = prefit_sums[iz+1]   + prefit_tmp
                prefit_sums[pco_iz] = prefit_sums[pco_iz] + prefit_tmp
                prefit_sums[siz]    = prefit_sums[siz]    + prefit_tmp
                prefit_sums[siz+1]  = prefit_sums[siz+1]  + prefit_tmp

                prefit_res[iz]     = prefit_res[iz]     + pre_res 
                prefit_res[iz+1]   = prefit_res[iz+1]   + pre_res 
                prefit_res[pco_iz] = prefit_res[pco_iz] + pre_res 
                prefit_res[siz]    = prefit_res[siz]    + pre_res 
                prefit_res[siz+1]  = prefit_res[siz+1]  + pre_res 

                numObs_sums[iz]     = numObs_sums[iz]     + 1
                numObs_sums[iz+1]   = numObs_sums[iz+1]   + 1
                numObs_sums[pco_iz] = numObs_sums[pco_iz] + 1
                numObs_sums[siz]    = numObs_sums[siz]    + 1
                numObs_sums[siz+1]  = numObs_sums[siz+1]  + 1

    prefit_rms = np.sqrt(prefit/numObs) 
    postfit_rms = np.sqrt(postfit/numObs)
    mod_rms = np.sqrt(mod_rms/numObs)

    print("PREFIT rms :",prefit_rms,"Postfit rms:",postfit_rms,"Model rms:",mod_rms)
    print("post/pre:",postfit_rms/prefit_rms, "diff:", np.sqrt(prefit_rms**2 - postfit_rms**2))
    print("NumObs:",numObs,np.size(numObs_sums))

    #np.savez_compressed(params['site']+'residuals.npz',residuals=np.array(res_all))

    return prefit,prefit_sums,prefit_res, postfit, postfit_sums, postfit_res, numObs, numObs_sums, params, modelNum

def setUpPostFitTasks(cl3files,svs,Sol,params,svdat,args,tSat,numParamsPerSite,tParams):

    prefit = 0
    prefit_sums = np.zeros(tParams)
    prefit_res = np.zeros(tParams)
    postfit = 0
    postfit_res = np.zeros(tParams)
    numObs = 0
    numObs_sums = np.zeros(tParams)

    print('cpu_count() = {:d}\n'.format(multiprocessing.cpu_count()))
    NUMBER_OF_PROCESSES = multiprocessing.cpu_count()

    if args.cpu < NUMBER_OF_PROCESSES:
        NUMBER_OF_PROCESSES = int(args.cpu)

    pool = multiprocessing.Pool(NUMBER_OF_PROCESSES)

    # Submit the tasks
    results = []
    mdlCtr = 0

    for i in range(0,np.size(cl3files)) :
        print("Submitting job:",params[i]['site'])
        info = params[i]
        results.append(pool.apply_async(calcPostFitBySite,(cl3files[i],svs,Sol,params[i],svdat,args,mdlCtr)))
        mdlCtr = mdlCtr + params[i]['numModels']

    # Wait for all of them to finish before moving on
    for r in results:
        r.wait()
        print("Waiting for results")
        prefit_tmp, prefit_sums_tmp,prefit_res_tmp, postfit_tmp, postfit_sums_tmp,postfit_res_tmp, numObs_tmp, numObs_sums_tmp, info, mdlCtr = r.get()
        print("Received results back")
        prefit = prefit + prefit_tmp
        prefit_sums[0:tSat] = prefit_sums[0:tSat] + prefit_sums_tmp[0:tSat]
        prefit_res[0:tSat] = prefit_ress[0:tSat] + prefit_res_t[0:tSat]

        postfit = postfit + postfit_tmp
        postfit_sums[0:tSat] = postfit_sums[0:tSat] + postfit_sums_tmp[0:tSat]
        postfit_res[0:tSat] = postfit_res[0:tSat] + postfit_res_tmp[0:tSat]

        numObs = numObs + numObs_tmp
        numObs_sums[0:tSat] = numObs_sums[0:tSat] + numObs_sums_tmp[0:tSat]
        #print("RGET:",info['site'],info['numModels'],postfit,mdlCtr,numParamsPerSite)
        ctr = 0
        for m in range(mdlCtr,(info['numModels']+mdlCtr)) :
            # Add in the station dependent models
            start = tSat + m * numParamsPerSite  
            end   = tSat + (m+1) * numParamsPerSite 

            tmp_start = tSat + numParamsPerSite * ctr 
            tmp_end   = tSat + numParamsPerSite * (ctr+1) # + numParamsPerSite 

            prefit_sums[start:end] = prefit_sums[start:end] + prefit_sums_tmp[tmp_start:tmp_end]
            prefit_res[start:end] = prefit_res[start:end] + prefit_res_tmp[tmp_start:tmp_end]

            postfit_sums[start:end] = postfit_sums[start:end] + postfit_sums_tmp[tmp_start:tmp_end]
            postfit_res[start:end] = postfit_res[start:end] + postfit_res_tmp[tmp_start:tmp_end]

            numObs_sums[start:end] = numObs_sums[start:end] + numObs_sums_tmp[tmp_start:tmp_end]

            ctr += 1

    return prefit, prefit_sums,prefit_res, postfit, postfit_sums,postfit_res, numObs, numObs_sums

def prepareSites(cl3files,dt_start,dt_end,args,siteIDList):
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
        for s in range(0,info['numModels']):
            siteIDList.append(info['site']+"_model_"+str(s+1))

    return params, numModels, siteIDList

def stackStationNeqs(Neq,AtWb,nadir_freq,siteIDList,params,tSat):
    #=====================================================================
    # Now read in all of the numpy compressed files
    #=====================================================================
    nctr = 0
    totalSiteModels = 0

    for f in range(0,np.size(params)):
        nfile = params[f]['npzfile'] 

        npzfile = np.load(nfile)
        Neq_tmp  = npzfile['neq']
        AtWb_tmp = npzfile['atwb']

        nadir_freq = np.add(nadir_freq,npzfile['nadirfreq'])

        # only need one copy of the svs array, they should be eactly the same
        if nctr == 0:
            svs_tmp  = npzfile['svs']
            svs = np.sort(svs_tmp)
            del svs_tmp

        # Add the svn component to the Neq
        Neq[0:tSat, 0:tSat] = Neq[0:tSat,0:tSat] + Neq_tmp[0:tSat,0:tSat]
        AtWb[0:tSat]        = AtWb[0:tSat] + AtWb_tmp[0:tSat]

        #===================================
        # Loop over each model 
        #===================================
        for m in range(0,params[f]['numModels']) :
            # Add in the station dependent models
            start = tSat + nctr * numParamsPerSite 
            end = tSat + (nctr+1) * numParamsPerSite

            tmp_start = tSat + numParamsPerSite * m 
            tmp_end   = tSat + numParamsPerSite * m + numParamsPerSite

            #AtWb[start:end] = AtWb[start:end] + AtWb_tmp[tSat:(tSat+numParamsPerSite)]
            AtWb[start:end] = AtWb[start:end] + AtWb_tmp[tmp_start:tmp_end]
            #
            #   ------------------------------------------------
            #  | SVN         | svn + site | svn + site2 | ....
            #  | svn + site  | site       | 0           | ....
            #  | svn + site2 | 0          | site2       | ....
            #

            # Add in the site block 
            Neq[start:end,start:end] = Neq[start:end,start:end] + Neq_tmp[tmp_start:tmp_end,tmp_start:tmp_end]

            # Adding in the correlation with the SVN and site
            Neq[0:tSat,start:end] = Neq[0:tSat,start:end] + Neq_tmp[0:tSat,tmp_start:tmp_end]
            Neq[start:end,0:tSat] = Neq[start:end,0:tSat] + Neq_tmp[tmp_start:tmp_end,0:tSat]
            nctr += 1
            totalSiteModels = totalSiteModels + 1
            siteIDList.append(params[f]['site'])

    #return Neq, AtWb, svs, prefit, prefit_sums, totalSiteModels, siteIDList, nadir_freq
    return Neq, AtWb, svs, totalSiteModels, siteIDList, nadir_freq

def significantSolution(Sol,Cov):
    ctr = 0
    reduced = 0
    for s in Sol:
        if np.abs(s) < (3. * np.sqrt(Cov[ctr,ctr])) :
            #print("Reducing {:.2f} to 0, within 3 sigma {:.2f}".format(s,3.*np.sqrt(Cov[ctr,ctr])))
            Sol[ctr] = 0.
            reduced += 1
        ctr = ctr + 1
    print("Reduced:",reduced,"out of:",np.size(Sol))
    return Sol

#================================================================================
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
    parser.add_argument('--parse_only',dest='parse_only',action='store_true',default=False,help="parse the cl3 file and save the normal equations to a file (*.npz)") 

    parser.add_argument('-f', dest='resfile', default='',help="Consolidated one-way LC phase residuals")

    parser.add_argument('-p','--path',dest='path',help="Search for all CL3 files in the directory path") 

    parser.add_argument('-l','--load',dest='load_file',help="Load stored NEQ and AtWl matrices from a file")
    parser.add_argument('--lp','--lpath',dest='load_path',help="Path to search for .npz files")

    parser.add_argument('--sf1',dest='solutionfile1',help="Pickle Solution file")
    parser.add_argument('--sf2',dest='solutionfile2',help="Numpy Solution file")

    #===================================================================
    # Output options
    #===================================================================
    
    parser.add_argument('--sstk','--save_stacked_file',dest='save_stacked_file',default=False,action='store_true',help="Path to Normal equation stacked file")   
    parser.add_argument('--stk','--stacked_file',dest='stacked_file',help="Path to Normal equation stacked file")   
    parser.add_argument('--save',dest='save_file',default=False, action='store_true',help="Save the Neq and Atwl matrices into numpy compressed format (npz)")
    parser.add_argument('--save_solution','--ss',dest='solution',default='solution.pkl',help="Save the Solution vector and meta data as a pickle object, needs save_file flag to be selected")#,META="Pickle filename")

    #===================================================================
    # Processing options
    #===================================================================
    parser.add_argument('--nadir_grid', dest='nadir_grid', default=0.1, type=float,help="Grid spacing to model NADIR corrections (default = 0.1 degrees)")
    parser.add_argument('--zenith_grid', dest='zen', default=0.5, type=float,help="Grid spacing to model Site corrections (default = 0.5 degrees)")
    parser.add_argument('-m','--model',dest='model',choices=['pwl','pwlSite','pwlSiteDaily'], help="Create a ESM for satellites only, or for satellites and sites")
    parser.add_argument('--cpu',dest='cpu',type=int,default=4,help="Maximum number of cpus to use")
    parser.add_argument('--pf','--post_fit',dest='postfit',default=False,action='store_true',help="Calculate the postfit residuals")
    parser.add_argument('--cholesky',dest='cholesky',default=False,action='store_true',help="Use the cholesky inverse")
    parser.add_argument('--sig',dest='significant',default=False,action='store_true',help="Only apply the model if it is statistical significant from 0 (by 3 sigma)")
    #===================================================================
    # Time period to check for satellite parameters
    parser.add_argument("--syyyy",dest="syyyy",type=int,help="Start yyyy")
    parser.add_argument("--sdoy","--sddd",dest="sdoy",type=int,default=0,help="Start doy")
    parser.add_argument("--eyyyy",dest="eyyyy",type=int,help="End yyyyy")
    parser.add_argument("--edoy","--eddd",dest="edoy",type=int,default=365,help="End doy")

    #===================================================================
    # Constraints 
    parser.add_argument("--no_constraints",dest="apply_constraints",default=True,action='store_false',
                            help="Dont apply constraints")
    parser.add_argument("--nwc","--no_window_contraints",dest="window_constraint",default=True,action='store_false',
                            help="Do not apply a window constraint")
    parser.add_argument("--constrain_SATPCV","--SATPCV", dest="constraint_SATPCV", 
                         default=1.0, type=float, help="Satellite PCV constraint")
    parser.add_argument("--constrain_SATPCO","--SATPCO", dest="constraint_SATPCO", 
                         default=1.0 , type=float, help="Satellite PCO constraint")
    parser.add_argument("--constrain_SATWIN","--SATWIN", dest="constraint_SATWIN", 
                         default=0.5, type=float, help="Satellite Window constraint")
    parser.add_argument("--constrain_SITEPCV","--SITEPCV", dest="constraint_SITEPCV", 
                         default=10., type=float, help="Station PCV constraint")
    parser.add_argument("--constrain_SITEWIN","--SITEWIN", dest="constraint_SITEWIN", 
                         default=1.5, type=float, help="Station Window constraint")

    parser.add_argument("--nadir_zero",dest="constrain_nadir_zero",default=False,action='store_true', help="Constrain Nadir to 0")
    parser.add_argument("--zenith_zero",dest="constrain_zenith_zero",default=False,action='store_true', help="Constrain Zenith to 0")
    #===================================================================
    # Plot options
    parser.add_argument('--plot',dest='plotNadir', default=False, action='store_true', help="Produce an elevation dependent plot of ESM phase residuals")
    parser.add_argument('--ps','--plot_save',dest='savePlots',default=False,action='store_true', help="Save the plots in png format")
    
    #===================================================================
    # Debug function, not needed
    args = parser.parse_args()

    import matplotlib.pyplot as plt

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

    # Number of Parameters
    numNADS = int(14.0/args.nadir_grid) + 1 
    PCOEstimates = 1

    numParamsPerSat = numNADS + PCOEstimates
    numParamsPerSite = int(90./args.zen) + 1 
    #===================================================================
    # Check to see if we need to read in a consolidate residual file (*.CL3)
    #===================================================================
    if args.load_file or args.load_path:
        npzfiles = []

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
        #params = []
        prechis = []
        numds = []

        #=====================================================================
        # first thing, work out how many models and parameters we will need to account for
        #=====================================================================
        for n in range(0,np.size(npzfiles)):
            info = {}
            filename = os.path.basename(npzfiles[n])
            info['basename'] = filename
            info['site'] = filename[0:4]
            info['npzfile'] = npzfiles[n]
            info['cl3file'] = npzfiles[n][:-4]
            cl3files.append(info['cl3file'])
            npzfile = np.load(npzfiles[n])
            AtWb = npzfile['atwb']
            info['num_svs'] = np.size(npzfile['svs'])
            tSat = numParamsPerSat * np.size(npzfile['svs']) 
            info['numModels'] = int((np.size(AtWb) - tSat)/numParamsPerSite)
            mctr = mctr + info['numModels']
            params.append(info)
            prechis.append(npzfile['prechi'][0])
            numds.append(npzfile['numd'][0])
            print("LOADED prechi, numd:",npzfile['prechi'][0],npzfile['numd'][0])

            for s in range(0,info['numModels']):
                siteIDList.append(info['site']+"_model_"+str(s+1))
            del AtWb, npzfile

        #totalSiteModels = mctr
        numSVS = params[0]['num_svs']
        numSites = int(mctr) 
        tSite = numParamsPerSite * numSites
        numParams = tSat + tSite
        print("Total number of models",numSites,tSite,numParams)

    elif args.resfile :
        print("Reading in:", args.resfile)
        cl3files.append(args.resfile)
        siteIDList.append(os.path.basename(args.resfile)[0:4]+"_model_1")
    elif args.path:
        print("Checking {:s} for CL3 files".format(args.path))
        phsRGX = re.compile('.CL3$')
        phsGZRGX = re.compile('.CL3.gz$')
        files = os.listdir(args.path)
        #for root, dirs, files in os.walk(args.path):
        #    path = root.split('/')
        for lfile in files:
            if phsRGX.search(lfile) or phsGZRGX.search(lfile):
                print("Found:",args.path + "/" + lfile)
                cl3files.append(args.path + "/"+ lfile)
        cl3files = np.sort(cl3files)
        totalSites = np.size(cl3files)

    #===================================================================
    # Work out the time scale of observations, and number of parameters
    # that will be solved for. 
    #===================================================================
    #if not args.load_file and not args.load_path:
    if args.model or args.postfit:
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

        # already have the parameters defined if we are loading in the npzfiles..
        if not args.load_file and not args.load_path:
            params, numModels,siteIDList = prepareSites(cl3files,dt_start,dt_stop,args,siteIDList)

        antennas = ant.parseANTEX(args.antex)
        svdat = svnav.parseSVNAV(args.svnavFile)
        svs = ant.satSearch(antennas,dt_start,dt_stop)
        svs = np.sort(np.unique(svs))

    if args.model: 

        if not args.load_file and not args.load_path:
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
            numNADS             = int(14.0/args.nadir_grid) + 1 
            PCOEstimates        = 1
            numSVS              = np.size(svs)
            numParamsPerSat     = numNADS + PCOEstimates
            tSat                = numParamsPerSat * numSVS
            numSites            = numModels # np.size(cl3files)
            tSite               = 0
            numParamsPerSite    = 0

            if args.model == 'pwl':
                numParams = numSVS * (numParamsPerSat)
            elif args.model == 'pwlSite' or args.model== 'pwlSiteDaily':
                numParamsPerSite    = int(90./args.zen) + 1 
                numParams           = numSVS * (numParamsPerSat) + numParamsPerSite * numSites
                tSite               = numParamsPerSite * numSites

            print("Total satellite parameters:",tSat)
            print("Total site parameters     :",tSite)
            print("\t Have:",numParams,"parameters to solve for")

            Neq  = np.zeros((numParams,numParams))
            AtWb = np.zeros(numParams)
            nadir_freq = np.zeros((numSVS,numNADS))
            prefit = 0.0
            postfit = 0.0
            prefit_sums = np.zeros(numParams)
            postfit_sums = np.zeros(numParams)

            for f in range(0,np.size(params)):
                params[f]['npzfile'] = params[f]['filename']+'.npz'

            Neq, AtWb, svs, totalSiteModels, siteIDList, nadir_freq = stackStationNeqs(Neq,
                                                    AtWb,nadir_freq,siteIDList,params,tSat)
            #if args.save_file:
            #    prefitA = [prefit]
            #    postfitA = [postfit]
            #    np.savez_compressed('consolidated.npz',neq=Neq,atwb=AtWb,svs=svs,nadirfreq=nadir_freq,prefit=prefitA,prefitsums=prefit_sums,postfit=postfitA,postfitsums=postfit_sums)

            # remove the unwanted observations after it has been saved to disk
            # as we may want to add to Neq together, which may have observations to satellites not seen in the Neq..
            #Neq,AtWb,svs,nadir_freq,prefit_sums = compressNeq(Neq,AtWb,svs,numParamsPerSat,nadir_freq,prefit_sums)
            Neq,AtWb,svs,nadir_freq = compressNeq(Neq,AtWb,svs,numParamsPerSat,nadir_freq)
            tSat = np.size(svs) * numParamsPerSat
            numParams = tSat + tSite
            numSVS = np.size(svs)
            print("NumParams:",numParams)
            #=====================================================================
            # End of if not load_file or not load_path
            #=====================================================================
            print("FINISHED MP processing, now need to workout stacking:...\n") 

        if args.load_file or args.load_path:

            print("Total number of site models :",mctr, "Total number of paramters to solve for:",numParams)
            Neq = np.zeros((numParams,numParams))
            AtWb = np.zeros(numParams)
            nadir_freq = np.zeros((numSVS,numNADS))
            prefit = 0.
            prefit_sums = np.zeros(numParams)
            mdlCtr = 0
            
            Neq, AtWb, svs, totalSiteModels, siteIDList, nadir_freq = stackStationNeqs(Neq,
                                                    AtWb,nadir_freq,siteIDList,params,tSat)

            # check for any rows/ columns without any observations, if they are empty remove the parameters
            Neq,AtWb,svs,nadir_freq = compressNeq(Neq,AtWb,svs,numParamsPerSat,nadir_freq)
            tSat = np.size(svs) * numParamsPerSat
            numParams = tSat + tSite
            numSVS = np.size(svs)
            print("NumParams:",numParams)

        if args.save_stacked_file:
            np.savez_compressed('stacked.npz',neq=Neq,atwb=AtWb,svs=svs,prechi=prechis,numd=numds,nadirfreq=nadir_freq)

    # check if we are parsing in a pre-stacked file
    if args.stacked_file:
        npzfile = np.load(npzfiles[n])
        Neq     = npzfile['neq']
        AtWb    = npzfile['atwb']
        svs     = npzfile['svs']
        prechis = npzfile['prechi']
        numds   = npzfile['numd']
        nadir_freq = npzfile['nadir_freq']
        print("Just read in stacked file:",args.stacked_file)
        #print("Prechi Numd:",prechi,numd) 

    if args.apply_constraints and not args.solutionfile1:
        #========================================================================
        # Adding Constraints to the satellite parameters,
        # keep the site model free ~ 10mm  0.01 => 1/sqrt(0.01) = 10 (mm)
        # Adding 1 mm constraint to satellites
        #========================================================================
        sPCV_constraint = args.constraint_SATPCV **2  # 0.5 
        sPCO_constraint = args.constraint_SATPCO **2  # 1.5
        sPCV_window     = args.constraint_SATWIN      # 0.5
        site_constraint = args.constraint_SITEPCV **2 #10.0
        site_window     = args.constraint_SITEWIN     #1.5

        C = np.eye(numParams,dtype=float) * sPCV_constraint

        # Add in the Site constraints
        if args.model == 'pwlSite' or args.model == 'pwlSiteDaily' :
            for sitr in range(0,tSite):
                spar = tSat + sitr
                C[spar,spar] = site_constraint 

            # Now add in the off digonal commponents
            sPCV_corr = np.linspace(sPCV_constraint, 0., int(sPCV_window/args.nadir_grid))
            site_corr = np.linspace(site_constraint, 0., int(site_window/args.zen))

            # Add in the correlation constraints for the satellite PCVs
            if args.window_constraint:
                for s in range(0,numSVS):
                    for ind in range(0,numNADS ):
                        start = (s * numParamsPerSat) + ind
                        if ind > (numNADS - np.size(sPCV_corr)):
                            end = start + (numNADS - ind) 
                        else:
                            end = start + np.size(sPCV_corr)
                    
                        #print(start,end,np.shape(C),np.shape(sPCV_corr))
                        C[start,start:end] = sPCV_corr[0:(end - start)] 
                        C[start:end,start] = sPCV_corr[0:(end - start)] 

            # Add in the satellie PCO constraints
            for s in range(0,numSVS):
                ind = (s * numParamsPerSat) + numParamsPerSat - 1 
                #print("PCOCOnstraint",s,ind,sPCO_constraint)
                C[ind,ind] = sPCO_constraint

            if args.window_constraint:
                # Add in the correlation constraints for the sites PCVs
                for s in range(0,numSites):
                    #for ind in range(0,numParamsPerSite-np.size(site_corr) ):
                    for ind in range(0,numParamsPerSite ):
                        start = tSat + (s * numParamsPerSite) + ind
                        if ind > (numParamsPerSite - np.size(site_corr)):
                            end = start + (numParamsPerSite - ind) 
                        else:
                            end = start + np.size(site_corr)
                        
                        C[start,start:end] = site_corr[0:(end - start)] 
                        C[start:end,start] = site_corr[0:(end - start)] 

            # contrain the nadir 0 angle to 0
            if args.constrain_nadir_zero:
                for s in range(0,numSVS):
                    ind = (s * numParamsPerSat) 
                    C[ind,ind] = 0.00001

            # contrain the nadir 0 angle to 0
            if args.constrain_zenith_zero:
                for ind in range(0,numParamsPerSite ):
                    for s in range(0,numSites):
                        ind = tSat + (s * numParamsPerSite) 
                        C[ind,ind] = 0.00001



        C_inv = np.linalg.pinv(C)
        del C
        
        if args.plotNadir or args.savePlots:
            #============================================
            # Plot the sparsity of the matrix Neq before constraints are added
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.spy(Neq, precision=1e-3, marker='.', markersize=5)
            #ax2 = ax.twiny()
            xlabels = []
            xticks = []
            pcoticks = []
            pcolabels = []

            ctr = 0

            for svn in svs:
                siz = numParamsPerSat * ctr 
                eiz = numParamsPerSat *ctr + numNADS 
                ctr = ctr + 1
                xlabels.append(svn)
                tick = int((eiz-siz)/2)+siz
                xticks.append(tick)
                pcoticks.append(eiz)
                pcolabels.append(eiz)
                #print("sat",tick,"pco",eiz)

            for snum in range(0,totalSiteModels):
                siz = numParamsPerSat*numSVS + snum * numParamsPerSite 
                eiz = siz + numParamsPerSite 
                xlabels.append(siteIDList[snum])
                tick = int((eiz-siz)/2)+siz
                xticks.append(tick)
                pcoticks.append(siz)
                pcolabels.append(siz)
                #print("SITE",siz,eiz,siz-eiz)

            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels,rotation='vertical')

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                       ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(6)

            plt.tight_layout()

            if args.savePlots:
                plt.savefig("NeqMatrix.png")
            plt.tight_layout()

        # Add the parameter constraints to the Neq
        Neq = np.add(Neq,C_inv)

    if args.solutionfile1:
        npzfile = np.load(args.solutionfile2)
        Sol  = npzfile['sol']
        Cov  = npzfile['cov']
        nadir_freq = npzfile['nadirfreq']
        #prefit_sums = npzfile['prefitsums'] 

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

        #   meta['syyyy'] = args.syyyy
        #   meta['sddd']  = args.sdoy
        #   meta['eyyyy'] = args.eyyyy
        #   meta['eddd']  = args.edoy
        #   if args.stacked_file:
        #       meta['datafiles'] = args.stacked_file 
        #   else:
        #       meta['datafiles'] = npzfiles
        svs = meta['svs'] 
        numSites = meta['numSiteModels'] 
        siteIDList = meta['siteIDList'] 
        prechi = meta['prechi']   #= np.sqrt(prechi/numd)
        postchi = meta['postchi'] # = np.sqrt(postchi/numd)
        numd = meta['numd']
        chi_inc =  meta['chi_inc'] # = np.sqrt((prechi-postchi)/numd)
        #    meta['apply_constraints'] = args.apply_constraints
        #    if args.apply_constraints:
        #        meta['constraint_SATPCV']  = args.constraint_SATPCV # 0.5 
        #        meta['constraint_SATPCO']  = args.constraint_SATPCO # 1.5
        #        meta['constraint_SATWIN']  = args.constraint_SATWIN # 0.5
        #        meta['constraint_SITEPCV'] = args.constraint_SITEPCV #10.0
        #        meta['constraint_SITEWIN'] = args.constraint_SITEWIN #1.5
        #    meta['saved_file'] = args.solution + ".sol"
        prefit = meta['prefit']
        #    if args.postfit:
        #        meta['postfit'] = postfit
        tSat = np.size(svs) * (int(14.0/args.nadir_grid)+ 1 +1)
        numParamsPerSite = int(90.0/args.zen)+1
    else:
        print("Now trying an inverse of Neq",np.shape(Neq))
        if args.cholesky:
            print("Using cholsky inversion technique")
            Cho = np.linalg.cholesky(Neq)
            Cho_inv = np.linalg.pinv(Cho)
            Cov = np.dot(Cho_inv.T,Cho_inv)
        else:
            Cov = np.linalg.pinv(Neq)

        print("Now computing the solution")
        Sol = np.dot(Cov,AtWb)
        print("The solution is :",np.shape(Sol))

        prechi = np.sum(prechis)
        numd = np.sum(numds)
        #print("Prechi, numd",prechi,numd)
             
        postchi = prechi - np.dot(np.array(AtWb).T,np.array(Sol))
        print("STATS:",numd,np.sqrt(prechi/numd),np.sqrt(postchi/numd),np.sqrt((prechi-postchi)/numd))#,aic,bic)
        print("stats:",prefit,prefit_sums)

    if args.significant:
        print("Screen the solution to only accept values which are statistically significant")
        Sol = significantSolution(Sol,Cov)

    # calculate the post-fit residuals
    if args.postfit:
        # Initialise
        prefit = 0
        prefit_sums = np.zeros(numParams)
        prefit_res = np.zeros(numParams)

        postfit = 0
        postfit_sums = np.zeros(numParams)
        postfit_res = np.zeros(numParams)

        numObs_sums = np.zeros(numParams)

        print("Calculating post-fit residuals")
        # re calcaulte params based on raw data
        if args.load_file or args.load_path:
            params, numModels, siteIDList = prepareSites(cl3files,dt_start,dt_stop,args,siteIDList)

        # prepare multicore/cpu processing
        multiprocessing.freeze_support()

        prefit,prefit_sums,prefit_res,postfit,postfit_sums,postfit_res,numObs,numObs_sums = setUpPostFitTasks(cl3files,svs,Sol,params,svdat,args,tSat,numParamsPerSite,np.size(Sol))
        print("Prefit, Postfit, Postfit/Prefit",prefit,postfit,postfit/prefit) #np.sqrt(prechi/numd))

        prefit_svs = np.sum(prefit_sums[0:tSat])
        postfit_svs = np.sum(postfit_sums[0:tSat])
        #print("PREFIT_SUM summed:",np.sum(prefit_sums))
        #print("postfit_sums summed:",np.sum(postfit_sums))
        print("SVS Prefit, Postfit, Postfit/Prefit",prefit_svs,postfit_svs,postfit_svs/prefit_svs) 

        print("=====================")
        print("Prefit_rms : {:.2f}".format(np.sqrt(prefit/numObs)))
        print("Postfit_rms: {:.2f}".format(np.sqrt(postfit/numObs)))
        print("Ratio      : {:.2f}".format(np.sqrt(postfit/numObs)/np.sqrt(prefit/numObs)))
        print("Difference : {:.2f}".format(np.sqrt(prefit/numObs - postfit/numObs)))
        print("=====================")
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
            if args.postfit:
                meta['prefit'] = prefit 
                meta['postfit'] = postfit
            pickle.dump(meta,pklID,2)
            pklID.close()            

            #np.savez_compressed(args.solution+".sol",sol=Sol,cov=Cov,nadirfreq=nadir_freq)
            if args.postfit:
                np.savez_compressed(args.solution+".sol",sol=Sol,cov=Cov,nadirfreq=nadir_freq,
                                prefitsums=prefit_sums,postfitsum=postfit_sums,numobs=numObs_sums)
            else:
                np.savez_compressed(args.solution+".sol",sol=Sol,cov=Cov,nadirfreq=nadir_freq,
                                prefitsums=prefit_sums)

    if args.plotNadir or args.savePlots:
        nad = np.linspace(0,14, int(14./args.nadir_grid)+1 )
        numParamsPerSat = int(14.0/args.nadir_grid) + 2 

        #variances = np.diag(Neq)
        variances = np.diag(Cov)
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

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels,rotation='vertical')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                       ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(6)

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
            ax = fig.add_subplot(111)

            siz = numParamsPerSat * ctr 
            eiz = numParamsPerSat *ctr + numNADS 
           
        #    sol = Sol[siz:eiz]
            #print("SVN:",svn,siz,eiz,numParamsPerSat,tSat)
            ax.errorbar(nad,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,fmt='o')
            #ax.errorbar(nad,sol[::-1],yerr=np.sqrt(variances[siz:eiz])/2.)

            ax.set_xlabel('Nadir Angle (degrees)',fontsize=8)
            ax.set_ylabel('Phase Residuals (mm)',fontsize=8)

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                       ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(8)

            plt.tight_layout()

            if args.savePlots:
                plt.savefig("SVN_"+svn+"_nadirCorrectionModel.png")
            ctr += 1
                
        #    if ctr > 2:
        #        break

        #==================================================
        #fig = plt.figure(figsize=(3.62, 2.76))
        fig = plt.figure()
        fig.canvas.set_window_title("PCO_correction.png")
        ax = fig.add_subplot(111)
        ctr = 0
        numSVS = np.size(svs)
        numNADS = int(14.0/args.nadir_grid) + 1 
        numParamsPerSat = numNADS + PCOEstimates
        print("Number of Params per Sat:",numParamsPerSat,"numNads",numNADS,"Sol",np.shape(Sol))
        for svn in svs:
            eiz = numParamsPerSat *ctr + numParamsPerSat -1 
            #print("PCO:",eiz)
            ax.errorbar(ctr+1,Sol[eiz],yerr=np.sqrt(variances[eiz])/2.,fmt='o')
            ctr += 1

        ax.set_xlabel('SVN',fontsize=8)
        ax.set_ylabel('Adjustment to PCO (mm)',fontsize=8)

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                   ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)
        plt.tight_layout()

        if args.savePlots:
            plt.savefig("PCO_correction.png")

        #==================================================
        if args.model == 'pwlSite' or args.model == 'pwlSiteDaily':
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
                ax = fig.add_subplot(111)
                siz = numParamsPerSat*numSVS + snum * numParamsPerSite 
                eiz = siz + numParamsPerSite 
                ele = np.linspace(0,90,numParamsPerSite)
                #print("Sol",np.shape(Sol),"siz  ",siz,eiz)
                #ax1.plot(ele,Sol[siz:eiz],'k.',linewidth=2)
                ax.errorbar(ele,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,fmt='o')

                ax.set_xlabel('Zenith Angle',fontsize=8)
                ax.set_ylabel('Adjustment to PCV (mm)',fontsize=8)

                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(8)

                plt.tight_layout()
                if args.savePlots:
                    plt.savefig(siteIDList[snum]+"_elevation_model.png")
                
        if args.plotNadir:
            plt.show()

    print("FINISHED")
