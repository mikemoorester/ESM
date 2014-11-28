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
import nadir as NADIR
#import broadcastNavigation as brdc
import Navigation as rnxN


def formConstraints(args):
    
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

        # constrain the low nadir angle 13.8 ,13.9, 14.0 to 0
        # as these will have a low number of observations
        # but a high variance
        if args.constrain_nadir_low > 0.00000000:
            for s in range(0,numSVS):
                nadir = 13.8
                niz = int(np.floor(nadir/args.nadir_grid))
                iz = int((numParamsPerSat * s) + niz)
                C[iz,iz] = args.constrain_nadir_low 

                nadir = 13.8
                niz = int(np.floor(nadir/args.nadir_grid))
                iz = int((numParamsPerSat * s) + niz)
                C[iz,iz] = args.constrain_nadir_low

                nadir = 14.0
                niz = int(np.floor(nadir/args.nadir_grid))
                iz = int((numParamsPerSat * s) + niz)
                C[iz,iz] = args.constrain_nadir_low


    C_inv = np.linalg.pinv(C)
    del C
        
    return C_inv

def processAzSlice() :
    
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
        
        if(az - args.az/2. < 0) :
            criterion = (data[:,1] < (az + args.az/2.)) | (data[:,1] > (360. - args.az/2.) )
        else:
            criterion = (data[:,1] < (az + args.az/2.)) & (data[:,1] > (az - args.az/2.) )
            
        azind = np.array(np.where(criterion))[0]
        print("Size of data before azimuth search",np.size(data))
        data = data[azind,:]    
        print("Size of data after azimuth search",np.size(data))
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
            nadir = NADIR.calcNadirAngle(data[i,2],site_geocentric_distance,satnorm)
            #print("Ele {:.2f} Old: {:.2f} New:{:.2f}".format(data[i,2],oldnadir,nadir))
            #print("Ele {:.2f} New:{:.2f}".format(data[i,2],nadir))
            w = a**2 + b**2/np.sin(np.radians(90.-data[i,2]))**2
            w = 1./w
        
            # Work out the indices for the satellite parameters
            niz = int(np.floor(nadir/nadSpacing))
            iz = int((numParamsPerSat * ctr) + niz)
            pco_iz = numParamsPerSat * (ctr+1) - 1 
        
            # work out the location of site parameters
            nsiz = int(np.floor(data[i,2]/zenSpacing))
            aiz = int(np.floor(data[i,1]/azSpacing))
            
            #siz = int( tSat +  (m*numParamsPerSite) + (aiz * nZen) + nsiz)
            siz = int( tSat + nsiz)
        
            # check that the indices are not overlapping
            if iz+1 >= pco_iz or iz >= pco_iz:
                #print("WARNING in indices iz+1 = pco_iz skipping obs",nadir,iz,pco_iz)
                continue
        
            NadirFreq[ctr,niz] = NadirFreq[ctr,niz] +1
            SiteFreq[aiz,nsiz] = SiteFreq[aiz,nsiz] +1
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
                
                
            # Add the parameter constraints to the Neq
                #Neq = np.add(Neq,C_inv) 
            print("Inverting")
            Cov = np.linalg.pinv(Neq)
            Sol = np.dot(Cov,AtWb)
            model[az,:] = Sol[tSat:] 

def setUpAzTasks(site_residuals,svs,opts,params):
    prechi = 0
    numd = 0
    print('cpu_count() = {:d}\n'.format(multiprocessing.cpu_count()))
    NUMBER_OF_PROCESSES = multiprocessing.cpu_count()

    if opts.cpu < NUMBER_OF_PROCESSES:
        NUMBER_OF_PROCESSES = int(opts.cpu)

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
#==============================================================================
def solveSiteModel(site_residuals, svs, params, apr, nadSpacing=0.1, zenSpacing=0.5, azSpacing=0.5, brdc_dir="./"):
    """
    Create a model for the satellites and sites at the same time.
    PWL piece-wise-linear interpolation fit of phase residuals
    -construct a PWL fit for each azimuth bin, and then paste them all together to get 
     the full model
    -inversion is done within each bin

    site_residuals = the one-way L3 post-fit, ambiguity fixed phase residuals
    svs = an array of satellite SVN numbers that are spacebourne/operating
          for the period of this residual stack
    params = meta data about the solution bein attempted
             ['site'] = 4 char site id
             ['changes'] = dictionary of when model changes need to be applied
    apr = satellite apriori data
    
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

    nZen = int(90.0/zenSpacing) + 1
    nAz = int(360./azSpacing) 
    numParamsPerSite = nZen * nAz
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
    SiteFreq = np.zeros((nAz,nZen))
    gamitWeight = {}
    
    # create a new model everythime there has been a change of antenna
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

        for az in range(0,nAz):
            print("Running AZ:",az)
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
                
                if(az - args.az/2. < 0) :
                    criterion = (data[:,1] < (az + args.az/2.)) | (data[:,1] > (360. - args.az/2.) )
                else:
                    criterion = (data[:,1] < (az + args.az/2.)) & (data[:,1] > (az - args.az/2.) )
                    
                azind = np.array(np.where(criterion))[0]
                print("Size of data before azimuth search",np.size(data))
                data = data[azind,:]    
                print("Size of data after azimuth search",np.size(data))
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
                    nadir = NADIR.calcNadirAngle(data[i,2],site_geocentric_distance,satnorm)
                    #print("Ele {:.2f} Old: {:.2f} New:{:.2f}".format(data[i,2],oldnadir,nadir))
                    #print("Ele {:.2f} New:{:.2f}".format(data[i,2],nadir))
                    w = a**2 + b**2/np.sin(np.radians(90.-data[i,2]))**2
                    w = 1./w
                
                    # Work out the indices for the satellite parameters
                    niz = int(np.floor(nadir/nadSpacing))
                    iz = int((numParamsPerSat * ctr) + niz)
                    pco_iz = numParamsPerSat * (ctr+1) - 1 
                
                    # work out the location of site parameters
                    nsiz = int(np.floor(data[i,2]/zenSpacing))
                    aiz = int(np.floor(data[i,1]/azSpacing))
                    
                    #siz = int( tSat +  (m*numParamsPerSite) + (aiz * nZen) + nsiz)
                    siz = int( tSat + nsiz)
                
                    # check that the indices are not overlapping
                    if iz+1 >= pco_iz or iz >= pco_iz:
                        #print("WARNING in indices iz+1 = pco_iz skipping obs",nadir,iz,pco_iz)
                        continue
                
                    NadirFreq[ctr,niz] = NadirFreq[ctr,niz] +1
                    SiteFreq[aiz,nsiz] = SiteFreq[aiz,nsiz] +1
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
                        
                        
        # Add the parameter constraints to the Neq
        #Neq = np.add(Neq,C_inv) 
        print("Inverting")
        Cov = np.linalg.pinv(Neq)
        Sol = np.dot(Cov,AtWb)
        model[az,:] = Sol[tSat:] 

        prechi = prechi + np.dot(data[:,3].T,data[:,3])
        NUMD = NUMD + numd
    print("Normal finish of pwlNadirSiteDailyStack",prechi,NUMD)
    
    return Neq, AtWb, prechi, NUMD, NadirFreq, SiteFreq


def setUpTasks(cl3files,svs,opts,params):
    prechi = 0
    numd = 0
    print('cpu_count() = {:d}\n'.format(multiprocessing.cpu_count()))
    NUMBER_OF_PROCESSES = multiprocessing.cpu_count()

    if opts.cpu < NUMBER_OF_PROCESSES:
        NUMBER_OF_PROCESSES = int(opts.cpu)

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
   
def AzStripNeq(Neq,AtWb,svs,args,azimuth):
    # check for any rows/ columns without any observations, if they are empty remove the parameters

    numNADS = int(14.0/args.nadir_grid) + 1 
    PCOEstimates = 1
    numSVS = np.size(svs)
    numParamsPerSat = numNADS + PCOEstimates
    tSat = numParamsPerSat * numSVS

    nZen = int(90.0/args.zen) + 1
    nAz = int(360./args.az) 
    numParamsPerSite = nZen * nAz
    tSite = numParamsPerSite*params['numModels']
    numParams = tSat + tSite 

    satCtr = 0
    starts = []
    ends   = []
    remove = []

    # First remove any azimuth not in the requested strip
    nsiz = int(np.floor(data[i,2]/zenSpacing))
    aiz = int(np.floor(data[i,1]/azSpacing))
                
    siz = int( tSat +  (m*numParamsPerSite) + (aiz * nZen) + nsiz)    
    
    
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
                nadir = NADIR.calcNadirAngle(data[i,2],site_geocentric_distance,satnorm)

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
        prefit_res[0:tSat] = prefit_res[0:tSat] + prefit_res_tmp[0:tSat]

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
    > python ~/gg/com/nadirSiteModel.py --model -f ./t/YAR2.2012.CL3
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
    parser.add_argument('--nadir_grid', dest='nadir_grid', default=0.1, type=float, help="Grid spacing to model NADIR corrections (default = 0.1 degrees)")
    parser.add_argument('--zenith_grid', dest='zen', default=0.5, type=float, help="Zenith grid spacing to model Site corrections (default = 0.5 degrees)")
    parser.add_argument('--azimuth_grid', dest='az', default=0.5, type=float, help="Azimuth grid spacing to model Site corrections (default = 0.5 degrees)")
    parser.add_argument('-m','--model',dest='model',choices=['pwl'], help="Create a ESM for satellites only, or for satellites and sites")
    parser.add_argument('--cpu',dest='cpu',type=int,default=4,help="Maximum number of cpus to use")
    parser.add_argument('--pf','--post_fit',dest='postfit',default=False,action='store_true',help="Calculate the postfit residuals")
   
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
    #parser.add_argument('--plot',dest='plotNadir', default=False, action='store_true', help="Produce an elevation dependent plot of ESM phase residuals")
    #parser.add_argument('--ps','--plot_save',dest='savePlots',default=False,action='store_true', help="Save the plots in png format")
    
    #===================================================================
    # Debug function, not needed
    args = parser.parse_args()

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
    
    print("Reading in:", args.resfile)
    
    # read in the consolidated LC residuals
    site_residuals = res.parseConsolidatedNumpy(args.resfile)
    
    #===================================================================
    # Work out the time scale of observations, and number of parameters
    # that will be solved for. 
    #===================================================================
  
    if args.syyyy and args.eyyyy:
        dt_start = dt.datetime(int(args.syyyy),01,01) + dt.timedelta(days=int(args.sdoy))
        dt_stop  = dt.datetime(int(args.eyyyy),01,01) + dt.timedelta(days=int(args.edoy))
    else:
        print("")
        print("Warning:")
        print("\tusing:",args.resfile,"to work out the time period to determine how many satellites were operating.")
        print("")
        
        dt_start = gt.unix2dt(site_residuals[0,0])
        res_start = int(dt_start.strftime("%Y") + dt_start.strftime("%j"))
        dt_stop = gt.unix2dt(site_residuals[-1,0])
        res_stop = int(dt_stop.strftime("%Y") + dt_stop.strftime("%j"))
        print("\tResiduals run from:",res_start,"to:",res_stop)
    
    filename    = os.path.basename(args.resfile)
    siteID      = filename[0:4]
    sdata       = gsf.parseSite(args.station_file,siteID.upper())
    changes     = gsf.determineESMChanges(dt_start,dt_stop,sdata)
    sitepos     = gapr.getStationPos(args.apr_file,siteID)
    numModels   = np.size(changes['ind']) + 1
    
    info = {}
    info['filename']  = args.resfile
    info['basename']  = filename
    info['site']      = siteID
    info['numModels'] = np.size(changes['ind']) + 1 
    info['changes']   = changes
    info['sitepos']   = sitepos
    params.append(info)
    
    for s in range(0,info['numModels']):
        siteIDList.append(info['site']+"_model_"+str(s+1))    
    antennas = ant.parseANTEX(args.antex)
    svdat = svnav.parseSVNAV(args.svnavFile)
    svs = ant.satSearch(antennas,dt_start,dt_stop)
    svs = np.sort(np.unique(svs))
 
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
  
    if args.model== 'pwl':
        numParamsPerSite    = int(90./args.zen) + 1 
        numParams           = numSVS * (numParamsPerSat) + numParamsPerSite * numSites
        tSite               = numParamsPerSite * numSites
    tParams = tSat + tSite
    print("Total satellite parameters:",tSat)
    print("Total site parameters     :",tSite)
    print("\t Have:",numParams,"parameters to solve for")    
    
    apr = np.zeros(tParams)
        
    # create the site model
    # model = solveSiteModel(args.resfile)
    model = solveSiteModel(site_residuals, svs, info, apr, args.nadir_grid, args.zen, args.az, args.brdc_dir)

#==============================================================================
#     print("Prechi",prechi,np.sqrt(prechi/numd))
#     prechis.append(prechi)
#     numds.append(numd)
#     
#     #=====================================================================
#     # add one to make sure we have a linspace which includes 0.0 and 14.0
#     # add another parameter for the zenith PCO estimate
#     #=====================================================================
#     numNADS             = int(14.0/args.nadir_grid) + 1 
#     PCOEstimates        = 1
#     numSVS              = np.size(svs)
#     numParamsPerSat     = numNADS + PCOEstimates
#     tSat                = numParamsPerSat * numSVS
#     numSites            = numModels # np.size(cl3files)
#     tSite               = 0
#     numParamsPerSite    = 0
# 
#     
#     if args.model== 'pwl':
#         numParamsPerSite    = int(90./args.zen) + 1 
#         numParams           = numSVS * (numParamsPerSat) + numParamsPerSite * numSites
#         tSite               = numParamsPerSite * numSites
# 
#     print("Total satellite parameters:",tSat)
#     print("Total site parameters     :",tSite)
#     print("\t Have:",numParams,"parameters to solve for")
# 
#     Neq  = np.zeros((numParams,numParams))
#     AtWb = np.zeros(numParams)
#     nadir_freq = np.zeros((numSVS,numNADS))
#     prefit = 0.0
#     postfit = 0.0
#     prefit_sums = np.zeros(numParams)
#     postfit_sums = np.zeros(numParams)
# 
#     
#     # remove the unwanted observations after it has been saved to disk
#     # as we may want to add to Neq together, which may have observations to satellites not seen in the Neq..
#     Neq,AtWb,svs,nadir_freq = compressNeq(Neq,AtWb,svs,numParamsPerSat,nadir_freq)
#     tSat = np.size(svs) * numParamsPerSat
#     numParams = tSat + tSite
#     numSVS = np.size(svs)
#     print("NumParams:",numParams)
#     #=====================================================================
#     # End of if not load_file or not load_path
#     #=====================================================================
#     print("FINISHED MP processing, now need to workout stacking:...\n") 
# 
#         prechi = np.sum(prechis)
#         numd = np.sum(numds)
#         #print("Prechi, numd",prechi,numd)
#              
#         postchi = prechi - np.dot(np.array(AtWb).T,np.array(Sol))
#         print("STATS:",numd,np.sqrt(prechi/numd),np.sqrt(postchi/numd),np.sqrt((prechi-postchi)/numd))#,aic,bic)
#         print("stats:",prefit,prefit_sums)
# 
#     # calculate the post-fit residuals
#     if args.postfit:
#         # Initialise
#         prefit = 0
#         prefit_sums = np.zeros(numParams)
#         prefit_res = np.zeros(numParams)
# 
#         postfit = 0
#         postfit_sums = np.zeros(numParams)
#         postfit_res = np.zeros(numParams)
# 
#         numObs_sums = np.zeros(numParams)
# 
#         print("Calculating post-fit residuals")
#         # re calcaulte params based on raw data
#         if args.load_file or args.load_path:
#             params, numModels, siteIDList = prepareSites(cl3files,dt_start,dt_stop,args,siteIDList)
# 
#         # prepare multicore/cpu processing
#         multiprocessing.freeze_support()
# 
#         prefit,prefit_sums,prefit_res,postfit,postfit_sums,postfit_res,numObs,numObs_sums = setUpPostFitTasks(cl3files,svs,Sol,params,svdat,args,tSat,numParamsPerSite,np.size(Sol))
#         print("Prefit, Postfit, Postfit/Prefit",prefit,postfit,postfit/prefit) #np.sqrt(prechi/numd))
# 
#         prefit_svs = np.sum(prefit_sums[0:tSat])
#         postfit_svs = np.sum(postfit_sums[0:tSat])
#         #print("PREFIT_SUM summed:",np.sum(prefit_sums))
#         #print("postfit_sums summed:",np.sum(postfit_sums))
#         print("SVS Prefit, Postfit, Postfit/Prefit",prefit_svs,postfit_svs,postfit_svs/prefit_svs) 
# 
#         print("=====================")
#         print("Prefit_rms : {:.2f}".format(np.sqrt(prefit/numObs)))
#         print("Postfit_rms: {:.2f}".format(np.sqrt(postfit/numObs)))
#         print("Ratio      : {:.2f}".format(np.sqrt(postfit/numObs)/np.sqrt(prefit/numObs)))
#         print("Difference : {:.2f}".format(np.sqrt(prefit/numObs - postfit/numObs)))
#         print("=====================")
#==============================================================================


    print("FINISHED")
