#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

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
    # Pretty picture optons
    parser.add_argument("--plot",dest="plot",action='store_true',default=False,help="Display plots of prefit and postfit residuals")
    parser.add_argument("--SATPCV",dest="SATPCV",action='store_true',default=False,help="Display plots of prefit and postfit residuals")
    parser.add_argument("--SATPCO",dest="SATPCO",action='store_true',default=False,help="Display plots of prefit and postfit residuals")
    parser.add_argument("--PCV",dest="PCV",action='store_true',default=False,help="Display plots of prefit and postfit residuals")
    #===================================================================
    # Debug function, not needed
    args = parser.parse_args()
   
    
    if args.sum:
        ctr = 0
        pftRGX = re.compile('.pft.npz')
        files = os.listdir(args.sum)
        for lfile in files:
            if pftRGX.search(lfile):
                npzfile = np.load(lfile)
                print("Loading file:",lfile)
                if ctr == 0:
                    prefit_sums = npzfile['prefitsum']
                    prefit_res = npzfile['prefitres']
                    prefit = npzfile['prefit'][0]

                    postfit_res  = npzfile['postfitres']
                    postfit_sums = npzfile['postfitsum']
                    postfit = npzfile['postfit'][0]

                    numObs = npzfile['numobs'][0]
                    numObs_sums = npzfile['numobssum']
                else:
                    prefit_res = np.add(prefit_res,npzfile['prefitres'])
                    prefit_sums = np.add(prefit_sums,npzfile['prefitsum'])
                    prefit += npzfile['prefit'][0]

                    postfit_res = np.add(postfit_res,npzfile['postfitres'])
                    postfit_sums = np.add(postfit_sums,npzfile['postfitsum'])
                    postfit += npzfile['postfit'][0]
                    numObs += npzfile['numobs'][0]
                    numObs_sums = np.add(numObs_sums,npzfile['numobssum'])

                ctr += 1
        print("=====================")
        print("Prefit_rms : {:.2f}".format(np.sqrt(prefit/numObs)))
        print("Postfit_rms: {:.2f}".format(np.sqrt(postfit/numObs)))
        print("Ratio      : {:.2f}".format(np.sqrt(postfit/numObs)/np.sqrt(prefit/numObs)))
        print("Difference : {:.2f}".format(np.sqrt(prefit/numObs - postfit/numObs)))
        print("=====================")

        print("postfit is",postfit)    
        print("Prefit, Postfit, Postfit/Prefit",prefit,postfit,postfit/prefit)
        print("SUMMED:",np.sum(prefit_sums),np.sum(postfit_sums),np.sum(postfit_sums)/np.sum(prefit_sums))
    	# Now read the pickle file
    	with open(args.solutionfile1,'rb') as pklID:
            meta = pickle.load(pklID)
        pklID.close()

        args.nadir_grid     = meta['nadir_grid'] 
        args.zen            = meta['zenith_grid'] 
        svs                 = meta['svs'] 

        # Number of Parameters
        numNADS             = int(14.0/args.nadir_grid) + 1 
        PCOEstimates        = 1
        numSVS              = np.size(svs)
        numParamsPerSat     = numNADS + PCOEstimates
        tSat                = numParamsPerSat * numSVS    

        prefit_svs = np.sum(prefit_sums[0:tSat])
        postfit_svs = np.sum(postfit_sums[0:tSat])
   
        print("SVS Prefit, Postfit, Postfit/Prefit",prefit_svs,postfit_svs,postfit_svs/prefit_svs)
        #sys.exit(0)                    
    else:                
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

        svs         = meta['svs'] 
        numSites    = meta['numSiteModels'] 
        siteIDList  = meta['siteIDList'] 
        prechi      = meta['prechi']   #= np.sqrt(prechi/numd)
        postchi     = meta['postchi'] # = np.sqrt(postchi/numd)
        numd        = meta['numd']
        chi_inc     = meta['chi_inc'] # = np.sqrt((prechi-postchi)/numd)
  
        #prefit      = meta['prefit']
    
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

        # Number of Parameters
        numNADS             = int(14.0/args.nadir_grid) + 1 
        PCOEstimates        = 1
        numSVS              = np.size(svs)
        numParamsPerSat     = numNADS + PCOEstimates
        tSat                = numParamsPerSat * numSVS    
        numParamsPerSite    = int(90.0/args.zen)+1
        numParams           = np.size(Sol)
   
        print("SVS",svs)
        print(siteIDList)
        #IDX = siteIDList.index(siteIDSRCH)
        mdlCtr = siteIDList.index(siteIDSRCH+'_model_1')
    
        print("Model Counter",mdlCtr)
        # get ready to start caclulation
        postfit = 0.0
        prefit  = 0.0
        postfit_sums = np.zeros(numParams)
        postfit_res = np.zeros(numParams)
        prefit_sums = np.zeros(numParams)
        prefit_res = np.zeros(numParams)
        numObs = 0.0
        numObs_sums = np.zeros(numParams)
        
        #prefit, prefit_sums, postfit, postfit_sums, info, mdlCtr = calcPostFitBySite(cl3files[0],svs,Sol,params[0],args,mdlCtr)
        prefit_tmp, prefit_sums_tmp,prefit_res_tmp, postfit_tmp, postfit_sums_tmp,postfit_res_tmp,numObs_tmp,numObs_sums_tmp, info,modelNum = NADIR.calcPostFitBySite(cl3files[0],svs,Sol,params[0],svdat,args,mdlCtr)
 
        prefit = prefit + prefit_tmp
        prefit_sums[0:tSat] = prefit_sums[0:tSat] + prefit_sums_tmp[0:tSat]
        prefit_res[0:tSat]  = prefit_res[0:tSat] + prefit_res_tmp[0:tSat]

        postfit = postfit + postfit_tmp
        postfit_sums[0:tSat] = postfit_sums[0:tSat] + postfit_sums_tmp[0:tSat]
        postfit_res[0:tSat] = postfit_res[0:tSat] + postfit_res_tmp[0:tSat]
        numObs = numObs + numObs_tmp
        numObs_sums[0:tSat] = numObs_sums[0:tSat] + numObs_sums_tmp[0:tSat]

        ctr = 0
        for m in range(mdlCtr,(info['numModels']+mdlCtr)) :
            # Add in the station dependent models
            start = tSat + m * numParamsPerSite
            end   = tSat + (m+1) * numParamsPerSite

            tmp_start = tSat + numParamsPerSite * ctr
            tmp_end   = tSat + numParamsPerSite * (ctr+1) # + numParamsPerSite 
            #print(m,start,end,tmp_start,tmp_end,np.size(Sol))
            #print("postfit_sums",start,end,np.size(postfit_sums))
            #print("postfit_sums_tmp",tmp_start,tmp_end,np.size(postfit_sums_tmp))
            prefit_sums[start:end] = prefit_sums[start:end] + prefit_sums_tmp[tmp_start:tmp_end]
            prefit_res[start:end] = prefit_res[start:end] + prefit_res_tmp[tmp_start:tmp_end]
            postfit_sums[start:end] = postfit_sums[start:end] + postfit_sums_tmp[tmp_start:tmp_end]
            postfit_res[start:end] = postfit_res[start:end] + postfit_res_tmp[tmp_start:tmp_end]
            numObs_sums[start:end] = numObs_sums[start:end] + numObs_sums_tmp[tmp_start:tmp_end]

            ctr += 1

        
        print("Prefit, Postfit, Postfit/Prefit",prefit,postfit,postfit/prefit)
        print("Summed:",np.sum(prefit_sums),np.sum(postfit_sums)) 
        prefit_svs = np.sum(prefit_sums[0:tSat])
        postfit_svs = np.sum(postfit_sums[0:tSat])
        print("SVS Prefit, Postfit, Postfit/Prefit",prefit_svs,postfit_svs,postfit_svs/prefit_svs)
        prefitA = [ prefit ]
        postfitA = [ postfit ]
        numobsA = [ numObs ]
        np.savez_compressed(siteIDSRCH+".pft", prefit=prefitA, prefitsum=prefit_sums,prefitres=prefit_res, postfit=postfitA, postfitsum=postfit_sums,postfitres=postfit_res,numobs=numobsA,numobssum=numObs_sums)

    #===================================================================================================
    if args.plot or args.SATPCV or args.SATPCO or args.PCV:
        import matplotlib.pyplot as plt
        npzfile = np.load(args.solutionfile2)
        Sol  = npzfile['sol']
        Cov  = npzfile['cov']
        #Sol = NADIR.significantSolution(Sol,Cov)
        # Number of Parameters
        numNADS             = int(14.0/args.nadir_grid) + 1 
        PCOEstimates        = 1
        numSVS              = np.size(svs)
        numParamsPerSat     = numNADS + PCOEstimates
        tSat                = numParamsPerSat * numSVS    
        numParamsPerSite    = int(90.0/args.zen)+1
        numParams           = np.size(Sol)

        totalSiteModels = meta['numSiteModels']
        nad = np.linspace(0,14, int(14./meta['nadir_grid'])+1 )
        numParamsPerSat = int(14.0/meta['nadir_grid']) + 2
        variances = np.diag(Cov)

    if args.SATPCV or args.plot:
        ctr = 0
        for svn in svs:
            fig = plt.figure()
            fig.canvas.set_window_title("SVN_"+svn+"_nadirCorrectionComparison")
            ax  = fig.add_subplot(311)

            siz = numParamsPerSat * ctr
            eiz = (numParamsPerSat * (ctr+1)) - 1

            sol = Sol[siz:eiz]

            ratio = np.sum(postfit_sums[siz:eiz])/np.sum(prefit_sums[siz:eiz])
            titleStr = "{:s} RATIO: {:.2f}".format(svn,ratio) 
            print(titleStr)
            plt.title(titleStr,fontsize=10)
            ax.errorbar(nad,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,linewidth=2,fmt='b')

            ax2 = fig.add_subplot(312) 
            ax3 = fig.add_subplot(313) 

            ax2.plot(nad,prefit_res[siz:eiz]/numObs_sums[siz:eiz],'b-',linewidth=2)
            #ax2.plot(nad,postfit_res[siz:eiz]/numObs_sums[siz:eiz],'g-')
            stdev = []  # standard deviation of postfit residual
            mean = []   # mean postfit residual
            ictr = 0
            for i in range(siz,eiz):
                if numObs_sums[i] > 1:
                    mean.append( postfit_res[i]/numObs_sums[i] )
                    stdev.append( np.sqrt(postfit_sums[i]/numObs_sums[i] - mean[ictr]**2) )
                else:
                    mean.append(0.)
                    stdev.append(0.)
                ictr += 1
            ax2.errorbar(nad,mean,yerr=np.array(stdev)/2.,fmt='g-')

            ax3.plot(nad,np.sqrt(postfit_sums[siz:eiz]/numObs_sums[siz:eiz])/np.sqrt(prefit_sums[siz:eiz]/numObs_sums[siz:eiz]),'r-')
            ax2.set_ylabel('Residuals (mm)',fontsize=8)
            ax3.set_ylabel('Post/Pre ',fontsize=8)
            ax3.plot([0,14],[1, 1],'k-')
            ax.set_xlim([0,14])
            ax2.set_xlim([0,14])
            ax3.set_xlim([0,14])
            plt.tight_layout()
            ctr = ctr+1

    if args.plot or args.SATPCO:
        # PCO plots
        fig = plt.figure()
        fig.canvas.set_window_title("PCO_correction.png")
        ax = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ctr = 1
        xlabels = []
        xticks = []
        ctr = 1
        for svn in svs:
            eiz = (numParamsPerSat *ctr) -1
            ax.errorbar(ctr,Sol[eiz],yerr=np.sqrt(variances[eiz])/2.,fmt='o')
            ax2.plot(ctr,prefit_res[eiz]/numObs_sums[eiz],'bo')
            ax2.plot(ctr,postfit_res[eiz]/numObs_sums[eiz],'go')
            ax3.plot(ctr,postfit_sums[eiz]/prefit_sums[eiz],'ro')
            xlabels.append(svn)
            xticks.append(ctr)
            ctr += 1
        ax2.set_ylabel('Residuals (mm)',fontsize=8)
        ax3.set_ylabel('Post/Pre',fontsize=8)
        ax3.plot([1,ctr],[1, 1],'k-')
        
        #ax.set_xticks(xticks)
        #ax.set_xticklabels(xlabels,rotation='vertical')
        
    if args.plot or args.PCV:
        #========================================
        zen = np.linspace(0,90,numParamsPerSite)
        for snum in range(0,totalSiteModels):
            #fig = plt.figure(figsize=(3.62, 2.76))
            fig = plt.figure()
            fig.canvas.set_window_title(meta['siteIDList'][snum]+"_elevation_model.png")
            ax = fig.add_subplot(311)

            siz = numParamsPerSat*numSVS + snum * numParamsPerSite
            eiz = siz + numParamsPerSite
            ratio = np.sum(postfit_sums[siz:eiz])/np.sum(prefit_sums[siz:eiz])
            titleStr = meta['siteIDList'][snum] + " RATIO:{:.2f}".format(ratio) 
            print(titleStr)
            plt.title(titleStr,fontsize=10)

            ax.errorbar(zen,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.)

            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)

            ax2.plot(zen,prefit_res[siz:eiz]/numObs_sums[siz:eiz],'b-')
            #ax2.plot(zen,postfit_res[siz:eiz]/numObs_sums[siz:eiz],'g-')

            stdev = []  # standard deviation of postfit residual
            mean = []   # mean postfit residual
            ictr = 0
            for i in range(siz,eiz):
                if numObs_sums[i] > 1:
                    mean.append( postfit_res[i]/numObs_sums[i] )
                    stdev.append( np.sqrt(postfit_sums[i]/numObs_sums[i] - mean[ictr]**2) )
                else:
                    mean.append(0.)
                    stdev.append(0.)
                ictr += 1
            ax2.errorbar(zen,mean,yerr=np.array(stdev)/2.,fmt='g-')

            ax3.plot(zen,postfit_sums[siz:eiz]/prefit_sums[siz:eiz],'r-')
            ax3.plot([0,90],[1, 1],'k-')
            ax.set_xlabel('Zenith Angle',fontsize=8)
            ax.set_ylabel('Adjustment to PCV (mm)',fontsize=8)
            ax2.set_ylabel('Residuals (mm)',fontsize=8)
            ax3.set_ylabel('Post/Pre',fontsize=8)
            ax.set_xlim([0,90])
            ax2.set_xlim([0,90])
            ax3.set_xlim([0,90])
            
        #========================================
        ele = np.linspace(0,90,numParamsPerSite)
        fig = plt.figure()
        ax = fig.add_subplot(211)
        for snum in range(0,totalSiteModels):
            siz = numParamsPerSat*numSVS + snum * numParamsPerSite
            eiz = siz + numParamsPerSite

            ax.plot(ele,prefit_res[siz:eiz]/numObs_sums[siz:eiz],'b-')
            ax.plot(ele,postfit_res[siz:eiz]/numObs_sums[siz:eiz],'g-')

            ax.set_ylabel('Residuals (mm)',fontsize=8)
            ax.set_xlim([0,90])

        ax2 = fig.add_subplot(212)
        ctr = 0
        for svn in svs:
            siz = numParamsPerSat * ctr
            eiz = (numParamsPerSat * (ctr+1)) - 1

            ax2.plot(nad,prefit_res[siz:eiz]/numObs_sums[siz:eiz],'b-')
            ax2.plot(nad,postfit_res[siz:eiz]/numObs_sums[siz:eiz],'g-')
            ctr += 1

        ax2.set_xlim([0,14]) 

    if args.plot or args.SATPCV or args.SATPCO or args.PCV:
        plt.show()
    print("FINISHED")
