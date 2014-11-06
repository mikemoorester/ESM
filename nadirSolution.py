#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import calendar
import datetime as dt

import pprint
import pickle
import sys

import svnav

#=====================================
if __name__ == "__main__":
#    import warnings
#    warnings.filterwarnings("ignore")

    import argparse

    parser = argparse.ArgumentParser(prog='nadirSolution',description='Plot and analyase the pickle data object obatined from a nadir processing run',
            formatter_class=argparse.RawTextHelpFormatter,
            epilog='''\
    Example:

    To create a consolidated phase residual file:
    > python nadirSolution.py --model -f ./t/YAR2.2012.CL3
                   ''')

    #===================================================================
    # Station meta data options
    parser.add_argument('-f', dest='solutionFile', default='',help="Pickled solution file")
    parser.add_argument('-n', dest='nfile', default='',help="Numpy solution file")
    #===================================================================
    # Plot options
    parser.add_argument('--plot',dest='plot', default=False, action='store_true', help="Produce an elevation dependent plot of ESM phase residuals")
    parser.add_argument('--satPCO',dest='satPCO', default=False, action='store_true', help="Plot the PCO estimates")
    parser.add_argument('--satPCV',dest='satPCV', default=False, action='store_true', help="Plot the sat PCV estimates")
    parser.add_argument('--sitePCV',dest='sitePCV', default=False, action='store_true', help="Plot the site PCV estimates")
    parser.add_argument('--ps','--plot_save',dest='plot_save',default=False,action='store_true', help="Save the plots in png format")
    parser.add_argument('--about','-a',dest='about',default=False,action='store_true',help="Print meta data from solution file then exit")    
    # Debug function, not needed
    args = parser.parse_args()

    #=======================================================================================================
    #
    #       Parse pickle data structure
    #
    #=======================================================================================================
    with open(args.solutionFile,'rb') as pklID:
        meta = pickle.load(pklID)

        # Just print the meta data and exit 
        if args.about:
            pprint.pprint(meta)
            sys.exit(0)

    npzfile = np.load(args.nfile)
    Sol  = npzfile['sol']
    Cov  = npzfile['cov']
    nadir_freq = npzfile['nadirfreq']
        #meta['model'] = args.model
        #meta['nadir_grid'] = args.nadir_grid
        #meta['antex_file'] = args.antex
        #meta['svnav'] = args.svnavFile
        #meta['station_info'] = args.station_file
        #meta['zenith_grid'] = args.zen
        #meta['syyyy'] = args.syyyy
        #meta['sddd']  = args.sdoy
        #meta['eyyyy'] = args.eyyyy
        #meta['eddd']  = args.edoy
        #meta['datafiles'] = npzfiles
        #meta['svs'] = svs
        #meta['numSiteModels'] = numSites 
        #meta['siteIDList']  = siteIDList
        #meta['prechi']   = np.sqrt(prechi/numd)
        #meta['postchi']  = np.sqrt(postchi/numd)
        #meta['numd']     = numd
        #meta['chi_inc']  = np.sqrt((prechi-postchi)/numd)
        #meta['apply_constraints'] = args.apply_constraints
        #if args.apply_constraints:
        #    meta['constraint_SATPCV']  = args.constraint_SATPCV # 0.5 
        #    meta['constraint_SATPCO']  = args.constraint_SATPCO # 1.5
        #    meta['constraint_SATWIN']  = args.constraint_SATWIN # 0.5
        #    meta['constraint_SITEPCV'] = args.constraint_SITEPCV #10.0
        #    meta['constraint_SITEWIN'] = args.constraint_SITEWIN #1.5

    nad = np.linspace(0,14, int(14./meta['nadir_grid'])+1 )
    numParamsPerSat = int(14.0/meta['nadir_grid']) + 2 

    variances = np.diag(Cov)

    #============================================
    # Plot the SVN stacked residuals/correction
    #============================================
    if args.satPCV or args.plot :
        ctr = 0
        for svn in meta['svs']:
            # Now plot the distribution of the observations wrt to nadir angle
            fig = plt.figure()
            fig.canvas.set_window_title("SVN_"+svn+"_nadirCorrection.eps")
            ax = fig.add_subplot(111)

            siz = numParamsPerSat * ctr 
            eiz = (numParamsPerSat * (ctr+1)) - 1
           
            sol = Sol[siz:eiz]
            #ax.errorbar(nad,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,fmt='o')
            #ax.errorbar(nad,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,linewidth=2)
            ax.errorbar(nad,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,linewidth=2)
            ax1 = ax.twinx()
            ax1.bar(nad,nadir_freq[ctr,:],0.1,color='gray',alpha=0.75)
            ax1.set_ylabel('Number of observations',fontsize=8)

            ax.set_ylim([-4, 4])
            ax.set_xlabel('Nadir Angle (degrees)',fontsize=8)
            ax.set_ylabel('Correction to Nadir PCV (mm)',fontsize=8)

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                   ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(8)

            for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                   ax1.get_xticklabels() + ax1.get_yticklabels()):
                item.set_fontsize(8)

            plt.tight_layout()

            if args.plot_save:
                plt.savefig("SVN_"+svn+"_nadirCorrection.eps")
                plt.savefig("SVN_"+svn+"_nadirCorrection.png")
            ctr += 1
            
    #==================================================
    if args.satPCO or args.plot:
        #fig = plt.figure(figsize=(3.62, 2.76))
        fig = plt.figure()
        fig.canvas.set_window_title("PCO_correction.png")
        ax = fig.add_subplot(111)
        ctr = 1
        xlabels = []
        xticks = []
        for svn in meta['svs']:
            eiz = (numParamsPerSat *ctr) -1 
            ax.errorbar(ctr,Sol[eiz],yerr=np.sqrt(variances[eiz])/2.,fmt='o')
            xlabels.append(svn)
            xticks.append(ctr)
            ctr += 1

        ax.set_xlabel('SVN',fontsize=8)
        ax.set_ylabel('Adjustment to PCO (mm)',fontsize=8)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels,rotation='vertical')

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
               ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)
        plt.tight_layout()

        if args.plot_save:
            plt.savefig("PCO_correction.png")

    if args.sitePCV or args.plot :
        ctr = 0
        numSVS = np.size(meta['svs'])
        numNADS = int(14.0/meta['nadir_grid']) + 1 
        numParamsPerSat = numNADS + 1
        totalSiteModels = meta['numSiteModels']
        numParamsPerSite = int(90./meta['zenith_grid']) + 1
        numParams = numSVS * (numParamsPerSat) + numParamsPerSite * totalSiteModels 

        for snum in range(0,totalSiteModels):
            fig = plt.figure(figsize=(3.62, 2.76))
            fig.canvas.set_window_title(meta['siteIDList'][snum]+"_elevation_model.png")
            ax = fig.add_subplot(111)

            siz = numParamsPerSat*numSVS + snum * numParamsPerSite 
            eiz = siz + numParamsPerSite 
            print("plotting: ",meta['siteIDList'][snum],snum,np.shape(Sol),siz,eiz)

            ele = np.linspace(0,90,numParamsPerSite)
            ax.errorbar(ele,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.)

            ax.set_xlabel('Zenith Angle',fontsize=8)
            ax.set_ylabel('Adjustment to PCV (mm)',fontsize=8)
            ax.set_xlim([0,90])

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(8)

            plt.tight_layout()
            if args.plot_save:
                plt.savefig(meta['siteIDList'][snum]+"_elevation_model.png")

    del Cov,Sol 

    if args.plot or args.sitePCV or args.satPCO or args.satPCV:
        plt.show()

    print("FINISHED")
