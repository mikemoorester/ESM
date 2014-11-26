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

def plotFontSize(ax,fontsize=8):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)

    return ax

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
    parser.add_argument('--about','-a',dest='about',default=False,action='store_true',help="Print meta data from solution file then exit")    
    #===================================================================
    # Station meta data options
    #===================================================================
    parser.add_argument('-f','--f1', dest='solutionFile', default='',help="Pickled solution file")
    parser.add_argument('-n', dest='nfile', default='',help="Numpy solution file")
    #===================================================================
    # Plot options
    #===================================================================
    parser.add_argument('--plot',dest='plot', default=False, action='store_true', help="Produce an elevation dependent plot of ESM phase residuals")
    parser.add_argument('--SATPCO',dest='satPCO', default=False, action='store_true', help="Plot the PCO estimates")
    parser.add_argument('--SATPCV',dest='satPCV', default=False, action='store_true', help="Plot the sat PCV estimates")
    parser.add_argument('--SITEPCV',dest='sitePCV', default=False, action='store_true', help="Plot the site PCV estimates")
    parser.add_argument('--corr',dest='corr', default=False, action='store_true', help="Plot the covariance matrix")
    parser.add_argument('--ps','--plot_save',dest='plot_save',default=False,action='store_true', help="Save the plots in png format")

    #===================================================================
    # Compare Solutions 
    #===================================================================
    parser.add_argument('--compare',dest='compare',default=False,action='store_true',help="Compare two solutions")
    parser.add_argument('--f2', dest='comp2', default='',help="Pickled solution file")

    parser.add_argument('--mean',dest='mean',default=False,action='store_true',help="Compute the mean solution")
    parser.add_argument('--f3', dest="comp3", default='',help="Path to Pickled solution file")
    parser.add_argument('--f4', dest="comp4", default='',help="Path to Pickled solution file")
    parser.add_argument('--f5', dest="comp5", default='',help="Path to Pickled solution file")
    # Debug function, not needed
    args = parser.parse_args()

    if len(args.nfile) < 1 :
        args.nfile = args.solutionFile + ".sol.npz"

    args.compare_nfile = args.comp2 + ".sol.npz"
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

    if not args.mean:
        npzfile = np.load(args.nfile)
        Sol  = npzfile['sol']
        Cov  = npzfile['cov']
        nadir_freq = npzfile['nadirfreq']
        variances = np.diag(Cov)
    
    if args.corr:
        fig = plt.figure()
        #fig.canvas.set_window_title("SVN_"+svn+"_nadirCorrectionComparison")
        ax   = fig.add_subplot(111)

        ax.pcolor(Cov)
        #fig.colorbar()

    elif args.compare:
        compare_npzfile     = np.load(args.compare_nfile)
        compare_Sol         = compare_npzfile['sol']
        compare_Cov         = compare_npzfile['cov']
        compare_nadir_freq  = compare_npzfile['nadirfreq']
        compare_variances   = np.diag(compare_Cov)

    nad = np.linspace(0,14, int(14./meta['nadir_grid'])+1 )
    numParamsPerSat = int(14.0/meta['nadir_grid']) + 2 


    #============================================
    # Plot the SVN stacked residuals/correction
    #============================================
    if args.satPCV or args.plot :
        ctr = 0
        if args.compare:
            for svn in meta['svs']:
                # Now plot the distribution of the observations wrt to nadir angle
                fig = plt.figure()
                fig.canvas.set_window_title("SVN_"+svn+"_nadirCorrectionComparison")
                ax   = fig.add_subplot(211)
                ax2  = fig.add_subplot(212)

                siz = numParamsPerSat * ctr 
                eiz = (numParamsPerSat * (ctr+1)) - 1
           
                sol = Sol[siz:eiz]
                ax.errorbar(nad,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,linewidth=2,fmt='b')
                #ax.plot(nad,Sol_compare[siz:eiz],linewidth=2,'b-')

                ax1 = ax.twinx()
                ax1.bar(nad,nadir_freq[ctr,:],0.1,color='gray',alpha=0.75)
                ax1.set_ylabel('Number of observations',fontsize=8)

                ax.errorbar(nad,compare_Sol[siz:eiz],yerr=np.sqrt(compare_variances[siz:eiz])/2.,linewidth=2,fmt='k')

                ax.set_xlabel('Nadir Angle (degrees)',fontsize=8)
                ax.set_ylabel('Correction to Nadir PCV (mm)',fontsize=8)

                ax = plotFontSize(ax,8)
                ax1 = plotFontSize(ax1,8)
                ax.legend(['File1','File 2'],fontsize=8)
                diff = Sol[siz:eiz] - compare_Sol[siz:eiz]
                ax2.plot(nad,diff,'r-',linewidth=2)

                plt.tight_layout()

                if args.plot_save:
                    plt.savefig("SVN_"+svn+"_nadirCorrection.eps")
                ctr = ctr + 1

        elif args.mean:
            legend = []
            solFlag = np.zeros(5)
            tSat = numParamsPerSat * np.size(meta['svs']) 

            # work out how many solutions we are looking at
            sctr = 0
            if len(args.solutionFile) > 1: sctr+= 1
            if len(args.comp2) > 1: sctr += 1
            if len(args.comp3) > 1: sctr += 1
            if len(args.comp4) > 1: sctr += 1
            if len(args.comp5) > 1: sctr += 1

            SOL = np.zeros((tSat,sctr))
            STD = np.zeros((tSat,sctr))

            legend.append("Mean")
            sctr = 0

            if len(args.solutionFile) > 1:
                nfile = args.nfile
                solFlag[0] = 1
                legend.append("Soln_1")
                npzfile_t    = np.load(nfile)
                Sol_t        = npzfile_t['sol']
                Cov_t        = npzfile_t['cov']
                nadir_freq_t = npzfile_t['nadirfreq']
                variances_t  = np.diag(Cov_t)
                SOL[:,sctr] = Sol_t[0:tSat]
                STD[:,sctr] = np.sqrt(variances_t[0:tSat])
                sctr += 1
                del npzfile_t, Sol_t, Cov_t, nadir_freq_t, variances_t
            if len(args.comp2) > 1:
                nfile = args.comp2+".sol.npz"
                solFlag[1] = 1
                legend.append("Soln_2")
                npzfile_t    = np.load(nfile)
                Sol_t        = npzfile_t['sol']
                Cov_t        = npzfile_t['cov']
                nadir_freq_t = npzfile_t['nadirfreq']
                variances_t  = np.diag(Cov_t)
                SOL[:,sctr] = Sol_t[0:tSat]
                STD[:,sctr] = np.sqrt(variances_t[0:tSat])
                sctr += 1
                del npzfile_t, Sol_t, Cov_t, nadir_freq_t, variances_t
            if len(args.comp3) > 1:
                nfile = args.comp3+".sol.npz"
                solFlag[2] = 1
                legend.append("Soln_3")
                npzfile_t    = np.load(nfile)
                Sol_t        = npzfile_t['sol']
                Cov_t        = npzfile_t['cov']
                nadir_freq_t = npzfile_t['nadirfreq']
                variances_t  = np.diag(Cov_t)
                SOL[:,sctr] = Sol_t[0:tSat]
                STD[:,sctr] = np.sqrt(variances_t[0:tSat])
                sctr += 1
                del npzfile_t, Sol_t, Cov_t, nadir_freq_t, variances_t
            if len(args.comp4) > 1:
                nfile = args.comp4+".sol.npz"
                solFlag[3] = 1
                legend.append("Soln_4")
                npzfile_t    = np.load(nfile)
                Sol_t        = npzfile_t['sol']
                Cov_t        = npzfile_t['cov']
                nadir_freq_t = npzfile_t['nadirfreq']
                variances_t  = np.diag(Cov_t)
                SOL[:,sctr] = Sol_t[0:tSat]
                STD[:,sctr] = np.sqrt(variances_t[0:tSat])
                sctr += 1
                del npzfile_t, Sol_t, Cov_t, nadir_freq_t, variances_t
            if len(args.comp5) > 1:
                nfile = args.comp5+".sol.npz"
                solFlag[4] = 1
                legend.append("Soln_5")
                npzfile_t    = np.load(nfile)
                Sol_t        = npzfile_t['sol']
                Cov_t        = npzfile_t['cov']
                nadir_freq_t = npzfile_t['nadirfreq']
                variances_t  = np.diag(Cov_t)
                SOL[:,sctr] = Sol_t[0:tSat]
                STD[:,sctr] = np.sqrt(variances_t[0:tSat])
                sctr += 1
                del npzfile_t, Sol_t, Cov_t, nadir_freq_t, variances_t

            fig = plt.figure()
            ax_all = fig.add_subplot(111)

            for svn in meta['svs']:
                # Now plot the distribution of the observations wrt to nadir angle
                fig = plt.figure()
                fig.canvas.set_window_title("SVN_"+svn+"_nadirCorrectionComparison")
                ax   = fig.add_subplot(211)
                ax2  = fig.add_subplot(212)

                siz = numParamsPerSat * ctr 
                eiz = (numParamsPerSat * (ctr+1)) - 1
         
                # PLot the mean 
                mean_val = np.zeros(numParamsPerSat)
                #mean_val = np.sum(SOL[siz:eiz-2,:],axis=1)/np.sum(solFlag)
                mean_val = np.mean(SOL[siz:eiz-2,:],axis=1)
                std_val = np.std(SOL[siz:eiz-2,:],axis=1)
              
                ax.errorbar(nad[:-2],mean_val,yerr=std_val/2.,fmt='k--',linewidth=3)
                ax_all.errorbar(nad[:-2],mean_val,yerr=std_val/2.)
                #ax_all.plot(nad[:-2],mean_val)

                solCTR = 0

                for flag in solFlag:
                    if flag == 0:
                    #    solCTR += 1
                        continue

                    ax.errorbar(nad[:-2],SOL[siz:eiz-2,solCTR],yerr=STD[siz:eiz-2,solCTR]/2.) #,linewidth=2)

                    diff = mean_val - SOL[siz:eiz-2,solCTR]
                    ax2.plot(nad[:-2],diff,'-',linewidth=2)
                    solCTR += 1

                ax.set_xlabel('Nadir angle (degrees)',fontsize=8)
                ax.set_ylabel('Correction to nadir PCV (mm)',fontsize=8)
                ax.plot([0,13.8],[0,0],'k-')
                ax2.plot([0,13.8],[0,0],'k-')
                ax.set_xlim([0,13.8])
                ax2.set_xlim([0,13.8])

                ax.set_xlabel('Nadir angle (degrees)',fontsize=8)
                ax2.set_ylabel('Difference from mean nadir correction (mm)',fontsize=8)

                ax = plotFontSize(ax,8)
                ax2 = plotFontSize(ax2,8)
                ax.legend(legend,fontsize=8)

                plt.tight_layout()

                if args.plot_save:
                    plt.savefig("SVN_"+svn+"_nadirCorrection.eps")
                ctr = ctr + 1

        else:
            fig = plt.figure()
            fig.canvas.set_window_title("All SVNs")
            ax_all = fig.add_subplot(111)

            for svn in meta['svs']:
                # Now plot the distribution of the observations wrt to nadir angle
                fig = plt.figure()
                fig.canvas.set_window_title("SVN_"+svn+"_nadirCorrection.eps")
                ax = fig.add_subplot(111)

                siz = numParamsPerSat * ctr 
                eiz = (numParamsPerSat * (ctr+1)) - 1
           
                sol = Sol[siz:eiz]
                ax.errorbar(nad,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,linewidth=2)
                ax_all.errorbar(nad,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,linewidth=2)
                ax1 = ax.twinx()
                ax1.bar(nad,nadir_freq[ctr,:],0.1,color='gray',alpha=0.75)
                ax1.set_ylabel('Number of observations',fontsize=8)

                #ax.set_ylim([-4, 4])
                ax.set_xlabel('Nadir Angle (degrees)',fontsize=8)
                ax.set_ylabel('Correction to Nadir PCV (mm)',fontsize=8)

                ax = plotFontSize(ax,8)
                ax1 = plotFontSize(ax1,8)

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
            #fig = plt.figure(figsize=(3.62, 2.76))
            fig = plt.figure()
            fig.canvas.set_window_title(meta['siteIDList'][snum]+"_elevation_model.png")

            if args.compare:
                ax = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
            else:
                ax = fig.add_subplot(111)

            siz = numParamsPerSat*numSVS + snum * numParamsPerSite 
            eiz = siz + numParamsPerSite 
            print("plotting: ",meta['siteIDList'][snum],snum,np.shape(Sol),siz,eiz)

            zen = np.linspace(0,90,numParamsPerSite)
            ax.errorbar(zen,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,fmt='b-')

            # PLot on the number of observations
            #ax = ax.twinx()
            #ax.bar(zen,nadir_freq[ctr,:],0.1,color='gray',alpha=0.75)
            #ax.set_ylabel('Number of observations',fontsize=8)

            if args.compare:
                ax.errorbar(zen,compare_Sol[siz:eiz],yerr=np.sqrt(compare_variances[siz:eiz])/2.,fmt='k-')

                diff = Sol[siz:eiz] - compare_Sol[siz:eiz]
                ax2.plot(zen,diff,'r-',linewidth=2)

            ax.set_xlabel('Zenith Angle',fontsize=8)
            ax.set_ylabel('Adjustment to PCV (mm)',fontsize=8)
            ax.set_xlim([0,90])

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(8)

            plt.tight_layout()
            if args.plot_save:
                plt.savefig(meta['siteIDList'][snum]+"_elevation_model.png")

    #del Cov,Sol 

    if args.compare:
        del compare_Cov,compare_Sol 

    if args.plot or args.sitePCV or args.satPCO or args.satPCV or args.corr:
        plt.show()

    print("FINISHED")
