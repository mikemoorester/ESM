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
    #===================================================================
    # Plot options
    parser.add_argument('--plot',dest='plot', default=False, action='store_true', help="Produce an elevation dependent plot of ESM phase residuals")
    parser.add_argument('--ps','--plot_save',dest='savePlots',default=False,action='store_true', help="Save the plots in png format")
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
        if args.about:
            pprint.pprint(meta)
            sys.exit(0)
        Sol = pickle.load(pklID)
        Cov = pickle.load(pklID)
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
        #pickle.dump(meta,pklID)
        #pickle.dump(Sol,pklID)
        #pickle.dump(Cov,pklID)

    nad = np.linspace(0,14, int(14./meta['nadir_grid'])+1 )
    numParamsPerSat = int(14.0/meta['nadir_grid']) + 2 

    #variances = np.diag(Neq)
    #print("Variance:",np.shape(variances))

    #============================================
    # Plot the sparsity of the matrix Neq
#   fig = plt.figure()
#   ax = fig.add_subplot(111)
#   ax.spy(Neq, precision=1e-3, marker='.', markersize=5)

#   ctr = 0
#   xlabels = []
#   xticks = []
#   for svn in svs:
#       siz = numParamsPerSat * ctr 
#       eiz = numParamsPerSat *ctr + numNADS 
#       ctr = ctr + 1
#       xlabels.append(svn)
#       tick = int((eiz-siz)/2)+siz
#       xticks.append(tick)

        
#   for snum in range(0,totalSiteModels):
#       siz = numParamsPerSat*numSVS + snum * numParamsPerSite 
#       eiz = siz + numParamsPerSite 
#       xlabels.append(siteIDList[snum])
#       tick = int((eiz-siz)/2)+siz
#       xticks.append(tick)

#   ax.set_xticks(xticks)
#   ax.set_xticklabels(xlabels,rotation='vertical')
#   for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#                      ax.get_xticklabels() + ax.get_yticklabels()):
#           item.set_fontsize(6)

#   plt.tight_layout()
#   if args.savePlots:
#       plt.savefig("NeqMatrix_with_constraints.eps")

#   del Neq, AtWb

#   #============================================
#   # Plot the SVN stacked residuals/correction
#   #============================================
#   ctr = 0
#   for svn in svs:
#       #fig = plt.figure(figsize=(3.62, 2.76))
#       fig = plt.figure()
#       fig.canvas.set_window_title("SVN_"+svn+"_nadirCorrectionModel.png")
#       ax = fig.add_subplot(212)
#       ax1 = fig.add_subplot(211)

#       siz = numParamsPerSat * ctr 
#       eiz = numParamsPerSat *ctr + numNADS 
#          
#       sol = Sol[siz:eiz]
#       #print("SVN:",svn,siz,eiz,numParamsPerSat,tSat)
#       #ax1.plot(nad,Sol[siz:eiz],'r-',linewidth=2)
#       ax1.plot(nad,sol[::-1],'r-',linewidth=2)
#       #ax.errorbar(nad,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,fmt='o')
#       ax.errorbar(nad,sol[::-1],yerr=np.sqrt(variances[siz:eiz])/2.,fmt='o')

#       #print(svn,Sol[siz:eiz],np.sqrt(variances[siz:eiz])/2.)
#       ax.set_xlabel('Nadir Angle (degrees)',fontsize=8)
#       ax.set_ylabel('Phase Residuals (mm)',fontsize=8)
#       #ax.set_xlim([0, 14])

#       for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#                  ax.get_xticklabels() + ax.get_yticklabels()):
#           item.set_fontsize(8)

#       plt.tight_layout()

#       for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
#                  ax1.get_xticklabels() + ax1.get_yticklabels()):
#               item.set_fontsize(8)

#       plt.tight_layout()
#       if args.savePlots:
#           plt.savefig("SVN_"+svn+"_nadirCorrectionModel.png")
#       ctr += 1
                
        #if ctr > 2:
        #    break

#       #==================================================
#       #fig = plt.figure(figsize=(3.62, 2.76))
#       fig = plt.figure()
#       fig.canvas.set_window_title("PCO_correction.png")
#       #ax = fig.add_subplot(111)
#       ax = fig.add_subplot(212)
#       ax1 = fig.add_subplot(211)
#       ctr = 0
#       numSVS = np.size(svs)
#       numNADS = int(14.0/args.nadir_grid) + 1 
#       numParamsPerSat = numNADS + PCOEstimates
#       print("Number of Params per Sat:",numParamsPerSat,"numNads",numNADS,"Sol",np.shape(Sol))
#       for svn in svs:
#           eiz = numParamsPerSat *ctr + numParamsPerSat -1 
#           #print(ctr,"PCO:",eiz)
#           ax1.plot(ctr,Sol[eiz],'k.',linewidth=2)
#           #print(svn,Sol[eiz],np.sqrt(variances[eiz])/2.)
#           #ax.subplot(212)
#           ax.errorbar(ctr,Sol[eiz],yerr=np.sqrt(variances[eiz])/2.,fmt='o')
#           ctr += 1

#       ax.set_xlabel('SVN',fontsize=8)
#       ax.set_ylabel('Adjustment to PCO (mm)',fontsize=8)
#       #ax.set_xlim([0, 14])

#       for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#                  ax.get_xticklabels() + ax.get_yticklabels()):
#           item.set_fontsize(8)
#       plt.tight_layout()

#       for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
#                  ax1.get_xticklabels() + ax1.get_yticklabels()):
#           item.set_fontsize(8)
#       plt.tight_layout()

#       if args.savePlots:
#           plt.savefig("PCO_correction.png")

#       #==================================================
#       if args.model == 'pwlSite' or args.model == 'pwlSiteDaily':
#           ctr = 0
#           numSVS = np.size(svs)
#           numNADS = int(14.0/args.nadir_grid) + 1 
#           numParamsPerSat = numNADS + PCOEstimates
#           print("Number of Params per Sat:",numParamsPerSat,"numNads",numNADS,"Sol",np.shape(Sol),"TotalSites:",totalSiteModels)
#           numParams = numSVS * (numParamsPerSat) + numParamsPerSite * totalSiteModels 
#           for snum in range(0,totalSiteModels):
#               #fig = plt.figure(figsize=(3.62, 2.76))
#               fig = plt.figure()
#               fig.canvas.set_window_title(siteIDList[snum]+"_elevation_model.png")
#               ax = fig.add_subplot(212)
#               ax1 = fig.add_subplot(211)
#               siz = numParamsPerSat*numSVS + snum * numParamsPerSite 
#               eiz = siz + numParamsPerSite 
#               ele = np.linspace(0,90,numParamsPerSite)
#               #print("Sol",np.shape(Sol),"siz  ",siz,eiz)
#               ax1.plot(ele,Sol[siz:eiz],'k.',linewidth=2)
#               ax.errorbar(ele,Sol[siz:eiz],yerr=np.sqrt(variances[siz:eiz])/2.,fmt='o')
#               #print(svn,Sol[siz:eiz],np.sqrt(variances[siz:eiz])/2.)

#               ax.set_xlabel('Zenith Angle',fontsize=8)
#               ax.set_ylabel('Adjustment to PCV (mm)',fontsize=8)
#               #ax.set_xlim([0, 14])

#               for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
#                               ax1.get_xticklabels() + ax1.get_yticklabels()):
#                   item.set_fontsize(8)
#               plt.tight_layout()

#               for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#                   ax.get_xticklabels() + ax.get_yticklabels()):
#                   item.set_fontsize(8)

#               plt.tight_layout()
#               if args.savePlots:
#                   plt.savefig(siteIDList[snum]+"_elevation_model.png")
#               
#       if args.plot:
#           plt.show()

    print("FINISHED")
