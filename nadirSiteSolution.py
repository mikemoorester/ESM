#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
#import calendar
#import datetime as dt

#import pprint
#import pickle
import sys

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

    parser = argparse.ArgumentParser(prog='nadirSiteSolution',description='Plot and analyase the pickle data object obatined from a nadir processing run',
            formatter_class=argparse.RawTextHelpFormatter,
            epilog='''\
    Example:

    To create a consolidated phase residual file:
    > python nadirSolution.py --model -f ./t/YAR2.2012.CL3
                   ''')

    #===================================================================
    parser.add_argument('--about','-a',dest='about',default=False,action='store_true',help="Print meta data from solution file then exit")    
    #===================================================================
    parser.add_argument('-f','--f1', dest='solutionFile', default='',help="Pickled solution file")
    parser.add_argument('-n', dest='nfile', default='',help="Numpy solution file")
    parser.add_argument('--pf',dest='post_fit',default=False,action='store_true',help="Plot post fit residuals")
    #===================================================================
    # Plot options
    #===================================================================
    parser.add_argument('--plot',dest='plot', default=False, action='store_true', help="Produce an elevation dependent plot of ESM phase residuals")
    parser.add_argument('--SITEPCV',dest='sitePCV', default=False, action='store_true', help="Plot the site PCV estimates")
    parser.add_argument('--ps','--plot_save',dest='plot_save',default=False,action='store_true', help="Save the plots in png format")

    #===================================================================
    # Compare Solutions 
    #===================================================================
    parser.add_argument('--compare',dest='compare',default=False,action='store_true',help="Compare two solutions")
    parser.add_argument('--f2', dest='comp2', default='',help="Pickled solution file")

    # Debug function, not needed
    args = parser.parse_args()

    #if len(args.nfile) < 1 :
    #    args.nfile = args.solutionFile + ".sol.npz"

    #args.compare_nfile = args.comp2 + ".sol.npz"
    #=======================================================================================================
    #
    #       Parse pickle data structure
    #
    #=======================================================================================================
#   with open(args.solutionFile,'rb') as pklID:
#       meta = pickle.load(pklID)

#       # Just print the meta data and exit 
#       if args.about:
#           pprint.pprint(meta)
#           sys.exit(0)
#    if args.post_fit:
#        npzfile = np.load(args.nfile)
        
#        prefit       = npzfile['prefit']
#        prefit_sums  = npzfile['prefit_sums']
#        prefit_res   = npzfile['prefit_res']
        
#        postfit      = npzfile['postfit']
#        postfit_sums = npzfile['postfit_sums']
#        postfit_res  = npzfile['postfit_res']
        
#        numObs = npzfile['numObs']
#        numObs_sums = npzfile['numObs_sums']
#        fig = plt.figure()
#        #fig.canvas.set_window_title("All SVNs")
#        ax = fig.add_subplot(111)
#        ax.plot(nad,np.sqrt(postfit_sums[siz:eiz]/numObs_sums[siz:eiz])/np.sqrt(prefit_sums[siz:eiz]/numObs_sums[siz:eiz]),'r-')     
#        plt.show()
#        sys.exit(0)
             
    npzfile = np.load(args.nfile)
    model  = npzfile['model']
    stdev  = npzfile['stdev']
    site_freq  = npzfile['site_freq']
    ele_model  = npzfile['ele_model']
    ele_stdev  = npzfile['ele_model_stdev']
    ele_site_freq  = npzfile['ele_site_freq']
    
    #if args.compare:
    #    compare_npzfile     = np.load(args.compare_nfile)
    #    compare_Sol         = compare_npzfile['sol']
    #    compare_Cov         = compare_npzfile['cov']
    #    compare_nadir_freq  = compare_npzfile['nadirfreq']
    #    compare_variances   = np.diag(compare_Cov)

    #zen = np.linspace(0,90, int(90./meta['zen_grid'])+1 )
    #az  = np.linspace(0,360. - meta['zen_grid'], int(360./meta['zen_grid']) )

    print("Shape of model:",np.shape(model))
    zen = np.linspace(0,90, np.shape(model)[1] )
    print("zen:",zen,np.shape(model)[1])
    az  = np.linspace(0,360. - 360./np.shape(model)[0], np.shape(model)[0] )
    print("az:",az,np.shape(model)[0])

    #============================================
    # Plot the Elevation depndent phase residual corrections
    #============================================
    fig = plt.figure()
    #fig.canvas.set_window_title("All SVNs")
    ax = fig.add_subplot(111)
    ax.errorbar(zen,ele_model[0,:],yerr=ele_stdev[0,:]/2.,linewidth=2)

    ax1 = ax.twinx()
    ax1.bar(zen,ele_site_freq[0,:],0.1,color='gray',alpha=0.75)
    ax1.set_ylabel('Number of observations',fontsize=8)

    ax.set_xlabel('Zenith angle (degrees)',fontsize=8)
    ax.set_ylabel('Correction to PCV (mm)',fontsize=8)

    ax = plotFontSize(ax,8)
    ax1 = plotFontSize(ax1,8)

    plt.tight_layout()

    #============================================
    fig = plt.figure()
    #fig.canvas.set_window_title("All SVNs")
    ax = fig.add_subplot(111)

    for i in range(0,np.size(az)):
        for j in range(0,np.size(zen)): 
            ax.errorbar(zen[j],model[i,j],yerr=np.sqrt(stdev[i,j])/2.,linewidth=2)
            #ax.plot(zen[j],model[i,j],'b.')
            #ax1 = ax.twinx()
            #ax1.bar(nad,nadir_freq[ctr,:],0.1,color='gray',alpha=0.75)
            #ax1.set_ylabel('Number of observations',fontsize=8)

    ax.set_xlabel('Zenith angle (degrees)',fontsize=8)
    ax.set_ylabel('Correction to PCV (mm)',fontsize=8)

    ax = plotFontSize(ax,8)

    plt.tight_layout()

    #============================================
    # Do a polar plot 
    #============================================
    fig = plt.figure()
    #fig.canvas.set_window_title("All SVNs")
    ax = fig.add_subplot(111,polar='true')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.radians(90.))
    ax.set_ylim([0,1])
    ax.set_rgrids((0.00001, np.radians(20.)/np.pi*2, np.radians(40.)/np.pi*2,np.radians(60.)/np.pi*2,np.radians(80.)/np.pi*2),labels=('0', '20', '40', '60', '80'),angle=180)

    ma,mz = np.meshgrid(az,zen,indexing='ij')
    ma = ma.reshape(ma.size,)
    mz = mz.reshape(mz.size,)

    polar = ax.scatter(np.radians(ma), np.radians(mz)/np.pi*2., c=model[:,:], s=50, alpha=1., cmap=cm.RdBu,vmin=-15,vmax=15, lw=0)
    cbar = plt.colorbar(polar,shrink=0.75,pad=.10)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('ESM (mm)',size=8)
    ax = plotFontSize(ax,8)
    plt.tight_layout()

    fig = plt.figure()
    #fig.canvas.set_window_title("All SVNs")
    ax = fig.add_subplot(111,polar='true')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.radians(90.))
    ax.set_ylim([0,1])
    ax.set_rgrids((0.00001, np.radians(20.)/np.pi*2, np.radians(40.)/np.pi*2,np.radians(60.)/np.pi*2,np.radians(80.)/np.pi*2),labels=('0', '20', '40', '60', '80'),angle=180)

    polar = ax.scatter(np.radians(ma), np.radians(mz)/np.pi*2., c=stdev[:,:], s=50, alpha=1., cmap=cm.RdBu,vmin=-15,vmax=15, lw=0)
    cbar = plt.colorbar(polar,shrink=0.75,pad=.10)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Standard Deviation (mm)',size=8)
    ax = plotFontSize(ax,8)
    plt.tight_layout()

    plt.show()
    print("FINISHED")
