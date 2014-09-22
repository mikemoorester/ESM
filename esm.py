#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import re
import gzip
import calendar

from scipy import interpolate
from scipy.stats.stats import nanmean, nanmedian, nanstd

import antenna as ant
import residuals as res
import gpsTime as gt
import GamitStationFile as gsf
import time
import svnav

import os, sys
import datetime as dt


def reject_outliers(data, m=5):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def reject_abs(data, val):
    return data[abs(data) < val]

def reject_outliers_arg(data,nSigma):
    """
    Do a simple outlier removal at 3 sigma, with two passes over the data
    """
    criterion = ( (data[:] < (data[:].mean() + data[:].std() * nSigma)) &
                  (data[:] > (data[:].mean() - data[:].std() * nSigma)) )
    ind = np.array(np.where(criterion))[0]
    
    return ind

def reject_outliers_byelevation_arg(data,nSigma,zenSpacing=0.5):
    zen = np.linspace(0,90,int(90./zenSpacing)+1)
    tmp = []
    for z in zen:
        criterion = ( (data[:,2] < (z + zenSpacing/2.)) & 
                      (data[:,2] > (z - zenSpacing/2.)) ) 
        ind = np.array(np.where(criterion))[0]
        rout = reject_outliers_arg(data[ind,3],nSigma)
        tmp.append(rout.tolist()) 
    return tmp 

def reject_outliers_elevation(data,nSigma,zenSpacing=0.5):
    zen = np.linspace(0,90,int(90./zenSpacing)+1)
    for z in zen:
        criterion = ( (data[:,0] < (z + zenSpacing/2.)) & 
                      (data[:,0] > (z - zenSpacing/2.)) ) 
        ind = np.array(np.where(criterion))
        ind = ind.reshape((np.size(ind),))

        tdata = np.zeros((np.size(ind),3))
        tdata = data[ind,:]

        din = data[ind,2].reshape((np.size(ind),))
        rout = reject_outliers_arg(din,5)

        # if its the first interation initialise tmp
        if z < zenSpacing:
            tmp = tdata[rout,:]
        else:
            tmp = np.vstack((tmp,tdata[rout,:]))

    return tmp 

def blockMedian(data, azSpacing=0.5,zenSpacing=0.5):
    """
        bMedian = blockMedian(residuals,args, idSpacing)

        where,

            residuals => f
        
            gridSpacing  => float ie every 0.5 degrees
            
        output:
            bType 1 => bMedian is a 2d matrix of shape [nAz,nZd]

            bMedian = blockMedian('./MOBS_DPH_2012_001_143.all',0.5)

            bMedian.shape => 721,181
    """
    az = np.linspace(0,360.,int(360./azSpacing)+1)
    zz = np.linspace(0,90,int(90./zenSpacing)+1)

    azCtr = 0
    iCtr = 0
    bMedian = np.zeros((az.size,zz.size))
    bMedianStd = np.zeros((az.size,zz.size))

    for i in az:
        if(i - azSpacing/2. < 0) :
            criterion = (data[:,0] < (i + azSpacing/2.)) | (data[:,0] > (360. - azSpacing/2.) )
        else:
            criterion = (data[:,0] < (i + azSpacing/2.)) & (data[:,0] > (i - azSpacing/2.) )

        ind = np.array(np.where(criterion))[0]
        #print("Azimuth,ind:",np.shape(ind),ind) 
        jCtr = 0
        if ind.size == 0:
            for j in zz :
                bMedian[iCtr,jCtr] = float('NaN')
                jCtr += 1
        else: 
            tmp = data[ind,:]
            for j in zz:
                # disregard any observation above 80 in zenith, too noisy
                if j >= 80:
                    bMedian[iCtr,jCtr] = float('NaN')
                else:
                    criterion = (tmp[:,1] < (j + zenSpacing/2.)) & (tmp[:,1] > (j - zenSpacing/2.) ) 
                    indZ = np.array(np.where(criterion))[0]
                    if indZ.size > 3 :
                        bMedian[iCtr,jCtr] = nanmedian( reject_outliers(reject_abs( tmp[indZ,2],70. ),5.))
                        bMedianStd[iCtr,jCtr] = nanstd(reject_outliers(reject_abs( tmp[indZ,2],70. ),5.))
                    else:
                        bMedian[iCtr,jCtr] = float('NaN') 

                jCtr += 1
        iCtr += 1

    return bMedian, bMedianStd

def interpolate_eleMean(model):
    """ Salim --- When the residuals are NaN, replace them with the mean of 
    # the all the data at the same elevation
    """
    # Get mean of columns (data at the same elevation) without taking int account NaNs
    el_mean = nanmean(model,axis=0)
    #print(el_mean) 
    # Find indices for NaNs, and replace them by the column mean
    ind_nan = np.where(np.isnan(model))
    model[ind_nan] = np.take(el_mean,ind_nan[1])

    return model

def modelStats(model,data, azSpacing=0.5,zenSpacing=0.5):
    """
        bMedian = blockMedian(residuals,gridSpacing)

        where,

            residuals => f
        
            gridSpacing  => float ie every 0.5 degrees
            
        output:
            bType 1 => bMedian is a 2d matrix of shape [nAz,nZd]

        Example:

            bMedian = blockMedian('./MOBS_DPH_2012_001_143.all',0.5)

            bMedian.shape => 721,181
    """
    az = np.linspace(0,360.,int(360./azSpacing)+1)
    zz = np.linspace(0,90,int(90./zenSpacing)+1)

    azCtr = 0
    iCtr = 0
    chi = 0
    SS_tot = 0
    SS_res = 0
    SS_reg = 0
    test_ctr = 0
    reg_ctr = 0

    for i in az:
        if(i - azSpacing/2. < 0) :
            criterion = (data[:,0] < (i + azSpacing/2.)) | (data[:,0] > (360. - azSpacing/2.) )
        else:
            criterion = (data[:,0] < (i + azSpacing/2.)) & (data[:,0] > (i - azSpacing/2.) )

        ind = np.array(np.where(criterion))
 
        if ind.size > 0:
            tmp = data[ind,:]
            tmp = tmp.reshape(tmp.shape[1],tmp.shape[2])
            jCtr = 0
            for j in zz:
                # disregard any observation above 80 in zenith, too noisy
                if j < 80. + zenSpacing:
                    criterion = (tmp[:,1] < (j + zenSpacing/2.)) & (tmp[:,1] > (j - zenSpacing/2.) ) 
                    indZ = np.array(np.where(criterion))
                    tmpZ = np.array( tmp[indZ[:],2] )
                    if indZ.size > 3 and not np.isnan(model[iCtr,jCtr]): # and (model[iCtr,jCtr] > 0.00001 or model[iCtr,jCtr] < -0.00001):
                        test_data = reject_outliers(reject_abs( tmp[indZ,2],70. ),5.)
                        #print(i,j,test_data,model[iCtr,jCtr])
                        if test_data.size > 0:
                            y_mean = np.mean(test_data)
                            SS_reg += (model[iCtr,jCtr] - y_mean)**2
                            reg_ctr += 1
                            for obs in test_data:
                                #chi += (obs - model[iCtr,jCtr]) ** 2 / model[iCtr,jCtr]
                                SS_tot += (obs - y_mean) ** 2
                                SS_res += (obs - model[iCtr,jCtr])**2
                                test_ctr += 1
                jCtr += 1
        iCtr += 1

    rr = 1. - SS_res/SS_tot

    rms = np.sqrt(SS_res) * 1./test_ctr

    # gives an indication of how different the models would be between the test and training data set 
    rms2 = np.sqrt(SS_reg) * 1./reg_ctr

    # Used in matlab instead of rr
    norm_res = np.sqrt(SS_res)
    mse = 1./(2.*test_ctr) * SS_res
    return mse,rr,rms,rms2

# calculate the cost function, that is the Mean Square Error
def calcMSE(model,data,azGridSpacing=0.5,zenGridSpacing=0.5):

    az = np.linspace(0,360, int(360./azGridSpacing)+1 )
    zen = np.linspace(0,90, int(90./zenGridSpacing)+1 )

    model = np.nan_to_num(model)
    model_test = interpolate.interp2d(az, zen, model.reshape(az.size * zen.size,), kind='linear')

    mse = 0

    for i in range(0,np.shape(data)[0]):
        mse += (data[i,2] - model_test(data[i,0],data[i,1]))[0]**2
    mse  = 1./(2.*np.shape(data)[0]) * mse
    #print("CalcMSE:",mse)

    return mse

def setPlotFontSize(ax,fsize):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fsize)
    return ax

def plotModel(model,figname):
    az = np.linspace(0, 360, np.shape(model)[0])
    zz = np.linspace(0, 90, np.shape(model)[1])

    plt.ioff()
#   fig = plt.figure(figsize=(3.62, 2.76))

    #ax = fig.add_subplot(111,polar=True)
    ax = plt.subplot(111,polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.radians(90.))
    ax.set_ylim([0,1])
    ax.set_rgrids((0.00001, np.radians(20)/np.pi*2, np.radians(40)/np.pi*2,np.radians(60)/np.pi*2,np.radians(80)/np.pi*2),labels=('0', '20', '40', '60', '80'),angle=180)

    ma,mz = np.meshgrid(az,zz,indexing='ij')
    ma = ma.reshape(ma.size,)
    mz = mz.reshape(mz.size,)
           
    polar = ax.scatter(np.radians(ma), np.radians(mz)/np.pi*2., c=model[:,:], s=50, alpha=1., cmap=cm.RdBu,vmin=-15,vmax=15, lw=0)
    cbar = plt.colorbar(polar,shrink=0.75,pad=.10)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('ESM (mm)',size=8)
    ax = setPlotFontSize(ax,8)

    plt.tight_layout()
    plt.savefig(figname)
    plt.close()
    #plt.show()
    return

def plotModelElevationSlices(model,figname):
    az = np.linspace(0, 360, np.shape(model)[0])
    zz = np.linspace(0, 90, np.shape(model)[1])
    ele = 90. - zz[::-1]
    zenRes = model[0,:]
    eleRes = zenRes[::-1]

    plt.ioff()

    ax = plt.subplot(111)
    for i in range(0,np.shape(model)[0]):
        #print("Plotting Slice:",i)
        zenRes = model[i,:]
        eleRes = zenRes[::-1]
        ax.plot(ele,eleRes)

    #ax.set_ylim([0,1])
    ax = setPlotFontSize(ax,8)

    plt.tight_layout()
    plt.savefig(figname)
    plt.close()
    return

def plotModelsElevationSlices(models,figname):
    az = np.linspace(0, 360, np.shape(models)[1])
    zz = np.linspace(0, 90, np.shape(models)[2])
    ele = 90. - zz[::-1]

    plt.ioff()

    ax = plt.subplot(111)
    for i in range(0,np.shape(models)[1]):
        #print("Plotting Slice:",i)
        for j in range(0,np.shape(models)[0]):
            print("From Model segment:",j)
            zenRes = models[j,i,:]
            eleRes = zenRes[::-1]
            ax.plot(ele,eleRes)

    ax = setPlotFontSize(ax,8)

    plt.tight_layout()
    plt.savefig(figname)
    plt.close()
    return

def plotModelsElevationSlices3D(models,figname):
    az = np.linspace(0, 360, np.shape(models)[1])
    zz = np.linspace(0, 90, np.shape(models)[2])
    ele = 90. - zz[::-1]

    dd = range(0,np.shape(models)[1])
    mz,md = np.meshgrid(zz,dd,indexing='ij')
    md = md.reshape(md.size,)
    mz = mz.reshape(mz.size,)
    ax = plt.subplot(111,projection='3d')

    for i in range(0,np.shape(models)[1]):
        for j in range(0,np.shape(models)[0]):
            zenRes = models[j,i,:]
            eleRes = zenRes[::-1]
            blah = np.ones(np.size(zenRes))*j
            ax.plot(ele,blah,eleRes)
           
    ax = setPlotFontSize(ax,8)

    plt.tight_layout()
    plt.savefig(figname)
    plt.close()

    return

def print_antex_file_header(f):
    print('     1.4            M                                       ANTEX VERSION / SYST',file=f)
    print('A                                                           PCV TYPE / REFANT',file=f)
    print('                                                            END OF HEADER',file=f)

    return

# TODO fix gridspacing to be dynamc
def print_antex_header(antType,valid_from,valid_to,f):
    """
        print_antex_header antenna name
                           grid_spacing
    """
    f.write("                                                            START OF ANTENNA\n")
    f.write("{:<20s}                                        TYPE / SERIAL NO\n".format(antType))
    f.write("CALCULATED          ANU                      0    25-MAR-11 METH / BY / # / DATE\n")
    f.write("     0.5                                                    DAZI\n")
    f.write("     0.0  90.0   0.5                                        ZEN1 / ZEN2 / DZEN\n")
    f.write("     2                                                      # OF FREQUENCIES\n")

    # valid_from is a dto (datetime object
    yyyy, MM, dd, hh, mm, ss, ms = gt.dt2validFrom(valid_from)
    # force seconds to 0.00 for valid from
    f.write("{:>6s} {:>5s} {:>5s} {:>5s} {:>5s}    0.0000000                 VALID FROM\n".format(yyyy,MM,dd,hh,mm))
    yyyy, MM, dd, hh, mm, ss, ms = gt.dt2validFrom(valid_to)
    hh = str(23)
    mm = str(59)
    f.write("{:>6s} {:>5s} {:>5s} {:>5s} {:>5s}   59.9999999                 VALID UNTIL\n".format(yyyy,MM,dd,hh,mm))
    #
    # Change the numbers after ANU to the same code as the previous antenna 
    #
    f.write("ANU08_1648                                                  SINEX CODE\n")
    f.write("CALCULATED From MIT repro2                                  COMMENT\n")

    return 1

def print_start_frequency(freq,pco,f):
    f.write("   {:3s}                                                      START OF FREQUENCY\n".format(freq))

    pco_n = "{:0.2f}".format(pco[0])
    pco_n = "{:>10s}".format(pco_n)
    pco_e = "{:0.2f}".format(pco[1])
    pco_e = "{:>10s}".format(pco_e)
    pco_u = "{:0.2f}".format(pco[2])
    pco_u = "{:>10s}".format(pco_u)

    f.write(pco_n+pco_e+pco_u+"                              NORTH / EAST / UP\n")

def print_antex_noazi(data,f):
    noazi = "{:>8s}".format('NOAZI')

    for d in data:
        d = "{:>8.2f}".format(d)
        noazi = noazi + d

    f.write(noazi)
    f.write("\n")

def print_antex_line(az,data,f):
    az = "{:>8.1f}".format(az)

    for d in data:
        d = "{:>8.2f}".format(d)
        az = az+d

    f.write(az)
    f.write("\n")

def print_end_frequency(freq,f):
    f.write("   {:3s}                                                      END OF FREQUENCY\n".format(freq))

def print_end_antenna(f):
    f.write("                                                            END OF ANTENNA\n")

def create_esm(med,azGrid,zenGrid,antennas,antType):
    # add the block median residuals to an interpolate PCV file...
    # args.grid should come from the antenna data based on the grid spacing of the antex file

    print("Searching for :",antType)
    antenna = ant.antennaType(antType,antennas)
    print("antenna",antenna,antType)
    dzen = antenna['dzen'][2]
    x = np.linspace(0,360, int(360./dzen)+1 )
    y = np.linspace(0,90, int(90./dzen)+1 )

    L1_data = np.array(antenna['data'][0])
    L2_data = np.array(antenna['data'][1])

    # check to see if it is an elevation only model..
    # if it is then copy the elevation fields to all of the azimuth fields
    if np.shape(L1_data)[0] == 1:
        L1_tmp = np.zeros((np.size(x),np.size(y)))
        L2_tmp = np.zeros((np.size(x),np.size(y)))
        # often elevation only models only go down to 80 degrees in zenith
        for j in range(0,np.size(y)):
            if j >= np.shape(L1_data)[1] :
                L1_tmp[:,j] = 0. 
                L2_tmp[:,j] = 0.
            else:
                L1_tmp[:,j] = L1_data[0,j]
                L2_tmp[:,j] = L2_data[0,j]
        del L1_data, L2_data
        L1_data = L1_tmp
        L2_data = L2_tmp

    tmp = L1_data.reshape(x.size * y.size,)
    L1 = interpolate.interp2d(x, y, tmp, kind='linear')
    L2 = interpolate.interp2d(x, y, L2_data.reshape(x.size * y.size,), kind='linear')
    
    x_esm = np.linspace(0,360, int(360./azGrid)+1 )
    y_esm = np.linspace(0,90, int(90./zenGrid)+1 )
    esm = np.zeros((x_esm.size,y_esm.size,2))
   
    i = 0

    for az in x_esm :
        j = 0
        #==========================================
        # med is in elevation angle order
        # need to reverse the med array around so that it is in zenith order
        #==========================================
        for zen in y_esm :
            if med[i,j] > 0.00001 or med[i,j] < -0.00001 :
                esm[i,j,0] = med[i,j] + L1(az,zen)[0] 
                esm[i,j,1] = med[i,j] + L2(az,zen)[0]
            else:
                esm[i,j,0] = L1(az,zen)[0]
                esm[i,j,1] = L2(az,zen)[0]
            j += 1
        i += 1
    #================================================================
    #print("test interpolation for az=>360 zenith =>90 => 13.67:")
    #az = 360.
    #zen = 90.
    #print(L1(az,zen),L1(zen,az),L1(az,90.-zen),L1(90.-zen,az))
    #print("test interpolation for az=>355 zenith =>85 => 8.65:")
    #az = 355.
    #zen = 85.
    #print(L1(az,zen),L1(zen,az),L1(az,90.-zen),L1(90.-zen,az))
    #print(L1(355,80),L1(355,82.5),L1(355,85),L1(355,86),L1(355,87),L1(355,88),L1(355,89),L1(355,90))

    return esm

def parse_gmt(gmt_file):
    gmt = np.genfromtxt(gen_file)

    x_esm = np.linspace(0,360, int(360./azGrid)+1 )
    y_esm = np.linspace(0,90, int(90./zenGrid)+1 )
    med = np.zeros((x_esm.size,y_esm.size,2))

    azS = 0.1
    zenS = 0.1

    for az in x_esm :
        j = 0
        for zen in y_esm :
            if(i5 - azSpacing/2. < 0) :
                criterion = (gmt[:,0] < (az + azS/2.)) | (gmt[:,0] > (360. - azS/2.) ) & (gmt[:,1] < zen + zenZ/2.) & (gmt[:,1] > zen - zenS/2. )
            else:
                criterion = (gmt[:,0] < (az + azS/2.)) & (gmt[:,0] > (az - azS/2.) ) & (gmt[:,1] < zen + zenZ/2.) & (gmt[:,1] > zen - zenS/2. )

            ind = np.where(criterion)
            if ind.size > 1:
                print("ERROR have more than one value for :",az,zen, ind, gmt[ind,:])
    return med

def traverse_directory(args) :
    """
        traverse_directory(args)

        Search through a specified GAMIT project to look for DPH residuals files
        consolidate them into a compressed L3 format (.CL3), for analysis later.

    """
    siteRGX = re.compile('DPH.'+args.site.upper())
    s = []

    # report non-unique residuals
    for root, dirs, files in os.walk(args.traverse):
        path = root.split('/')
        for gamitFile in files:
            if siteRGX.search(gamitFile):
                gamitFile = root+'/'+gamitFile
                #check for potential duplicates in the same path, only want to use one of the DOH files
                if len(path[-1]) > 4:
                    regex = re.compile(root[:-2])
                else:
                    regex = re.compile(root)


                # only check for duplicates when there is more than one network
                # being processed...
                if args.network == 'yyyy_dddnN':
                    if len(s) == 0:
                        s.append(gamitFile)
                    else:
                        # for each element in s, check to see if the root path does not match
                        # any of the files already stored in the list
                        m = 0
                        for item in s:
                            if regex.search(item) :
                                m = 1
                        if not m :
                            s.append(gamitFile)
                else:
                    s.append(gamitFile)

    s.sort()
    lines = ''
    # Now loop through each file and consolidate the residuals
    for dfile in s :
        dphs = res.parseDPH(dfile)

        # check if the dph files are being searched are from
        #a GAMIT network of type yyyy/dddn?/
        root, filename = os.path.split(dfile)
        if args.network == 'yyyy_dddnN':
            ddd = root[-5:-2]
            year = int(root[-10:-6])
            startDT = dt.datetime(year,01,01)
            startDT = startDT + dt.timedelta(days=(int(ddd) -1))
        elif args.network == 'ddd':
            ddd = root[-3:]
            year = root[-8:-4] 
            startDT = dt.datetime(int(year),01,01)
            startDT = startDT + dt.timedelta(days=(int(ddd) -1))

        line = res.consolidate(dphs,startDT)
        lines = lines + line

        # if its larger than 1GB dump it to a file
        # this is designed to keep the load n the file system lighter
        if sys.getsizeof(lines) > 1073741824 :
            f = gzip.open(args.save_file,'a',9)
            f.write(lines)
            f.close()
            lines = ''
            #print(lines)

    # dump any remaining memory to file
    f = gzip.open(args.save_file,'a',9)
    f.write(lines)
    f.close()
    lines = ''

    return

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

    ant.printSatelliteModel(antenna)
    return 1

def calcNadirAngle(ele):
    """
        Calculate the NADIR angle based on the station's elevation angle

    """

    nadeg = np.arcsin(6378.0/26378.0 * np.cos(ele/180.*np.pi)) * 180./np.pi

    return nadeg

def applyNadirCorrection(svdat,nadirData,site_residuals):
    """
    sr = applyNadirCorrection(nadirData,site_residuals)

    Apply the nadir residual model to the carrier phase residuals.

    nadirData is a dictionary of satellite corrections
    nadirData['1'] = <array 70 elements> 0 to 13.8 degrees

    """
    #print("Attempting to apply the corrections")

    # form up a linear interpolater for each nadir model..
    nadirAngles = np.linspace(0,13.8,70)
    linearInt = {}
    for svn in nadirData:
        linearInt[svn] = interpolate.interp1d(nadirAngles ,nadirData[svn]) 
    # slow method, can;t assume the PRN will be the same SV over time..
    # can break residuals in daily chunks and then apply correction
    for i in range(0,np.shape(site_residuals)[0]):
        nadeg = calcNadirAngle(site_residuals[i,2])
        if nadeg > 13.8:
            nadeg = 13.8
        dto = gt.unix2dt(site_residuals[i,0])
        svn = svnav.findSV_DTO(svdat,int(site_residuals[i,4]),dto)
        #print("Looking for svn:",svn, int(site_residuals[i,4]),nadeg)
        site_residuals[i,3] = site_residuals[i,3] + linearInt[str(svn)](nadeg)

    return site_residuals 

#==============================================================================
#
# TODO:
#      test plots
#      error bars on elevation plot?

#=====================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    import argparse

    parser = argparse.ArgumentParser(prog='esm',description='Create an Empirical Site Model from one-way GAMIT phase residuals',
            formatter_class=argparse.RawTextHelpFormatter,
            epilog='''\
    Example:

    To create a consolidated phase residual file:
    > python ~/gg/com/esm.py --trav /short/dk5/repro2/2012/ --site YAR2 --network yyyy_dddnN --save_file YAR2.2012.CL3.gz

    To create a model:
    > python ~/gg/com/esm.py --model --site yar2 -f ./t/YAR2.2012.CL3
                   ''')

    #===================================================================
    # Consolidate DPH file options:
    parser.add_argument('--save_file',dest='save_file',default='./',
                            help="Location to save the consolidated phase files")
    parser.add_argument('--traverse',dest='traverse',
                            help="Location to search for DPH files from")

    # only support yyyy_dddn? and ddd
    parser.add_argument('--network',dest='network',default='yyyy_dddnN',choices=['yyyy_dddnN','ddd'],
                            help="Format of gps subnetworks")
    #===================================================================
    # Station meta data options
    parser.add_argument('-s', '--site', dest='site', required=True, help="SITE 4 character id")
    parser.add_argument('-a', '--antex', dest='antex', default="~/gg/tables/antmod.dat",help="Location of ANTEX file (default = ~/gg/tables/antmod.dat)")
    parser.add_argument('--station_file', dest='station_file', default="~/gg/tables/station.info",help="GAMIT station file with metadata (default= ~/gg/tables/station.info)")
    parser.add_argument('-g', '--grid', dest='grid', default=5.,type=float,help="ANTEX grid spacing (default = 5 degrees)")

    
    parser.add_argument('--esm_grid', dest='esm_grid', default=0.5, type=float,help="Grid spacing to use when creating an ESM (default = 0.5 degrees)")
    parser.add_argument('-f', dest='resfile', default='',help="Consolidated one-way LC phase residuals")
    #parser.add_argument('-t', '--AntType',dest='AntType')

    #===================================================================
    # Plot options
    parser.add_argument('--polar',dest='polar', default=False, action='store_true', help="Produce a polar plot of the ESM phase residuals (not working in development")
    parser.add_argument('--elevation',dest='elevation', default=False, action='store_true', help="Produce an elevation dependent plot of ESM phase residuals")
    
    #===================================================================
    # Start from a consolidated CPH file of the DPH residuals 
    #parser.add_argument('--dph',dest='dphFile')
    parser.add_argument('--model', dest='model', default=False, action='store_true',help="Create an ESM\n (default = False)")
    parser.add_argument('-o','--outfile',help='filename for ESM model (default = antmod.ssss)')

    parser.add_argument('--nadir',dest='nadir',help="location of satellite nadir residuals SV_RESIDUALS.ND3")
    parser.add_argument('--nm','--nadirModel',dest='nadirModel',default=False,action='store_true',
                        help="Create an ESM model for the satellites")

    parser.add_argument('--nadirPlot',dest='nadirPlot',default=False,action='store_true',help="Plot nadir residuals")
    parser.add_argument('--nadirCorrection',dest='nadirCorrection',default=False,action='store_true',help="Apply the satellite Nadir correction to the phase residuals")

    parser.add_argument('--test',dest='test',default=False,action='store_true')
    # Interpolation/extrapolation options
    # TODO: nearneighbour, polynomial, surface fit, etc..
    parser.add_argument('-i','--interpolate',dest='interpolate',choices=['ele_mean'],
                            help="ele_mean use the elevation mean to fill any missing values in the model")
    #===================================================================
    # Debug function, not needed
    #parser.add_argument('--gmt',dest='gmt_file',help="Debug function not implemented, use a GMT fit of residuals")
    parser.add_argument('--sv','--svnav', dest="svnavFile",default="~/gg/tables/svnav.dat", help="Location of GAMIT svnav.dat")
    #parser.add_argument('--version', action='version', version='%(prog)s 0.01') 
    args = parser.parse_args()

    # expand any home directory paths (~) to the full path, otherwise python won't find the file
    if args.resfile : args.resfile = os.path.expanduser(args.resfile)
    args.antex = os.path.expanduser(args.antex)
    args.station_file = os.path.expanduser(args.station_file)
    args.svnavFile = os.path.expanduser(args.svnavFile)

    svdat = []
    nadirData = {}
    #===================================================================
    # Look through the GAMIT processing subdirectories for DPH files 
    # belonging to a particular site.
    #===================================================================
    if args.test:
        ele = np.linspace(0,90,10)
        nad = calcNadirAngle(ele)
        print(ele,nad)
    if args.traverse :
        traverse_directory(args)
        if args.model:
            args.resfile = args.save_file 

    if args.nadir:
        nadir = np.genfromtxt(args.nadir)
        sv_nums = np.unique(nadir[:,2])
        nadirDataStd = {}
        for sv in sv_nums:
            criterion = nadir[:,2] == sv 
            ind = np.array(np.where(criterion))[0]
            nadir_medians = nanmean(nadir[ind,3:73],axis=0)
            nadir_stdev   = nanstd(nadir[ind,3:73],axis=0)
            nadirData[str(int(sv))] = nadir_medians
            #nadirDataStd[str(int(sv))] = nadir_stdev

        if args.nadirPlot:
            nadir = np.linspace(0,13.8, int(14.0/0.2) )
            svdat = svnav.parseSVNAV(args.svnavFile)

            # prepare a plot for each satellite block
            figBLK = []
            axBLK = []
            for i in range(0,7):
                figTmp = plt.figure(figsize=(3.62, 2.76))
                figBLK.append(figTmp)
                axTmp  = figBLK[i].add_subplot(111)
                axBLK.append(axTmp)

            # now plot by block
            for sv in nadirData:
                blk = svnav.findBLK_SV(svdat,sv)
                axBLK[int(blk)-1].plot(nadir,nadirData[sv],'-',alpha=0.7,linewidth=1,label="SV "+str(sv))
            # tidy each plot up
            for i in range(0,7):
                axBLK[i].set_xlabel('Nadir Angle (degrees)',fontsize=8)
                axBLK[i].set_ylabel('Residual (mm)',fontsize=8)
                axBLK[i].set_xlim([0, 14])
                axBLK[i].set_ylim([-5,5])
                axBLK[i].legend(fontsize=8,ncol=3)
                title = svnav.blockType(i+1)
                axBLK[i].set_title(title,fontsize=8)
                for item in ([axBLK[i].title, axBLK[i].xaxis.label, axBLK[i].yaxis.label] +
                    axBLK[i].get_xticklabels() + axBLK[i].get_yticklabels()):
                    item.set_fontsize(8)

            # Do a plot of all the satellites now..
            fig = plt.figure(figsize=(3.62, 2.76))
            ax = fig.add_subplot(111)

            for sv in nadirData:
                ax.plot(nadir,nadirData[sv],'-',alpha=0.7,linewidth=1)

            ax.set_xlabel('Nadir Angle (degrees)',fontsize=8)
            ax.set_ylabel('Residual (mm)',fontsize=8)
            ax.set_xlim([0, 14])
            ax.set_ylim([-5,5])

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(8)

            plt.tight_layout()
    
            # Plot the satellites by block
            blocks = np.unique(nadir[:,])
            plt.show()

    
        if args.nadirModel:
            antennas = ant.parseANTEX(args.antex)
            for sv in nadirData:
                svn = "{:03d}".format(int(sv))
                scode = 'G' + str(svn)
                antenna = ant.antennaScode(scode,antennas)
                for a in antenna:
                    satelliteModel(a, nadirData[sv])

    if args.model or args.elevation or args.polar:
        #===================================================================
        # get the antenna information from an antex file
        antennas = ant.parseANTEX(args.antex)

        # read in the consolidated LC residuals
        print("")
        print("Reading in the consolidated phase residuals from:",args.resfile)
        print("")
        site_residuals = res.parseConsolidatedNumpy(args.resfile)

        dt_start = gt.unix2dt(site_residuals[0,0])
        res_start = int(dt_start.strftime("%Y") + dt_start.strftime("%j"))
        dt_stop = gt.unix2dt(site_residuals[-1,0])
        res_stop = int(dt_stop.strftime("%Y") + dt_stop.strftime("%j"))
        print("\tResiduals run from:",res_start,"to:",res_stop)
        
        if args.nadirCorrection:
            svdat = svnav.parseSVNAV(args.svnavFile)
            print("\n\t** Applying the Satellite dependent Nadir angle correction to the phase residuals")
            print("svdat:",np.shape(svdat))
            site_residuals = applyNadirCorrection(svdat,nadirData,site_residuals)
            

        # work out how many models need to be created for the time period the residuals cover
        # check the station file, and looks for change in antenna type or radome type
        print("")
        print("Working out how many models need to be generated for",args.site.upper(),"using metadata obtained from:",args.station_file)
        print("")
        print("\t A new model will be formed whenever there is a change in:")
        print("\t\t1) Antenna type")
        print("\t\t2) Antenna serial number")
        print("\t\t3) Antenna height")
        print("\t\t4) Change of radome")
        print("")
        sdata = gsf.parseSite(args.station_file,args.site.upper())
    
        change = {}
        change['start_yyyy'] = [] # date antenna was installed on site
        change['start_ddd']  = []
        change['stop_yyyy']  = [] # date antenna was removed from the site
        change['stop_ddd']   = []
        change['ind']        = []
        change['valid_from'] = []
        change['valid_to']   = []


        # set up the initial instrument to have a start time as the first epoch we deal with
        change['start_yyyy'].append(dt_start.strftime("%Y"))
        change['start_ddd'].append(dt_start.strftime("%j"))

        # find the indices where the change occurs due to an antenna type / radome change
        ind = gsf.antennaChange(sdata)

        for i in ind:
            sdd = "{:03d}".format(int(sdata['start_ddd'][i]))
            stag = int( sdata['start_yyyy'][i] + sdd ) 
            if stag >= res_start and stag <= res_stop :
                print("There is a change on",sdata['start_yyyy'][i],sdata['start_ddd'][i],"to",sdata['antenna_type'][i],sdata['dome_type'][i])
                change['start_yyyy'].append(sdata['start_yyyy'][i])
                change['start_ddd'].append(sdata['start_ddd'][i])
                change['ind'].append(i)

                # update the stop time for the previous record
                change['stop_yyyy'].append(sdata['start_yyyy'][i])
                change['stop_ddd'].append(sdata['start_ddd'][i])

        change['stop_yyyy'].append(dt_stop.strftime("%Y"))
        change['stop_ddd'].append(dt_stop.strftime("%j"))

        models = np.zeros((np.size(change['ind'])+1,int(360./0.5)+1,int(90/0.5)+1,2))
        num_models = np.size(change['ind'])+1
        print("\nNumber of models which need to be formed:", num_models)

        ctr = 0

        for i in range(0,num_models):
            print("\t\tCreating model",i+1,"of",num_models)
            minVal_dt = gt.ydhms2dt(change['start_yyyy'][i],change['start_ddd'][i],0,0,0)
            maxVal_dt = gt.ydhms2dt(change['stop_yyyy'][i],change['stop_ddd'][i],0,0,0)

            criterion = ( ( site_residuals[:,0] >= calendar.timegm(minVal_dt.utctimetuple()) ) &
                    ( site_residuals[:,0] < calendar.timegm(maxVal_dt.utctimetuple()) ) )
            mind = np.array(np.where(criterion))

            change['valid_from'].append(minVal_dt)
            change['valid_to'].append(maxVal_dt)

            data = np.zeros((np.size(mind),3))
            data[:,0] = site_residuals[mind,1]
            data[:,1] = site_residuals[mind,2]
            data[:,2] = site_residuals[mind,3]

            # get the correct antenna type for this station at this time
            antType = gsf.antennaType(sdata,minVal_dt.strftime("%Y"),minVal_dt.strftime("%j"))
            print("get station info:",antType,minVal_dt.strftime("%Y"),minVal_dt.strftime("%j"))
            # do a block median with 5 sigma outlier detection at 0.5 degree grid
            med, medStd = blockMedian(data)
            if args.interpolate == 'ele_mean':
                med = interpolate_eleMean(med)

            # Take the block median residuals and add them to the ANTEX file
            esm = create_esm(med, 0.5, 0.5, antennas,antType)
            models[ctr,:,:,:] = esm
            ctr +=1

    #===========================================================================================
    # If we want an elevation or polar plot....
    #===========================================================================================
    if args.elevation or args.polar :
        import matplotlib.pyplot as plt
        from matplotlib import cm

        if args.polar :
            az = np.linspace(0, 360, int(360./0.5)+1)
            zz = np.linspace(0, 90, int(90./0.5)+1)

            # Plot the ESM... (antenna PCV + residuals)
            fig = plt.figure(figsize=(3.62, 2.76))
            ax = fig.add_subplot(111,polar=True)

            ax.set_theta_direction(-1)
            ax.set_theta_offset(np.radians(90.))
            ax.set_ylim([0,1])
            ax.set_rgrids((0.00001, np.radians(20)/np.pi*2, 
                                    np.radians(40)/np.pi*2,
                                    np.radians(60)/np.pi*2,
                                    np.radians(80)/np.pi*2),
                                    labels=('0', '20', '40', '60', '80'),angle=180)

            ma,mz = np.meshgrid(az,zz,indexing='ij')
            ma = ma.reshape(ma.size,)
            mz = mz.reshape(mz.size,)

            polar = ax.scatter(np.radians(ma), np.radians(mz)/np.pi*2., c=models[0,:,:,0], s=5, alpha=1., cmap=cm.RdBu,vmin=-15,vmax=15, lw=0)

            cbar = fig.colorbar(polar,shrink=0.75,pad=.10)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(args.site+' ESM (mm)',size=8)

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(8)

            plt.tight_layout()

            # Plot the blkm of the residuals
            fig2 = plt.figure(figsize=(3.62, 2.76))
            ax2 = fig2.add_subplot(111,polar=True)

            ax2.set_theta_direction(-1)
            ax2.set_theta_offset(np.radians(90.))
            ax2.set_ylim([0,1])
            ax2.set_rgrids((0.00001, np.radians(20)/np.pi*2, 
                                    np.radians(40)/np.pi*2,
                                    np.radians(60)/np.pi*2,
                                    np.radians(80)/np.pi*2),
                                    labels=('0', '20', '40', '60', '80'),angle=180)

            ma,mz = np.meshgrid(az,zz,indexing='ij')
            ma = ma.reshape(ma.size,)
            mz = mz.reshape(mz.size,)

            polar = ax2.scatter(np.radians(ma), np.radians(mz)/np.pi*2., c=med, s=5, alpha=1., cmap=cm.RdBu,vmin=-15,vmax=15, lw=0)

            cbar = fig2.colorbar(polar,shrink=0.75,pad=.10)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(args.site+' L3 Residuals (mm)',size=8)

            for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                    ax2.get_xticklabels() + ax2.get_yticklabels()):
                item.set_fontsize(8)

            plt.tight_layout()


        if args.elevation :
            #===========================================================
            # TODO: loop over changes in equipment...
            #===========================================================
            # Plot the residuals
            # the antenna model 
            # then the ESM
            # 
            #===========================================================
            # Do an elevation only plot of the residuals
            #===========================================================
            fig = plt.figure(figsize=(3.62, 2.76))
            ax = fig.add_subplot(111)
            ele = np.linspace(0,90, int(90./0.5)+1 )
            ele_model = []
            for i in range(0,720):
                ax.scatter(90.-ele,med[i,:],s=1,alpha=0.5,c='k')

            elevation = []
            for j in range(0,181):
                elevation.append(90.- j * 0.5)
                ele_model.append(nanmean(med[:,j]))

            ax.plot(elevation,ele_model[:],'r-',linewidth=2)
            ax.set_xlabel('Elevation Angle (degrees)',fontsize=8)
            ax.set_ylabel('ESM (mm)',fontsize=8)
            ax.set_xlim([0, 90])
            ax.set_ylim([-15,15])

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(8)

            plt.tight_layout()
            
            #===========================================================
            # Plot the antenna model before the residuals are added
            # on L1 
            #===========================================================
            #ax2 = fig.add_subplot(312)
            #ele = np.linspace(0,90, int(90./5)+1 )
            
            #minVal_dt = gt.ydhms2dt(change['start_yyyy'][0],change['start_ddd'][0],0,0,0)
            #maxVal_dt = gt.ydhms2dt(change['stop_yyyy'][0],change['stop_ddd'][0],0,0,0)

            #antType = gsf.antennaType(sdata,minVal_dt.strftime("%Y"),minVal_dt.strftime("%j"))
            #antenna = ant.antennaType(antType,antennas)

            #L1_data = np.array(antenna['data'][0])
            #L1_mean = np.mean(L1_data,axis=0)
            #f = interpolate.interp1d(ele,L1_mean)

            #L1_int = []
            #ele_i = []
            #for j in range(0,181):
            #    ele_i.append(j*0.5)
            #    L1_int.append(f(j*0.5)) 
            #L1_int = np.array(L1_int)

            #ax2.plot(ele,L1_mean[::-1],'b-',alpha=0.5,linewidth=2)
            #ax2.plot(ele_i,L1_int[::-1],'k--')
            #ax2.set_xlabel('Elevation Angle (degrees)',fontsize=8)
            #ax2.set_ylabel('L1 PCV (mm)',fontsize=8)
            #ax2.set_xlim([0, 90])

            #for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
            #    ax2.get_xticklabels() + ax2.get_yticklabels()):
            #    item.set_fontsize(8)

            #plt.tight_layout()
            #===========================================================
            # Do an elevation only plot of the ESM
            #===========================================================
            #fig = plt.figure(figsize=(3.62, 2.76))
            #ax3 = fig.add_subplot(313)
            #ele = np.linspace(0,90, int(90./0.5)+1 )
            #ele_esm = []

            #esm = create_esm(med, 0.5, 0.5, antennas,antType)

            #for j in range(0,181):
            #    ele_esm.append(np.mean(esm[:,j,0]))

            # plot the ele only esm model
            #ax3.plot(ele, ele_esm[::-1], 'g-',alpha=0.5,linewidth=2)

            # plot the interpolated  ant ele PCV
            #ax3.plot(ele_i,L1_int[::-1],'b--',alpha=0.5,linewidth=2)

            # plot the esm - antenna model => should get the residuals
            #ax3.plot(ele, ele_esm - L1_int, 'b--',alpha=0.5,linewidth=2)

            # plot the ele only residuals 
            #ax3.plot(ele, ele_model[::-1] , 'r-',alpha=0.3,linewidth=2)
           
            #ax3.legend(['esm','pcv','esm-pcv','residuals'],loc='best')

            # Try a crude attempt at an esm
            #fix = L1_int[::-1] + ele_model[::-1]
            #ax3.plot(ele, fix,'k--',alpha=0.5,linewidth=2)
            #ax3.plot(ele, fix - L1_int[::-1],'r--')
            #ax3.set_xlabel('Elevation Angle (degrees)',fontsize=8)
            #ax3.set_ylabel('ESM (mm)',fontsize=8)
            #ax3.set_xlim([0, 90])

            #for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
            #    ax3.get_xticklabels() + ax3.get_yticklabels()):
            #    item.set_fontsize(8)

            #plt.tight_layout()
        plt.show()
    #===================================================
    # print the esm model residuals + antenna to an ANTEX file 
    #===================================================
    if args.model:
        if not args.outfile:
            args.outfile = "antmod."+args.site.lower()

        print("")
        print("Adding the ESM to the antenna PCV model to be saved to:",args.outfile)
        print("")
        f = open(args.outfile,'w')
        for m in range(0,num_models):
            
            antType = gsf.antennaType( sdata, change['start_yyyy'][m], change['start_ddd'][m] )
            antenna = ant.antennaType(antType,antennas)
            print("Model",m+1," is being added to the antenna PCV for:",antType)
            print_antex_header(antType, change['valid_from'][m],change['valid_to'][m],f)
            freq_ctr = 0
            for freq in ['G01','G02'] :
                pco = antenna['PCO_'+freq]
                print_start_frequency(freq,pco,f)
                noazi = np.mean(models[m,:,:,freq_ctr],axis=0)
                print_antex_noazi(noazi,f)

                for i in range(0,int(360./args.esm_grid)+1):
                    print_antex_line(float(i*args.esm_grid),models[m,i,:,freq_ctr],f)
                print_end_frequency(freq,f)
                freq_ctr +=1
            print_end_antenna(f)
        f.close()
    print("FINISHED")
