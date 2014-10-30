#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import numpy as np
import re
import gzip
import calendar

from scipy.stats.stats import nanmean, nanmedian, nanstd

import gpsTime as gt
import datetime as dt

import esm

def file_opener(filename):
    '''
    Decide what kind of file opener should be used to parse the data:
    # file signatures from: http://www.garykessler.net/library/file_sigs.html
    '''

    # A Dictionary of some file signatures,
    # Note the opener statements are not correct for bzip2 and zip
    openers = {
        "\x1f\x8b\x08": gzip.open,
        "\x42\x5a\x68": open,      # bz2 file signature
        "\x50\x4b\x03\x04": open   # zip file signature
    }

    max_len = max(len(x) for x in openers)
    with open(filename) as f:
        file_start = f.read(max_len)
        for signature, filetype in openers.items():
            if file_start.startswith(signature):
                return filetype
    return open

def file_len(fname):
    """
    file_len : work out how many lines are in a file
    
    Usage: file_len(fname)

    Input: fname - filename, can take a txt file or a gzipp'd file

    Output: i - number of lines in the file

    """
    file_open = file_opener(fname)
    with file_open(fname) as f:
        i=-1 # account for 0-length files
        for i, l in enumerate(f):
            pass
    return i + 1

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
    #tmp = np.array(tmp)
    for z in zen:
        criterion = ( (data[:,2] < (z + zenSpacing/2.)) &
                      (data[:,2] > (z - zenSpacing/2.)) )
        ind = np.array(np.where(criterion))[0]
        rout = reject_outliers_arg(data[ind,3],nSigma)
        tmp.append(rout.tolist())
        #tmp = np.concatenate([tmp,rout])
    return tmp

def reject_outliers_elevation(data,nSigma,zenSpacing=0.5):
    zen = np.linspace(0,90,int(90./zenSpacing)+1)
    init = 0
    for z in zen:
        criterion = ( (data[:,2] < (z + zenSpacing/2.)) &
                      (data[:,2] > (z - zenSpacing/2.)) )
        ind = np.array(np.where(criterion))[0]
        if ind.size < 1:
            continue
        tdata = np.zeros((np.size(ind),3))
        tdata = data[ind,:]

        criterion = ( (data[ind,3] < (data[ind,3].mean() + data[ind,3].std() * nSigma)) &
                  (data[ind,3] > (data[ind,3].mean() - data[ind,3].std() * nSigma)) )
        rout = np.array(np.where(criterion))[0]

        # if its the first iteration initialise tmp
        if init == 0 and np.size(rout) > 0:
            tmp = tdata[rout,:]
            init = 1
        elif np.size(rout) > 0:
            tmp = np.vstack((tmp,tdata[rout,:]))

    return tmp

def reject_absVal(data,val):
    criterion = ( (data[:,3] > -1. * val) & (data[:,3] < val) )
    ind = np.array(np.where(criterion))[0]
    tmp = data[ind,:]
    return tmp

def parseDPH(dphFile) :
    """
    dph = parseDPH(dphFile)

    Read in a GAMIT undifferenced phase residual file.
    Return a DPH structure

    Will skip any lines in the file which contain a '*' 
    within any column 

    Checks there are no comments in the first column of the file
    Checks if the file is gzip'd or uncompressed

    """

    asterixRGX = re.compile('\*')

    dph = {}
    obs = {}

    obs['satsViewed'] = set()
    obs['epochs'] = set()

    debug = 0

    # work out if the file is compressed or not,
    # and then get the correct file opener.
    file_open = file_opener(dphFile)

    with file_open(dphFile) as f:
        for line in f:
            dph = {}
            if line[0] != ' ':
                if debug :
                    print('A comment',line)
            elif asterixRGX.search(line):
                if debug :
                    print('Bad observation',line)
            else :
                # If the lccyc is greater than 1, reject this epoch
                if float(line[43:51]) > 1. or float(line[43:51]) < -1.:
                    continue 
                # if elevation is below 10 degress ignore
                if float(line[105:112]) < 10:
                    continue 

                dph['epoch'] = int(line[1:5])
                dph['l1cyc'] = float(line[6:15])
                dph['l2cyc'] = float(line[16:24])
                dph['p1cyc'] = float(line[25:33])
                dph['p2cyc'] = float(line[34:42])
                dph['lccyc'] = float(line[43:51])
                dph['lgcyc'] = float(line[52:60])
                dph['pccyc'] = float(line[61:69])
                dph['wlcyc'] = float(line[70:78])
                dph['ncyc']  = float(line[79:87])
                dph['lsv']   = int(line[88:91])
                dph['az']    = float(line[94:102])
                dph['el']    = float(line[105:112])
                dph['pf']    = int(line[113:114])
                dph['dataf'] = int(line[115:127])

                # these fields are not always preset
                if str(line[128:148]).strip() != '' :
                    dph['L1cycles'] = float(line[128:148])
                if str(line[149:169]).strip() != '' :
                    dph['L2cycles'] = float(line[149:169])

                dph['prn']   = int(line[171:173])
                prnSTR = 'prn_'+str(dph['prn'])
                epoch = str(dph['epoch'])

                # store the data in lists accessed by the sat prn key
                if dph['prn'] in obs['satsViewed'] :
                    obs[prnSTR].append(dph)
                else:
                    obs[prnSTR] = []
                    obs[prnSTR].append(dph)

                # keep a record of which indice each epoch is located at
                ind = len(obs[prnSTR]) - 1

                # Keep a record of each satellite viewed at each epoch in a set
                epochStr = str(dph['epoch'])
                if dph['epoch'] in obs['epochs']:
                    obs[epochStr][str(dph['prn'])]=ind    
                else :
                    obs['epochs'].add(dph['epoch'])
                    obs[epochStr] = {} 
                    obs[epochStr][str(dph['prn'])]=ind    

                # keep a record of all the unique satellies which have residuals
                obs['satsViewed'].add(dph['prn'])
        
    return obs 

#def parseConsolidatedNumpy(cfile,dt_start=0,dt_stop=0):
def parseConsolidatedNumpy(cfile):
    '''
    parseConsolidated   Read in a consolidate phase residual file that contains all of the epochs
                        for a particular site
    
    Usage:  residuals = parseConsolidated('TOW2.2012.DPH.gz')

    Input:  file - TOW2.2012.DPH.gz, can take gzipp'd or plane txt files
    
    Output: residuals - an array of dictionaries

    '''
    nlines = file_len(cfile)
    residuals = np.zeros((nlines,5))

    # work out if the file is compressed or not,
    # and then get the correct file opener.
    file_open = file_opener(cfile)
    ctr = 0

    with file_open(cfile) as f:
        for line in f:
            tmp = {}
            yyyy, ddd, ts, az, zen, lc, prn = line.split( )
            hh,mm,ss = ts.split(':')
            dto = gt.ydhms2dt(yyyy,ddd,hh,mm,ss)
            if float(lc) > 1000:
                next
            else:
                residuals[ctr,0] = calendar.timegm(dto.utctimetuple())
                residuals[ctr,1] = float(az)
                residuals[ctr,2] = float(zen)
                residuals[ctr,3] = float(lc)
                residuals[ctr,4] = int(prn)
                ctr += 1
 
    # check to see if we are tie filtering the residuals
    #if dt_start > 0.0001 :
    #    criterion = ( ( residuals[:,0] >= calendar.timegm(dt_start.utctimetuple()) ) &
    #                         ( residuals[:,0] < calendar.timegm(dt_stop.utctimetuple()) ) )
    #    tind = np.array(np.where(criterion))[0]
    #    print("going from:",nlines,"to:",np.size(tind))
    #    res = np.zeros((np.size(tind,5)))
    #    res = residuals[tind,:]
    #else:
    #print("no time filtering")
    res = np.zeros((ctr,5))
    res = residuals[0:ctr,:]

    return res 

def parseConsolidated(cfile):
    res = parseConsolidatedNumpy(cfile)
    return res

def consolidate(dphs,startDT) :
    '''
    consolidate look through a GAMIT DPH file strip out the epcoh, azimuth, zenith angle
    lcresidual and PRN, and dump it to a file as:

            timestamp az zen lc(mm) prn

    Input: 
        dphs a parsed dph structe obtained from resiudals.parseDPH(file)
        startDT a datetime object specify the start time of the first residual at epoch 1

    Output:
        filename if it ends in gz it will be automatically compressed
    '''
    lines = ''
    sep = ' '

    # Iterate over each epoch
    for epoch in dphs['epochs']:
        for sat in dphs[str(epoch)]:
            satPRN = 'prn_'+str(sat)
            ep  = dphs[str(epoch)][str(sat)]
            az  = dphs[satPRN][ep]['az']
            zen = 90. - dphs[satPRN][ep]['el']
            epoch = dphs[satPRN][ep]['epoch']
            lc_mm = dphs[satPRN][ep]['lccyc'] * 190.
            timeStamp = startDT + dt.timedelta(seconds=epoch*30)
            time = timeStamp.strftime("%Y %j %H:%M:%S")
            lines = lines+str(time)+sep+str(az)+sep+str(zen)+sep+str(lc_mm)+sep+str(sat)+"\n"

    return lines

def findVal(value,attr,siteRes):
    '''
    findVal   Find the all occurances of the atrribute with a value within 
              a residuals data structure.
                
    Usage: i = findVal(attr,value,siteRes)

    Input:  attr 'time', 'az', 'zen', 'lc', 'prn'
            value is a date time object to find the first occurence of a residual
            res a consolidate phase residual data structure

    Output: ind and array of indicies which match the values

    Best used for searching for a specific epoch or prn.

    SEE ALSO: findValRange() - good for searching for az, and zenith values 
                               within a range or tolerance

    '''
    ind = []
    for (index, d) in enumerate(siteRes):
        if d[attr] == value :
            ind.append(index)

    return ind

def findValRange(minVal,maxVal,attr,siteRes):
    '''
    findValiRange   Find the all occurances of the atrribute with a value within 
                    a residuals data structure, within a certain tolerance. 
                    For instance 23.0 amd 23.5
                
    Usage: i = findValRange(minVal,maxVal,attr,siteRes)

    Input:  attr 'time', 'az', 'zen', 'lc', 'prn'
            minVal value 
            maxVal value
            res a consolidate phase residual data structure

    Output: i index in array that has the first matching observation

    Best used for searching for az, zen or lc.
    Search is based on minVal <= val < maxVal

    SEE ALSO: findVal() - good for searching for specific PRNs or epochs 

    '''
    #print('minVal',minVal,'maxVal',maxVal,'attr',attr)
    ind = []
    for (index, d) in enumerate(siteRes):
        if d[attr] >= minVal and d[attr] < maxVal  :
            ind.append(index)

    return ind

def findTimeRange(minVal,maxVal,siteRes):
    '''
    findValiRange   Find the all occurances of the atrribute with a value within 
                    a residuals data structure, within a certain tolerance. 
                    For instance 23.0 and 23.5
                
    Usage: i = findValRange(minVal,maxVal,attr,siteRes)

    Input:  attr 'time', 'az', 'zen', 'lc', 'prn'
            minVal value 
            maxVal value
            res a consolidate phase residual data structure

    Output: i index in array that has the first matching observation

    Best used for searching for az, zen or lc.
    Search is based on minVal <= val < maxVal

    SEE ALSO: findVal() - good for searching for specific PRNs or epochs 

    '''
    #print('minVal',minVal,'maxVal',maxVal,'attr',attr)
    ind = []

    #criterion = (siteRes[:]['time'] > minVal) & (siteRes[:]['time'] < maxVal)
    criterion = (siteRes[:,0] > minVal) & (siteRes[:,0] < maxVal)
    ind = np.array(np.where(criterion))
    #for (index, d) in enumerate(siteRes):
    #    if d[attr] >= minVal and d[attr] < maxVal  :
    #        ind.append(index)

    return ind

def gamitWeight(site_residuals):
    """
    Determine the gamit weighting of the phase residuals

    see ~/gg/kf/ctogobs/proc_phsin.f line ~ 2530
    """
    # norm  - Normal equation for sig**2 = a**2 + b**2/sine(elevation)**2
    # b     - Solution vector
    # det   - determinant of norm
    # zpart - Partial for 1/sine(el)**2
    # zdep  - A and B coefficients for the model
    #vel_light = 299792458.0
    #fL1 = 154.*10.23E6
    #cyc_to_mm = (vel_light/fL1) *1000.
    #print("cyc_to_mm:",cyc_to_mm)
    sums_lc = np.zeros(18)
    nums_lc = np.zeros(18)

    norm = np.zeros(3)
    b = np.zeros(2)
    zdep = np.zeros(2)

    # Split everything up into 17 bins
    for r in range(0,np.shape(site_residuals)[0]):
        ele_bin = int(site_residuals[r,2]/5.0)
        sums_lc[ele_bin] = sums_lc[ele_bin] + np.sqrt(site_residuals[r,3]**2)
        nums_lc[ele_bin] = nums_lc[ele_bin] + 1

    for i in range(0,18):
        if nums_lc[i] > 0:
            #sums_lc[i] = np.sqrt( sums_lc[i] / nums_lc[i] )#*cyc_to_mm
            sums_lc[i] = sums_lc[i] / nums_lc[i] #*cyc_to_mm

        zpart = 1. / np.sin(np.radians((i+1)*5.0 - 2.5))**2

        # Accumulate the normals weighted by the number of data points
        if nums_lc[i] > 0 :
            norm[0] = norm[0] + 1
            norm[1] = norm[1] + zpart
            norm[2] = norm[2] + zpart**2
            b[0] = b[0] + sums_lc[i]**2
            b[1] = b[1] + (zpart*sums_lc[i])**2

    # Now compute the determinate and solve the equations accounting
    # for both zdep(1) and zdep(2) need to be positive
    det = norm[0] * norm[2] - norm[1]**2
    if det > 0.:
        zdep[0] = (b[0] * norm[2] - b[1]*norm[1]) / det
        zdep[1] = (b[1] * norm[0] - b[0]*norm[1]) / det
        #print("DET:",det,b[0],b[1],norm[0],norm[1],norm[2],zdep[0],zdep[1],zpart)

        # If the mean is less than zero, set it to 1 mm and use elevation angle dependence   
        if zdep[0] < 0.0 :
            zdep[0] = (zdep[0] + zdep[1])/2.
            b[1] = b[1] - norm[1]*zdep[0]
            zdep[1] = b[1]/norm[2]
            #print("1, mean is less than zero")
        # If the elevation term is zero, then just use a constant value
        if zdep[1] < 0.0 :
            zdep[0] = b[0]/norm[0]
            zdep[1] = 0.0
            #print("2,elevation term is zero, use a constan value")
    else:
        if norm[0] > 0:
            zdep[0] = b[0]/norm[0]
            zdep[1] = 0.0
            #print("3,blah")
        else:
            zdep[0] = 10.0
            zdep[1] = 0.0
            #print("4,blah")

    # Final check to make sure a non-zero value is given
    if zdep[0] < 0.01:
        zdep[0] = 10.0
        #print("5,blah")

    a = np.sqrt(zdep[0])
    b = np.sqrt(zdep[1])

    return a, b

#===========================================================================
if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from matplotlib import cm 

    #===================================
    # TODO Change this to argparse..
    #from optparse import OptionParser
    import argparse

    parser = argparse.ArgumentParser(prog='esm',description='Analyse one-way GAMIT phase residuals')

    parser.add_argument("-f", "--filename", dest="filename", help="Result file to plot")
    parser.add_argument("-e", "--elevation", dest="elevationPlot",action='store_true',default=False,
                        help="Plot Residuals vs Elevation Angle")
    parser.add_argument("-p", "--polar", dest="polarPlot",action='store_true',default=False,
                        help="Polar Plot Residuals vs Azimuth & Elevation Angle")
    parser.add_argument("--esm","--ESM",dest="esmFilename",help="Example Residual file from which to create an ESM")

    parser.add_argument("--dph",dest="dphFilename",help="DPH filename to parse, obtained from GAMIT") 

    parser.add_argument("-c", dest="consolidatedFile",help="Consolidated L3 residual file")
    parser.add_argument("--convert", dest="convertDphFile",help="Convert DPH file to consolidated")

    parser.add_argument("--daily",dest="daily",action='store_true',help="Plot daily variation of residuals")

    parser.add_argument("--sat",dest="sat",action='store_true',help="Plot residuals by satellite")
    args = parser.parse_args()
    #===================================
   
    if args.dphFilename :
        dphs = parseDPH(args.dphFilename)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        elevation = []
        lccyc = []
        # Iterate over each epoch
        for epoch in dphs['epochs']:
            for sat in dphs[str(epoch)]:
                satPRN = 'prn_'+str(sat)
                ep  = dphs[str(epoch)][str(sat)]

                lc = dphs[satPRN][ep]['lccyc']
                lccyc.append(dphs[satPRN][ep]['lccyc'])  
                elevation.append(dphs[satPRN][ep]['el'])
                #ax.scatter( dphs[satPRN][ep]['el'],lc,'k.',alpha=0.5)
       
        lccyc = np.array(lccyc)
        elevation = np.array(elevation)

        eleSpacing = 1
        ele = np.linspace(0,90,int(90./eleSpacing)+1)
        val = np.zeros(int(90./eleSpacing)+1) 
        ctr = 0
        for e in ele:
            criterion = ( (elevation < (e + eleSpacing/2.)) &
                          (elevation > (e - eleSpacing/2.)) )
            ind = np.array(np.where(criterion))[0]
            if np.size(ind) > 1:
                val[ctr] = np.median(lccyc[ind])
            else:
                val[ctr] = 0
            ctr+=1

        ax.plot( ele, val, 'r-', alpha=0.6,linewidth=2)
        ax.plot( ele, val*190, 'b-', alpha=0.6,linewidth=2)
        ax.plot( ele, val*107, 'g-', alpha=0.6,linewidth=2)
        ax.set_ylabel('lccyc',fontsize=8)
        ax.set_xlabel('Elevation Angle (degrees)',fontsize=8)
        ax.set_xlim([0, 90])

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)

        plt.tight_layout()
        plt.show() 
    
    # Calculate the block median
    zz = np.linspace(0,90,181)

    if args.consolidatedFile :
        cdata = parseConsolidated(args.consolidatedFile) 

    if args.daily:
        dt_start = gt.unix2dt(cdata[0,0])
        startDTO = dt_start
        res_start = int(dt_start.strftime("%Y") + dt_start.strftime("%j"))

        dt_stop = gt.unix2dt(cdata[-1,0])
        res_stop = int(dt_stop.strftime("%Y") + dt_stop.strftime("%j"))

        total_time = dt_stop - dt_start
        days = total_time.days + 1
        print("Residuals start from:",res_start," and end at ",res_stop,"total_time:",total_time,"in days:",total_time.days)

        eleMedians = np.zeros((days,181))
        d = 0
        while d < days:
            minDTO = startDTO + dt.timedelta(days = d)
            maxDTO = startDTO + dt.timedelta(days = d+1)

            criterion = ( ( cdata[:,0] >= calendar.timegm(minDTO.utctimetuple()) ) &
                          ( cdata[:,0] < calendar.timegm(maxDTO.utctimetuple()) ) )
            tind = np.array(np.where(criterion))[0]
            ele_model = []

            # check we have some data for each day
            if np.size(tind) > 0 :
                # split the data for this test
                blkm, blkmstd = esm.blockMedian(cdata[tind,1:4])
                for j in range(0,181):
                    ele_model.append(nanmean(blkm[:,j]))
                ele_model = np.array(ele_model)
                eleMedians[d,:] = np.array(ele_model)
            d += 1

        elevation = []
        for j in range(0,181):
            elevation.append(90.- j * 0.5)
        #===========================================================
        fig = plt.figure(figsize=(3.62, 2.76))
        ax = fig.add_subplot(111)

        for i in range(0,np.shape(eleMedians)[0]):
            ax.plot(elevation,eleMedians[i,:],alpha=0.5)

        # now compute the over all median
        blkm, blkmstd = esm.blockMedian(cdata[:,1:4])
        ele_model = []
        for j in range(0,181):
            ele_model.append(nanmean(blkm[:,j]))

        ax.plot(elevation,ele_model,'r-',alpha=0.5,linewidth=2)
        ax.set_xlabel('Elevation Angle (degrees)',fontsize=8)
        ax.set_ylabel('ESM (mm)',fontsize=8)
        ax.set_xlim([0, 90])
        #ax.set_ylim([-15,15])

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)

        plt.tight_layout()
        plt.show()

    if args.sat:
        dt_start = gt.unix2dt(cdata[0,0])
        startDTO = dt_start
        res_start = int(dt_start.strftime("%Y") + dt_start.strftime("%j"))

        dt_stop = gt.unix2dt(cdata[-1,0])
        res_stop = int(dt_stop.strftime("%Y") + dt_stop.strftime("%j"))

        total_time = dt_stop - dt_start
        days = total_time.days + 1
        print("Residuals start from:",res_start," and end at ",res_stop,"total_time:",total_time,"in days:",total_time.days)

        for prn in range(1,33):
            criterion = ( cdata[:,4] == prn) 
            prnd = np.array(np.where(criterion))[0]
            if np.size(prnd) < 1 :
                continue
            print("Checking:",prn)
            #===========================================================
            fig = plt.figure(figsize=(3.62, 2.76))
            ax = fig.add_subplot(111)
            data = cdata[prnd,:]
            zenSpacing = 0.5
            median = []
            zen = np.linspace(0,90,int(90./zenSpacing) +1)
            for z in zen :
                criterion = ( (data[:,2] < (z + zenSpacing/2.)) &
                          (data[:,2] > (z - zenSpacing/2.)) )
                ind = np.array(np.where(criterion))[0]
                tmp = data[ind,:]
                rout = esm.reject_outliers_arg(tmp[:,3],3)
                for i in rout :
                    ax.plot(90.- z, tmp[i,3],'k.',alpha=0.5)
                median.append(nanmedian(data[ind,3]))
            ax.plot(90.-zen,median,'r-',alpha=0.5)


            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(8)

            ax.set_ylim([-35,35])
            plt.tight_layout()
            plt.savefig(str(prn)+"_ele.png")
            plt.close()
            #================================================
            az = np.linspace(0,360,721)
            fig = plt.figure(figsize=(3.62, 2.76))

            ax = fig.add_subplot(111,polar=True)
            ax.set_theta_direction(-1)
            ax.set_theta_offset(np.radians(90.))
            ax.set_ylim([0,1])
            ax.set_rgrids((0.00001, np.radians(20)/np.pi*2, np.radians(40)/np.pi*2,np.radians(60)/np.pi*2,np.radians(80)/np.pi*2), 
                    labels=('0', '20', '40', '60', '80'),angle=180)

            ma,mz = np.meshgrid(az,zz,indexing='ij')
            ma = ma.reshape(ma.size,)
            mz = mz.reshape(mz.size,)
            med, medStd = esm.blockMedian(data[:,1:4])
            #tmp = reject_outliers_elevation(data,5,0.5)
            polar = ax.scatter(np.radians(ma), np.radians(mz)/np.pi*2., c=med ,s=5,alpha=1., cmap=cm.RdBu,vmin=-10,vmax=10, lw=0)
            del data,med,medStd
   
            #cbar = fig.colorbar(polar,shrink=0.75,pad=.10)
            #cbar.ax.tick_params(labelsize=8)
            #cbar.set_label('Residuals (mm)',size=8)

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(8)

            plt.tight_layout()
            plt.savefig(str(prn)+"_az.png")
            plt.close()


    if args.elevationPlot :

        # Do an elevation only plot
        fig = plt.figure(figsize=(3.62, 2.76))
        ax = fig.add_subplot(111)
        tmp = reject_outliers_elevation(cdata,5,0.5)
        ax.scatter(90.-tmp[:,2],tmp[:,3])#,'k.',alpha=0.2)
        #ax.plot(ele,np.median(med))
        ax.set_xlabel('Elevation Angle (degrees)',fontsize=8)
        ax.set_ylabel('Bias (mm)',fontsize=8)
        #ax.set_ylim([-17.5, 17.5])
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)

        plt.tight_layout()
    
        #fig.savefig('MOBS_Elevation_Median.eps')
    # Create a polar plot of the residuals
    if args.polarPlot:
        #blkMedian,blkMedianStd,rms = blockMedian(option.filename,0.5,1)
        az = np.linspace(0,360,721)
        #fig = plt.figure()
        fig = plt.figure(figsize=(3.62, 2.76))

        ax = fig.add_subplot(111,polar=True)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.radians(90.))
        ax.set_ylim([0,1])
        tmp = reject_outliers_elevation(cdata,5,0.5)
        ax.set_rgrids((0.00001, np.radians(20)/np.pi*2, np.radians(40)/np.pi*2,np.radians(60)/np.pi*2,np.radians(80)/np.pi*2), 
                    labels=('0', '20', '40', '60', '80'),angle=180)

        ma,mz = np.meshgrid(az,zz,indexing='ij')
        ma = ma.reshape(ma.size,)
        mz = mz.reshape(mz.size,)
        #polar = ax.scatter(np.radians(ma), np.radians(mz)/np.pi*2., c=blkMedian ,s=1,alpha=1., cmap=cm.RdBu,vmin=-15,vmax=15, lw=0)
        polar = ax.scatter(np.radians(ma), np.radians(mz)/np.pi*2., c=tmp ,s=1,alpha=1., cmap=cm.RdBu,vmin=-10,vmax=10, lw=0)
   
        cbar = fig.colorbar(polar,shrink=0.75,pad=.10)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Residuals (mm)',size=8)

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)

        plt.tight_layout()
   
        # Print out the ratio if the elvation plot has been selected as well
        if args.elevationPlot:
            ratio = rms/medrms
            print('{} {:.3f} {:.3f} {:.2f}').format(args.filename,medrms,rms,ratio)

    if args.polarPlot | args.elevationPlot :
        plt.show()

    if args.esmFilename :
        esm,esmStd = blockMedian(args.esmFilename,0.5,1)

    if args.convertDphFile:
        print("about to consolidate the file:",args.convertDphFile)
        dph2Consolidated(args.convertDphFile)
