#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import numpy as np
import string as s
import re

import gpsTime as gt

class Antenna:
    def __init__(self, name='ASH701945C_M   NONE', source='igs08.atx') :
        # check the antenna name has a radome defined, and is 20 characters long
        if len(name)> 20:
            print('Antenna name',name,'is:', len(name),'characters long')
            print('The antenna name/type should only be 20 characters long')
        self.name   = name
        if len(name) == 20:
            self.dome = name[-4:]
            # set blanks to NONE
            if self.dome == '    ':
                self.dome = 'NONE'
            self.model = name[0:15]
            #print("MODEL:",self.model,":")

        # Keep a record of where this calibration result came from
        self.source = source


    #def diff(self, other):
    #    '''

    #    antenna.diff(other)

    #    Calculate the difference in absolute antenna PCVs from another antenna

    #    '''
    #    print(self.name - other.name)
    #    return diff
    # source record the source of the antenna calibration igs081577.atx

    # Set / Get functions
    def dazi(self, dazi):
        '''
            dazi change in azimuth increment (float)
            ant.dazi(5.0)
            inc = ant.dazi()
        '''
        #if(dazi):
        self.dazi = dazi
        return self.dazi

    def startZen(self, zen):
        self.startZen = zen

    def endZen(self, zen):
        self.endZen = zen

    def incZenith(self, inc):
        self.incZenith = inc

    def dzen(self, dzen):
        '''
            dzen change in zenith increment (float)
            ant.dzen([0,90,5.0])
            inc = ant.dazi()
        '''
        self.startZen(dzen[0])
        self.endZen(dzen[1])
        self.incZenith(dzen[2])

    def dome(self,dome):
        self.dome = dome

    # number of frequencies that have a calibration
    # value
    def nFreq(self,nFreq):
        self.nFreq = nFreq
        # set up a list of frequencies names to be filled in later
        self.freqItr = 0
        self.frequencies = []

    def validFrom(self, validFrom):
        self.validFrom = validFrom

    def validTo(self, validTo):
        self.validTo = validTo

    def noAzi(self, noazi):
        self.noazi = np.array(noazi)
         
    def startFreq(self,cFreq):
        # convert G 1 to G01, etc..
        if cFreq.size == 2:
            tmp = "{}{:02d}".format(cFreq[0],int(cFreq[1]))
            self.frequencies.append(tmp)
        else:
            self.frequencies.append(cFreq[0])

# TO do def parseNGS(ngsFile):
def parseANTEX(atxFile):
    '''
    parseANTEX(antexFile)

    Read an ANTEX format file and return an array of antennas

    '''
    #=====================================================
    typeRGX             = re.compile('TYPE\s\/\sSERIAL\sNO')
    startOfAntennaRGX   = re.compile('START OF ANTENNA')
    endOfAntennaRGX     = re.compile('END OF ANTENNA')

    daziRGX             = re.compile('DAZI')
    dzenRGX             = re.compile('ZEN1\s\/\sZEN2\s\/\sDZEN')
    validFromRGX        = re.compile('VALID FROM')
    validToRGX          = re.compile('VALID TO')
    sinexCodeRGX        = re.compile('SINEX CODE')
    pcoRGX              = re.compile('NORTH / EAST / UP')

    numFreqRGX          = re.compile('# OF FREQUENCIES')

    startOfFrequencyRGX = re.compile('START OF FREQUENCY')
    endOfFrequencyRGX   = re.compile('END OF FREQUENCY')
    
    noaziRGX            = re.compile('NOAZI')
    typeRGX             = re.compile('TYPE\s\/\sSERIAL\sNO')
    #=====================================================

    freqFlag = 0

    numAntennas = 0
    antennas = []

    with open(atxFile) as f:
        for line in f:
            if typeRGX.search(line): 
                antenna['name'] = line[0:15]+' '+line[16:20]#+serial number
                antenna['type'] = line[0:15]
                antenna['dome'] = line[16:20]
                antenna['serialNum'] = line[20:40]

                # only applied for satellite PCV
                # scode "sNNN" s = sytem, NNN = SVN number 
                antenna['scode'] = line[40:50]
                # COSPAR ID "YYYY-XXXA" 
                antenna['cospar'] = line[50:60]

                ant = Antenna(line[0:15]+' '+line[16:20], atxFile)
                # might need to put in an exception for satellite antennas
                # if dome is blank set to none
                if antenna['dome'] == '    ':
                    antenna['dome'] = 'NONE'
                # define some defaults
                antenna['frequency'] = []
                antenna['data'] = np.zeros(0) 
            elif startOfAntennaRGX.search(line):
                antenna = {}
            elif daziRGX.search(line):
                dazi = np.array(s.split(line))[0]
                antenna['dazi'] = dazi.astype(np.float)
                ant.dazi( dazi.astype(np.float) )
                #print('Calling dazi from class',ant.dazi)
            elif dzenRGX.search(line):
                dzen = np.array(s.split(line))[0:-5]
                antenna['dzen'] = dzen.astype(np.float)
                ant.dzen(dzen.astype(np.float))
            elif numFreqRGX.search(line):
                numFreq = np.array(s.split(line))[0]
                antenna['numFreq'] = numFreq.astype(np.int)
                ant.nFreq(numFreq.astype(np.int))
            elif validFromRGX.search(line):
                # array should have size 6, YYYY MM DD HH MM SS.SSSS
                # maybe less if it has been entered like 2004 01 01, etc...
                validFrom = np.array(s.split(line)[0:-2])
                antenna['validFrom'] = validFrom.astype(np.float)
                ant.validFrom(validFrom.astype(np.float))
            elif validToRGX.search(line):
                validTo = np.array(s.split(line)[0:-2])
                antenna['validTo'] = validTo.astype(np.float) 
                ant.validTo(validTo.astype(np.float))
            elif startOfFrequencyRGX.search(line):
                cFreq = np.array(s.split(line))[0:-3]
                # convert G 1 to G01, etc..
                if cFreq.size == 2:
                    tmp = "{}{:02d}".format(cFreq[0],int(cFreq[1]))
                    antenna['frequency'].append(tmp)
                else:
                    antenna['frequency'].append(cFreq[0])
                freqFlag = 1
                ant.startFreq(cFreq)
            elif pcoRGX.search(line):
                pco = np.array(s.split(line))[0:-5]
                name = 'PCO_'+antenna['frequency'][-1]
                antenna[name] = pco.astype(np.float)

                if antenna['data'].size < 1:
                    if antenna['dazi'] < 0.0001 :
                        nAZI = 1
                    else:
                        nAZI = int(360. / antenna['dazi']) + 1
                    nZEN = int((antenna['dzen'][1] - antenna['dzen'][0])/antenna['dzen'][2])+1
                    antenna['data'] = np.zeros((antenna['numFreq'],nAZI,nZEN))
            elif noaziRGX.search(line):
                noazi = np.array(s.split(line)[1:])
                antenna['noazi'] = noazi.astype(np.float)
                ant.noAzi(noazi.astype(np.float))
            elif endOfFrequencyRGX.search(line):
                freqFlag = 0
                cFreq = ''
            # End of Antenna Flag
            elif endOfAntennaRGX.search(line):
                antennas.append(antenna)
                numAntennas += 1
                freqFlag = 0
            elif freqFlag :
                tmp = np.array(s.split(line)[1:],dtype=float)
                itr = 0
                f = np.size(antenna['frequency']) - 1
                for v in tmp:
                    antenna['data'][f][freqFlag-1][itr] = v 
                    itr += 1
                freqFlag += 1   

        return antennas              

#============================

def parseGEOPP(geoppFile):
    '''
    parseGEOPP(antmod.are)

    -parse the geopp .are file format
    -parse the geopp .arp file format
    -parse the geopp .ant file

    ane:
    are: is an elvation dependent only antenna calibration result, where the
    PCO has been pushed into the PCVs.

    ant:
    arp:

    An example file is in t/....

    '''

    #=====================================================
    # Set up some regexs to parse the data
    #=====================================================
    typeRGX            = re.compile('^TYPE=')
    serialRGX          = re.compile('SERIAL NUMBER=')
    calibrationTypeRGX = re.compile('CALIBRATION TYPE=')
    calibrationDateRGX = re.compile('CALIBRATION DATE=')
    numAntennasRGX     = re.compile('NO OF ANTENNAS=')
    numCalibrationRGX  = re.compile('NO OF CALIBRATIONS=')
    gnssTypeRGX        = re.compile('GNSS TYPE=')
    contentRGX         = re.compile('CONTENT TYPE=')
    pcvTypeRGX         = re.compile('PCV TYPE=')
    numFreqRGX         = re.compile('NO OF FREQUENCIES=')
    offsetL1RGX        = re.compile('OFFSETS L1=')
    offsetL2RGX        = re.compile('OFFSETS L2=')
    deleRGX            = re.compile('ELEVATION INCREMENT=')
    daziRGX            = re.compile('AZIMUTH INCREMENT=')
    varL1RGX           = re.compile('VARIATIONS L1=')
    varL2RGX           = re.compile('VARIATIONS L2=')
    #=====================================================

    L1flag = 0
    L2flag = 0
    ObsCtr = 0
    antenna = {}

    with open(geoppFile) as f:
        for line in f:
            line = line.rstrip()
            if typeRGX.search(line) :
                antenna['type'] = line[5:21]
                antenna['dome'] = line[21:25]
                if antenna['dome'] == '    ':
                    antenna['dome'] = 'NONE'
                antenna['name'] = antenna['type'] + antenna['dome']
            elif serialRGX.search(line) and len(line) > 14 :
                antenna['serialnum'] = line[14:]
                antenna['name'] = antenna['name'] + ' ' + antenna['serialnum']
            elif calibrationTypeRGX.search(line) and len(line) > 17 :
                antenna['calType'] = line[17:]
            elif calibrationDateRGX.search(line) and len(line) > 17 :
                antenna['calDate'] = line[17:]
            elif numAntennasRGX.search(line) and len(line) > 15 :
                antenna['numAntennas'] = int(line[15:])
            elif numCalibrationRGX.search(line) and len(line) > 19 :
                antenna['numCalibrations'] = int(line[19:])
            elif gnssTypeRGX.search(line) and len(line) > 10 :
                antenna['gnssType'] = line[10:]
            elif contentRGX.search(line) and len(line) > 13 :
                antenna['content'] = line[13:]
            elif pcvTypeRGX.search(line) and len(line) > 9:
                antenna['pcvType'] = line[9:]
            elif numFreqRGX.search(line) and len(line) > 18 :
                antenna['numFreq'] = int(line[18:])
            elif offsetL1RGX.search(line) and len(line) > 11 :
                antenna['offsetL1'] = line[11:]
            elif offsetL2RGX.search(line) and len(line) > 11 :
                antenna['offsetL2'] = line[11:]
            elif deleRGX.search(line) and len(line) > 20 :
                antenna['dele'] = float(line[20:])
            elif daziRGX.search(line) and len(line) > 18 :
                antenna['dazi'] = float(line[18:])
            elif varL1RGX.search(line) :
                L1flag = 1
                ObsCtr = 0
            elif varL2RGX.search(line) :
                L2flag = 1
                ObsCtr = 0
                #print("started the L2 obs")
            elif L1flag == 1:
                tmp = np.array(s.split(line))
                #check that all of the data has been read in before reseting the flag
                if antenna['dazi'] < 0.0001 :
                    antenna['L1PCV'] = tmp.astype(np.float)
                    L1flag = 0
                    ObsCtr = 0
                else :
                    if ObsCtr == 0:
                        rows = int(360./antenna['dazi']) + 1
                        cols = int(90./antenna['dele']) + 1
                        antenna['L1PCV'] = np.zeros((rows,cols))

                    antenna['L1PCV'][ObsCtr,:] = tmp.astype(np.float)
                    ObsCtr += 1

                    if ObsCtr == (int(360./antenna['dazi']) + 1):
                        L1flag = 0
                        ObsCtr = 0
            elif L2flag == 1:
                tmp = np.array(s.split(line))
                if antenna['dazi'] < 0.0001 :
                    antenna['L2PCV'] = tmp.astype(np.float)
                    L2flag = 0
                    ObsCtr = 0
                else :
                    if ObsCtr == 0:
                        rows = int(360./antenna['dazi']) + 1
                        cols = int(90./antenna['dele']) + 1
                        antenna['L2PCV'] = np.zeros((rows,cols))

                    antenna['L2PCV'][ObsCtr,:] = tmp.astype(np.float)
                    ObsCtr += 1

                    if ObsCtr == (int(360./antenna['dazi']) + 1):
                        L2flag = 0
                        ObsCtr = 0
                #L2flag = 0

    return antenna



#=====================================
def antennaType(antennaType,antennas):
    '''
    antenna = antennaType(antennaType,antennas)
    '''
    found = []
    for antenna in antennas:
        if antenna['name'] == antennaType:
            found.append(antenna)

    # if only one antenna is found return this one
    if np.size(found) == 1:
        return found[0]
    elif np.size(found) > 1:
        print("WARNING found more the one antenne of type:",antennaType)
        print("returning first one of",np.size(found))
        return(found[0])

    # try another serach with the radome set to NONE 
    antennaTypeNONE = antennaType[0:16]+'NONE'

    for antenna in antennas:
        if antenna['name'] == antennaTypeNONE:
            return antenna
    print('Could not find <'+antennaType+'>') 
    return -1

def antennaTypeSerial(antennaType,antennaSerial,antennas):
    '''
    antenna = antennaType(antennaType,antennaSerial,antennas)
    '''
    for antenna in antennas:
        if antenna['name'].rstrip() == antennaType.rstrip() and antenna['serialNum'].rstrip() == antennaSerial.rstrip():
            return antenna

    print('Could not find <'+antennaType+'><'+antennaSerial+'>') 

    return -1

def antennaTypeScode(antennaType,SatCode,antennas):
    '''
    antenna = antennaTypeSatCode(antennaType,SatCode,antennas)
    '''
    for antenna in antennas:
        if antenna['name'].rstrip() == antennaType.rstrip() and antenna['scode'].rstrip() == SatCode.rstrip():
            return antenna
    print('Could not find <'+antennaType+'><'+SatCode+'>') 
    return -1

def antennaScode(SatCode,antennas):
    '''
    [antenna] = antennaTypeSatCode(SatCode,antennas)
    '''
    found = []
    for antenna in antennas:
        if antenna['scode'].rstrip() == SatCode.rstrip():
            found.append(antenna)

    # if only one antenna is found return this one
    if np.size(found) == 1:
        return found
    elif np.size(found) > 1:
        print("WARNING found more the one antenne of type:",antennaType)
        #print("returning first one of",np.size(found))
        print("returning all of them",np.size(found))
        return found
    print('Could not find <'+SatCode+'>') 
    return -1

def printSatelliteModel(antenna):
    print("                                                            START OF ANTENNA")
    #print("{:<20s}                                        TYPE / SERIAL NO".format(antenna['type']))#antType))
    print("{:<20s}{:<20s}{:<10s}{:<10s}TYPE / SERIAL NO".format(antenna['type'],antenna['serialNum'],antenna['scode'],antenna['cospar']))
    print("EMPIRICAL MODEL     ANU                      0    25-MAR-11 METH / BY / # / DATE")
    print("     0.0                                                    DAZI")
    print("     0.0  17.0   1.0                                        ZEN1 / ZEN2 / DZEN")
    print("     2                                                      # OF FREQUENCIES")

    # valid_from is a dto (datetime object
    print("VALID FROM:",antenna['validFrom'])
    #yyyy, MM, dd, hh, mm, ss, ms = gt.dt2validFrom(antenna['validFrom'])
    # force seconds to 0.00 for valid from
    #print("{:>06d} {:>5s} {:>5s} {:>5s} {:>5s}    0.0000000                 VALID FROM\n".format(int(antenna['validFrom'][0]),antenna['validFrom'][1],antenna['validFrom'][2],antenna['validFrom'][3],antenna['validFrom'][4]))
    print("  {:>04d}  {:>02d}  0.0000000                 VALID FROM\n".format(int(antenna['validFrom'][0]),int(antenna['validFrom'][1])) )
    #print("VALID TO:",antenna['validTo'])
    #yyyy, MM, dd, hh, mm, ss, ms = gt.dt2validFrom(antenna['validTo'])
    #hh = str(23)
    #mm = str(59)
    #print("{:>6s} {:>5s} {:>5s} {:>5s} {:>5s}   59.9999999                 VALID UNTIL\n".format(yyyy,MM,dd,hh,mm))
    #
    # Change the numbers after ANU to the same code as the previous antenna 
    #
    print("ANU08_1648                                                  SINEX CODE")
    # TODO: add in date time, user, computer and version of esm model was used in COMMENTS
    print("Empirical model derived from MIT repro2                     COMMENT")

    print("   {:3s}                                                      START OF FREQUENCY".format('G01'))

    pco_n = "{:0.2f}".format(antenna['PCO_G01'][0])
    pco_n = "{:>10s}".format(pco_n)
    pco_e = "{:0.2f}".format(antenna['PCO_G01'][1])
    pco_e = "{:>10s}".format(pco_e)
    pco_u = "{:0.2f}".format(antenna['PCO_G01'][2])
    pco_u = "{:>10s}".format(pco_u)

    print(pco_n+pco_e+pco_u+"                              NORTH / EAST / UP")

    noazi = "{:>8s}".format('NOAZI')

    for d in antenna['noazi']:
        d = "{:>8.2f}".format(d)
        noazi = noazi + d

    print(noazi)
    print("   {:3s}                                                      END OF FREQUENCY".format('G01'))

    #================= G02 =======================

    print("   {:3s}                                                      START OF FREQUENCY".format('G02'))

    pco_n = "{:0.2f}".format(antenna['PCO_G02'][0])
    pco_n = "{:>10s}".format(pco_n)
    pco_e = "{:0.2f}".format(antenna['PCO_G02'][1])
    pco_e = "{:>10s}".format(pco_e)
    pco_u = "{:0.2f}".format(antenna['PCO_G02'][2])
    pco_u = "{:>10s}".format(pco_u)

    print(pco_n+pco_e+pco_u+"                              NORTH / EAST / UP")

    noazi = "{:>8s}".format('NOAZI')

    for d in antenna['noazi']:
        d = "{:>8.2f}".format(d)
        noazi = noazi + d

    print(noazi)
    print("   {:3s}                                                      END OF FREQUENCY".format('G02'))
    print("                                                            END OF ANTENNA")

    return 1


def plt_layout(ax,fontzise):
    """
    Set the axis and labels to the same fontsize
    """

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

    return
#=====================================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(prog='antenna',description='Parse ANTEX files')

    parser.add_argument('-f', '--file', dest='file1', default='./t/antmod.dat')

    parser.add_argument('-t', '--AntType',dest='AntType', default='ASH701945C_M    NONE')
    parser.add_argument('-s', '--AntSerial',dest='AntSerial')
    parser.add_argument('--SCODE',dest='SatCode')
    #parser.add_argument('-t', '--AntType',dest='AntType', default='TRM59800.00     NONE')

    parser.add_argument('-p', '--plot',dest='plot', default=False, action='store_true')
    parser.add_argument('--polar',dest='polar', default=False, action='store_true')
    parser.add_argument('--elevation',dest='elevation', default=False, action='store_true')
    parser.add_argument('--EM',dest='elevationMedian', default=False, action='store_true', help='Plot the Median PCV vs Elevation')

    parser.add_argument('--LC',dest='LCdata', default=False, action='store_true',help='Calculate an LC observable')

    args = parser.parse_args()

    antennas = parseANTEX(args.file1)

    if args.AntSerial:
        antenna = antennaTypeSerial(args.AntType,args.AntSerial,antennas)
    elif args.SatCode:
        antenna = antennaTypeScode(args.AntType,args.SatCode,antennas)
    else:
        antenna = antennaType(args.AntType,antennas)
    #print("Found:",antenna)    

    if args.polar or  args.elevation or args.elevationMedian :
        import matplotlib.pyplot as plt
        from matplotlib import cm

        aData = antenna['data'][0]

    if args.LCdata:
        tmpL1 = antenna['data'][0] * 2.5457
        tmpL2 = antenna['data'][1] * 1.5457
        aData = tmpL1 - tmpL2

    if args.polar :
        az = np.linspace(0,360,int(360./antenna['dazi'] +1))
        zz = np.linspace(antenna['dzen'][0],antenna['dzen'][1],int(90./antenna['dzen'][2] + 1))

        fig = plt.figure(figsize=(3.62, 2.76))

        ax = fig.add_subplot(111,polar=True)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.radians(90.))
        ax.set_ylim([0,1])
        ax.set_rgrids((0.00001, np.radians(20)/np.pi*2, np.radians(40)/np.pi*2,np.radians(60)/np.pi*2,np.radians(80)/np.pi*2),labels=('0', '20', '40', '60', '80'),angle=180)

        ma,mz = np.meshgrid(az,zz,indexing='ij')
        ma = ma.reshape(ma.size,)
        mz = mz.reshape(mz.size,)
        polar = ax.scatter(np.radians(ma), np.radians(mz)/np.pi*2., c=aData, s=50, alpha=1., cmap=cm.RdBu,vmin=-15,vmax=15, lw=0)

        cbar = fig.colorbar(polar,shrink=0.75,pad=.10)
        cbar.ax.tick_params(labelsize=8)
        if args.LCdata:
            cbar.set_label('LC PCV (mm)',size=8)
        else:
            cbar.set_label('L1 PCV (mm)',size=8)
        plt_layout(ax,8)
        plt.tight_layout()

    if args.elevation :
        zz = np.linspace(antenna['dzen'][0],antenna['dzen'][1],antenna['dzen'][1]/antenna['dzen'][2]+1)
        #zz = np.linspace(0,90,19)
        #zz = np.linspace(0,90,181)
        ele = 90. - zz[::-1]

        # Do an elevation only plot
        fig = plt.figure(figsize=(3.62, 2.76))
        ax = fig.add_subplot(111)
        # check to see if it is a satellite antenna (< 14 degrees)
        if antenna['dzen'][1] > 30. :
            for zen in aData :
                ax.plot(ele,zen[::-1])
            ax.set_xlabel('Elevation Angle (degrees)',fontsize=8)
        else:
            ax.plot(zz,antenna['noazi'])
            ax.set_xlabel('Nadir Angle (degrees)',fontsize=8)
        ax.set_ylabel('PCV (mm)',fontsize=8)
        #ax.set_ylim([-15, 15])
        plt_layout(ax,8)
        plt.tight_layout()

    if args.elevationMedian :
        zz = np.linspace(0,90,181)
        ele = 90. - zz[::-1]
        # Do an elevation only plot
        fig = plt.figure(figsize=(3.62, 2.76))
        ax = fig.add_subplot(111)
        #med = np.median(aData,axis=0)
        med = np.mean(aData,axis=0)
        ax.plot(ele,med[::-1])
        #for zen in aData :
        #   ax.plot(ele,zen[::-1])
        ax.set_xlabel('Elevation Angle (degrees)',fontsize=8)
        ax.set_ylabel('PCV (mm)',fontsize=8)
        ax.set_xlim([10, 90])
        ax.set_ylim([-10, 5])
        plt_layout(ax,8)
        plt.tight_layout()

    if args.polar or  args.elevation or args.elevationMedian :
        plt.show()

#==============================================================================

