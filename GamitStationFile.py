#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import re
import numpy as np

import gpsTime as gt

def parseSite(station_file,site):
    
    site_RGX = re.compile('^ '+site)
    data = {}
    data['start_yyyy'] = []
    data['start_ddd'] = []
    data['antenna_type'] = []
    data['antenna_serial'] = []
    data['antenna_height'] = []
    data['receiver_type'] = []
    data['dome_type'] = []
    data['end_yyyy'] = []
    data['end_ddd'] = []

    with open(station_file,'r') as sta:
        for line in sta:
            if site_RGX.search(line) :
                site = line[1:5]
                station_name = line[7:25].rstrip()
                start_yyyy = line[25:29]
                start_ddd  = line[30:33]
                end_yyyy  = line[44:48]
                end_ddd   = line[49:52]
                antenna_height = line[59:70].rstrip()
                receiver_type = line[97:119].rstrip()
                antenna_type = line[170:186].rstrip()
                dome_type    = line[187:191].rstrip()
                antenna_serial = line[194:214].rstrip()
                #print(line)
                data['start_yyyy'].append(start_yyyy)
                data['start_ddd'].append(start_ddd)
                data['antenna_type'].append(antenna_type) 
                data['antenna_serial'].append(antenna_serial)
                data['antenna_height'].append(antenna_height)
                data['receiver_type'].append(receiver_type)
                data['dome_type'].append(dome_type)
                data['end_yyyy'].append(end_yyyy)
                data['end_ddd'].append(end_ddd)
                #print("<",site,"><",station_name,"><",start_yyyy,"><",start_ddd,"><",end_yyyy,"><",end_ddd,"><",receiver_type,"><",antenna_type,"><",dome_type,">")

    return data

def determineESMChanges(start,end,sdata):
    '''
        Determine if there have been any changes, as reported by the station_file,
        that will require an new ESM to be computed
    '''
    change = {}
    change['start_yyyy'] = [] # date antenna was installed on site
    change['start_ddd']  = []
    change['stop_yyyy']  = [] # date antenna was removed from the site
    change['stop_ddd']   = []
    change['ind']        = []
    change['valid_from'] = []
    change['valid_to']   = []

    # set up the initial instrument to have a start time as the first epoch we deal with
    change['start_yyyy'].append(start.strftime("%Y"))
    change['start_ddd'].append(start.strftime("%j"))

    # find the indices where the change occurs due to an antenna type / radome change
    ind = antennaChange(sdata)
    res_start = int(start.strftime("%Y") + start.strftime("%j")) 
    res_stop = int(end.strftime("%Y") + end.strftime("%j"))

    for i in ind:
        sdd = "{:03d}".format(int(sdata['start_ddd'][i]))
        stag = int( sdata['start_yyyy'][i] + sdd )
        if stag >= res_start and stag <= res_stop :
            #print("There is a change on",sdata['start_yyyy'][i],sdata['start_ddd'][i],"to",sdata['antenna_type'][i],sdata['dome_type'][i])
            change['start_yyyy'].append(sdata['start_yyyy'][i])
            change['start_ddd'].append(sdata['start_ddd'][i])
            change['ind'].append(i)

            # update the stop time for the previous record
            change['stop_yyyy'].append(sdata['start_yyyy'][i])
            change['stop_ddd'].append(sdata['start_ddd'][i])

        change['stop_yyyy'].append(end.strftime("%Y"))
        change['stop_ddd'].append(end.strftime("%j"))

        models = np.zeros((np.size(change['ind'])+1,int(360./0.5)+1,int(90/0.5)+1,2))
        num_models = np.size(change['ind'])+1
    return change

def antennaChange(data): #,start_yyyy,start_ddd,end_yyyy,end_ddd):
    ctr = 0
    ind = []

    for ant in data['antenna_type']:
        if ctr > 0:
            if data['antenna_type'][ctr] != data['antenna_type'][ctr -1]:
                #print("Change of antenna: ",data['antenna_type'][ctr-1],data['antenna_type'][ctr])
                ind.append(ctr)
            elif data['dome_type'][ctr] != data['dome_type'][ctr -1]:
                #print("Change of radome: ",data['dome_type'][ctr-1],data['dome_type'][ctr])
                ind.append(ctr)
            elif data['antenna_serial'][ctr] != data['antenna_serial'][ctr -1]:
                #print("Change of serial number: ",data['antenna_serial'][ctr-1],data['antenna_serial'][ctr])
                ind.append(ctr)
            elif data['antenna_height'][ctr] != data['antenna_height'][ctr -1]:
                #print("Change of antenna height: ",data['antenna_height'][ctr-1],data['antenna_height'][ctr])
                ind.append(ctr)
        ctr += 1
    return ind

def antennaType(data,yyyy,ddd): #,start_yyyy,start_ddd,end_yyyy,end_ddd):
    ctr = 0
    ind = []
    # create a time tag of the form YYYYDDD
    tag = int(yyyy+ddd)
    antennaType = ''
    #I'm assuming the station file data is already in time order...
    for i in range(0,np.size( data['start_yyyy'])): 
        stag = int("{:>04d}{:>03d}".format(int(data['start_yyyy'][i]),int(data['start_ddd'][i])))
        #print("Tag:",tag,"stag:",stag,data['antenna_type'][i],data['dome_type'][i])

        if stag <= tag:
            antennaType = "{:<16s}{:<4s}".format(data['antenna_type'][i],data['dome_type'][i])

    return antennaType 

#==========================================================================================================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(prog='esm',description='Parse ANTEX files')

    parser.add_argument('-f', '--station_file', dest='station_file', default='./t/station.info',required=True)
    parser.add_argument('-s', '--site', dest='site', default='',required=True)

    parser.add_argument('--esm_report', dest='esm_report',default=False, action='store_true',
      help='Determine when a station will need to create a new ESM, based on equipmet changes')
    parser.add_argument('--start_yyyy',dest='start_yyyy',type=int,
            help='Year of first epoch in YYYY format')
    parser.add_argument('--start_ddd',dest='start_ddd',type=int,
            help='Day-of-Year of first epoch in DDD format')
    parser.add_argument('--end_yyyy',dest='end_yyyy',type=int,
            help='Year of last epoch in YYYY format')
    parser.add_argument('--end_ddd',dest='end_ddd',type=int,
            help='Day-of-Year of last epoch in DDD format')

    args = parser.parse_args()
    #=========================================

    # find the station meta data for a particular site
    station_metadata = parseSite(args.station_file, args.site)

    #print("data",data)

    #ind = antennaChange(data)
    #for i in ind:
    #    print(data['start_yyyy'][i],data['start_ddd'][i],data['antenna_type'][i],data['dome_type'][i])
    #print("ind:",ind)

    if args.esm_report:
        start = gt.ydhms2dt(args.start_yyyy,args.start_ddd,0,0,0)
        end   = gt.ydhms2dt(args.end_yyyy,args.end_ddd,0,0,0)
        changes = determineESMChanges(start,end,station_metadata)
        num_models = np.size(changes['ind'])+1
        print(args.site,"requires ",num_models)
