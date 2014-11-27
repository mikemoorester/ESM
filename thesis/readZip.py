import zipfile 
import re
import os
from os.path import join

import gpsTime as gt
import autclnParse as acln
import svnav

import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.dates as mdates

def plot_tight(ax,fontsize=8):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)
    return ax

#def gamitSummary(names) :

#==============================================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(prog='readZip',description='')
    parser.add_argument('-p', '--path', dest='path',nargs='+', help="Path to search for zip files")
    parser.add_argument('--plot', dest='plot',default=False,action='store_true', help="Path to search for zip files")
    parser.add_argument('--sum', dest='summary',default=False,action='store_true', help="Find the gamit summary files")
    parser.add_argument('--autcln', dest='autcln',default=False,action='store_true', help="Find the gamit autcln postfit summary files")
    parser.add_argument('--svnav', dest='svnavFile',default='', help="Location of the svnav.dat file")

    args = parser.parse_args()

    #==========================================================

    numStationsRGX = re.compile('Number of stations used ')
    ambiguitiesRGX = re.compile('Phase ambiguities WL fixed')
    summaryRGX     = re.compile('sh_gamit_') 
    autclnRGX      = re.compile('autcln.post.sum$') 

    files = []
    for path in args.path:
        t_files = os.listdir(path)
        for tf in t_files:
            files.append(join(path,tf))

    summary = []
    info = {}
    autclns = []

    for zfile in files:
        if zipfile.is_zipfile(zfile):
            #print("It is a zipfile")
            with zipfile.ZipFile(zfile,'r') as zf:
                names = zf.namelist()

                for name in names:
                    if args.summary and summaryRGX.search(name):
                        interestedFile = name
                    elif args.autcln and autclnRGX.search(name):
                        interestedFile = name

                try: 
                    data = zf.read(interestedFile)
                except:
                    print("Can't read: ",interestedFile)
                    continue

                lines = data.split('\n')

                if args.autcln:
                    autcln = acln.parseAUTCLN(lines)
                    #print("AUTCLN:",autcln)
                    # work out what network the data belongs to from the file name
                    # this cant be found in the file itself
                    autcln['network'] = int(interestedFile[-17])
                    #print("Network:",autcln['network'])
                    if 'date' in autcln.keys():
                        autclns.append(autcln)
                    else:
                        print("Cant find the date in autcln for:",interestedFile,autcln['date'])

                elif args.summary:
                    year = summaryFile[0:4]
                    doy  = summaryFile[5:8]
                    net  = summaryFile[9:10]

                    key = year+'_'+doy
                    if key in info:
                        data = info[key]
                    else:
                        data = {}
                        data['num_sites'] = 0
                        data['WL'] = 0
                        data['NL'] = 0
                        data['num_records'] = 0
                        data['year'] = int(year)
                        data['doy'] = int(doy)
                        info[key]= data

                    for line in lines:
                        if numStationsRGX.search(line):
                            num = int(line[24:26])
                            data['num_sites'] += num
                            data['num_records'] += 1
                        elif ambiguitiesRGX.search(line):
                            WL = float(line[29:33])
                            NL = float(line[45:49])
                            data['WL'] += WL
                            data['NL'] += NL

                interestedFile = ''

    #print("INFO:",info)

    if args.plot and args.summary:
        fig = plt.figure(figsize=(7.24, 2.76))
        ax = fig.add_subplot(111)
        time_stamps = []
        ambiguties_NL = []
        ambiguties_WL = []
        num_stations = []

        for data in info:
            #print("DATA:",data,info[data])
            mp_ts = gt.ydhms2mdt(info[data]['year'],info[data]['doy'],0.,0.,0.)
            time_stamps.append(mp_ts)
            amb_NL = info[data]['NL'] / float(info[data]['num_records'])
            amb_WL = info[data]['WL'] / float(info[data]['num_records'])
            ambiguties_NL.append(amb_NL)
            ambiguties_WL.append(amb_WL)
            num_stations.append(info[data]['num_sites'])
       
        ax.plot(time_stamps,ambiguties_WL,'b.')
        ax.set_ylabel('Ambiguity resolution (%)')
        ax.set_xlabel('Time in years')

        ax2 = ax.twinx()
        ax2.plot(time_stamps,num_stations,'k.')
        ax2.set_ylabel("Number of stations",fontsize=8)
        ax2 = plot_tight(ax2)
        # format the ticks
        years    = mdates.YearLocator()   # every year
        months   = mdates.MonthLocator()  # every month
        monthsFmt = mdates.DateFormatter('%Y-%m')
        yearsFmt = mdates.DateFormatter('%Y')

        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        minX = gt.ydhms2mdt(1997,1,0.,0.,0.)
        maxX = gt.ydhms2mdt(2014,1,0.,0.,0.)
        ax.set_xlim([minX,maxX])
        fig.autofmt_xdate()
        
        ax = plot_tight(ax)
        txtX = gt.ydhms2mdt(2009,1,0.,0.,0.)
        ax.text(txtX,85,'Ambiguity resolution',color='blue',fontsize=8,fontweight='bold')
        ax.text(txtX,80,'Number of stations',color='black',fontsize=8,fontweight='bold')
        #plt.legend(fontsize=8,loc=4)
        plt.tight_layout()

        plt.show()

    if args.autcln and args.svnavFile:
        svdat = svnav.parseSVNAV(args.svnavFile)

        for autcln in autclns:
            outfile = 'SV_RESIDUALS_NET' + str(autcln['network']) + '.ND3'
            acln.consolidateNadir(svdat,autcln,outfile)



