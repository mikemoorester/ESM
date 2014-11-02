#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import re
import numpy as np

def getStationPos(apr_file, site):
    site_RGX = re.compile('^ '+site.upper()+'_')
    with open(apr_file,'r') as FID:
        for line in FID:
            if site_RGX.search(line):
                X = line[10:25].rstrip().lstrip()
                Y = line[25:40].rstrip().lstrip()
                Z = line[40:55].rstrip().lstrip()
                print(site,X,Y,Z)
                return np.array([float(X), float(Y) , float(Z)])
    return -1


#==========================================================================================================
if __name__ == "__main__":

    import os
    import argparse

    parser = argparse.ArgumentParser(prog='GamitAprioriFile',description='Parse GAMIT Apriori files')

    parser.add_argument('-f', '--apr_file', dest='apr_file', default='~/gg/tables/itrf08_comb.apr')
    parser.add_argument('-s', '--site', dest='site', default='',required=True)

    args = parser.parse_args()
    #=========================================
    args.apr_file = os.path.expanduser(args.apr_file)
    # find the station meta data for a particular site
    sitepos = getStationPos(args.apr_file, args.site)
    print("SITEPOS:",sitepos)
