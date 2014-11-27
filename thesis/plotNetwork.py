from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

import GamitAprioriFile as gapr

import gpsCoords as gcoord
import geodetic

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog='plotNetwork',description='Create a plot of the network')

    parser.add_argument('--apr',dest='apr_file',help="Location of coordinate file - GAMIT .apr format" )
    parser.add_argument('-s','--sites',dest='sites',nargs='+',help="Sites to plot (4 charcater ID)")

    args = parser.parse_args()

    lons = []
    lats = []
    # get the reference frame parameters
    a,b,e2,finv = geodetic.refell('WGS84')

    for siteID in args.sites:
        sitepos = gapr.getStationPos(args.apr_file,siteID)
        dphi,dlambda,h = gcoord.togeod(a,finv,sitepos[0],sitepos[1],sitepos[2] )
        lats.append(dphi)
        lons.append(dlambda)

    # set up orthographic map projection with
    # perspective of satellite looking down at 50N, 100W.
    # use low resolution coastlines.
    #map = Basemap(projection='ortho',lat_0=-45,lon_0=140,resolution='l')
    #map = Basemap(projection='hammer',lat_0=-45,lon_0=10,resolution='l')
    map = Basemap(projection='robin',area_thresh = 1000.0,lat_0=-45,lon_0=10,resolution='l')

    # draw coastlines, country boundaries, fill continents.
    map.drawcoastlines(linewidth=0.25)
    map.drawcountries(linewidth=0.25)
    map.fillcontinents(color='gray',lake_color='aqua')
    #map.fillcontinents(color='gray')
    #map.bluemarble()
    # draw the edge of the map projection region (the projection limb)
    map.drawmapboundary(fill_color='aqua')

    # draw lat/lon grid lines every 30 degrees.
    map.drawmeridians(np.arange(0,360,30))
    map.drawparallels(np.arange(-90,90,30))

    # compute native map projection coordinates of lat/lon grid.
    #x, y = map(np.array(lons)*180./np.pi, np.array(lats)*180./np.pi)
    x, y = map(np.array(lons), np.array(lats))

    map.plot(x,y,'ro',markersize=6)
    plt.show()

