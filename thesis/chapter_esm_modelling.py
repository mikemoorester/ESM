from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats.stats import pearsonr,kendalltau

#vel_light = 299792458.0
#fL1 = 10.23e6*77.*2. 
#fL2 = 10.23e6*60.*2. 
#wL1 = vel_light/fL1 
#wL2 = vel_light/fL2
#lcl1 = 1./(1.-(fL2/fL1)**2)
#lcl2 = -(fL2/fL1)/(1.-(fL2/fL1)**2)

def plotFontSize(ax,fontsize=8):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
    return ax

def pcoBias(args):
    fig = plt.figure(figsize=(3.62, 2.76))
    fig.canvas.set_window_title('pco_pcv_correlation')
    ax = fig.add_subplot(111)

    plt.rc('text', usetex=True)

    nadir = np.linspace(0,14,141)
    for dr in [0.1, 0.5, 1.0]:
        dpcv = -dr *(1 - np.cos(np.radians(nadir)))
        ax.plot(nadir,dpcv)


    ax.set_ylabel('\Delta' ' Satellite PCV (m)')
    ax.set_xlabel('Nadir angle' r'($\displaystyle^\circ$)')# ($^\circ$)')
    ax.set_xlim([0,14])
    ax.legend([r'$\Delta r$ = 0.1 m', r'$\Delta r$ = 0.5 m',r'$\Delta r$ = 1.0 m'],fontsize=8,loc='best')
    ax = plotFontSize(ax,8)
    plt.tight_layout()

    if args.plot_save:
        plt.savefig('pco_pcv_correlation.eps')
        plt.close()
    return 1 

if __name__ == "__main__":

    import argparse 
    parser = argparse.ArgumentParser(prog='chapter_esm_modelling',description='Create some basic plots for thesis chapter')
    parser.add_argument("--pco", dest="pco",action="store_true",default=False,
                        help="Produce a plot of pco bias appearing as a pcv bias (correlation)")

    parser.add_argument("--ps","--plot_save", dest="plot_save",action="store_true",default=False,
            help="Save plots in eps format")

    args = parser.parse_args()

    if args.pco:
        pcoBias(args)

    if not args.plot_save:
        plt.show()
