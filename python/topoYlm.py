#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import skimage.measure

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap

import healpy as hp

def create_grid(ntheta, nphi):
    dtheta, dphi = np.pi/ntheta, 2*np.pi/nphi
    theta_grid = (np.arange(ntheta) + 0.5)*dtheta
    phi_grid = (np.arange(nphi) + 0.5)*dphi
    return np.meshgrid(theta_grid, phi_grid, sparse=True, indexing='ij')

def latlon_to_healpix(data,downsampling, nside, verbose=False):
    if downsampling is not None:
        ntheta, nphi = data.shape
        if ntheta % downsampling != 0 or nphi % downsampling != 0:
            raise ValueError('Downsampling factor {} does not evenly divide ntheta,phi = {},{}'.format(
                downsampling, ntheta, phi))
        dsdata = skimage.measure.block_reduce(data, block_size=(downsampling, downsampling), func=np.mean)
        if verbose:
            print 'Downsampled from {} to {}'.format((ntheta, nphi), dsdata.shape)
    else:
        dsdata = data
    # Calculate angles at each data point.
    theta, phi = create_grid(*dsdata.shape)
    # Calculate the corresponding pixel indices.
    npix = hp.nside2npix(nside)
    if verbose:
        print 'Mapping {} grid points to {} pixels.'.format(dsdata.size,npix)
    pixid = hp.pixelfunc.ang2pix(nside, theta, phi)
    # Average the input data that falls into each pixel.
    pixsum = np.bincount(pixid.flat, weights=dsdata.flat, minlength=npix)
    pixcount = np.bincount(pixid.flat, minlength=npix)
    if np.min(pixcount) == 0:
        raise ValueError('Some pixels are empty: lower downsampling or nside.')
    pixsum /= pixcount
    return pixsum

def build_ylm_map(alm, ell_max, nside):
    ell_max_in = hp.sphtfunc.Alm.getlmax(alm.size)
    ell,m = hp.sphtfunc.Alm.getlm(ell_max_in)
    return hp.sphtfunc.alm2map(alm[ell <= ell_max], nside=nside)

def get_theta_phi_grid(nside):
    npix = hp.pixelfunc.nside2npix(nside)
    return hp.pixelfunc.pix2ang(nside, np.arange(npix))

def create_cmap(control_points):
    # Find the min/max values.
    min_value = control_points[0][0]
    max_value = control_points[-1][0]
    scale = 1.0/(max_value - min_value)
    # Rescale values to 0-1.
    rescaled = [(scale*(value - min_value),color) for (value,color) in control_points]
    rescaled[-1] = (1.0,rescaled[-1][1])
    return LinearSegmentedColormap.from_list('topo_cmap',rescaled)

def main():
    # Initialize and parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action='store_true',
        help = 'Provide verbose output.')
    parser.add_argument('-i', '--input', type=str,
        default='/Data/ETOPO1/etopo1_bed_c_i2.bin',
        help='Location of i2-binary ETOPO1 input data.')
    parser.add_argument('-o', '--output', type=str,
        default='/Users/david/Desktop/Ylm/ellmax_{}.png',
        help='Pattern for saving frames.')
    parser.add_argument('--frames', type=str, default='32:34',
        help='Frames to generate in slice notation begin:end:stride.')
    parser.add_argument('--downsampling', type=int, default=4,
        help='Downsampling of lat,lon grid to apply before resampling to healpix.')
    parser.add_argument('--nside', type=int, default=256,
        help='Healpix resolution parameter, must be a power of 2.')
    args = parser.parse_args()

    # Load the ETOPO1 data.
    if args.verbose:
        print 'Loading ETOPO1 data.'
    data = np.fromfile(args.input, dtype=np.int16).reshape(10800, 21600)
    assert np.count_nonzero(data == -(1 << 15)) == 0

    # Resample to a healpix map.
    if args.verbose:
        print 'Resampling to helpix map with nside={}.'.format(args.nside)
    pixdata = latlon_to_healpix(data, downsampling=args.downsampling, nside=args.nside,
        verbose=args.verbose)

    # Calculate the spherical harmonic coefficients.
    if args.verbose:
        print 'Calculating spherical harmonics.'
    alm = hp.sphtfunc.map2alm(pixdata)

    # Initialize the map projection.
    m2d = Basemap(projection='moll', lon_0=0, resolution=None)

    # Precompute the lon,lat values at the center of each pixel
    theta, phi = get_theta_phi_grid(nside=args.nside)
    x2d, y2d = m2d(x=np.rad2deg(phi + np.pi), y=90-np.rad2deg(theta))

    # Initialize a terrain color map.
    topo_cmap = create_cmap([
        (-6000., '#0000FF'),
        (-4000., '#1E90FF'),
        (-1000., '#00CED1'),
        (   -1., '#99FFFF'),
        (    1., '#F0E68C'),
        (  100., '#008000'),
        ( 1000., '#228B22'),
        ( 1500., '#BDB76B'),
        ( 2000., '#B8860B'),
        ( 4000., '#708090'),
        ( 4500., '#C0C0C0'),
        ( 5500., '#FFFFFF')
    ])

    frames = range(*map(int,args.frames.split(':')))
    for ell_max in frames:

        print 'Generating frame for ell_max = {}...'.format(ell_max)
        lores = build_ylm_map(alm, ell_max, nside=args.nside)

        fig = plt.figure(figsize=(10.24,7.68), dpi=100.)
        m2d.pcolor(x=x2d, y=y2d, data=lores, tri=True, shading='gouraud',
            cmap=topo_cmap, vmin=-6000, vmax=+5000)
        m2d.drawmeridians(np.arange(0,360,30));
        m2d.drawparallels(np.arange(-90,90,30));

        angular_resolution = 360./ell_max
        distance_resolution = 2*np.pi*6371e3/ell_max
        label1 = '$\mathrm{resolution} = %.2f$$^\circ$'%angular_resolution
        label2 = '$(%.0f\,\mathrm{m})$'%distance_resolution
        label3 = ('$\sum_{\ell=0}^{\ell_{\mathrm{max}}}\,\sum_{m=-\ell}^{+\ell}\,a_{\ell m} Y_{\ell}^{m}(\\theta,\phi)\;,'+
            ('\;\ell_{\mathrm{max}} = %4d$'%ell_max))

        plt.annotate(label1,xy=(0,0),xytext=(0.05,0.88),xycoords='figure fraction',textcoords='figure fraction',
                     size=40,weight='bold',horizontalalignment='left',verticalalignment='bottom');
        plt.annotate(label2,xy=(0,0),xytext=(0.95,0.88),xycoords='figure fraction',textcoords='figure fraction',
                     size=40,weight='bold',horizontalalignment='right',verticalalignment='bottom');
        plt.annotate(label3,xy=(0,0),xytext=(0.12,0.01),xycoords='figure fraction',textcoords='figure fraction',
                     size=36,weight='bold',horizontalalignment='left',verticalalignment='bottom');
        plt.tight_layout();
        plt.savefig(args.output.format(ell_max),dpi=100.)
        plt.close();

if __name__ == '__main__':
    main()
