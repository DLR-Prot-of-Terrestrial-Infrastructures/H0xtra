# Copyright (c) 2025 German Aerospace Center - Deutsches Zentrum fuer
# Luft- und Raumfahrt e. V. (DLR)
#
# Author(s):        Jonas Gunkel jonas.gunkel@dlr.de
# Date of creation: 2025-06-01

"""
This file contains the code for generating the POI maps
"""

import pandas as pd
import numpy as np
import os


def create_poi_matrix(file_path, cats=85):
    """
    Transform POI data into a matrix
    """

    if not isinstance(file_path, list):
        file_path = [file_path]

    mtrx = np.zeros((len(file_path), 200, 200, cats))

    for (file_pth, id) in zip(file_path, range(len(file_path))):

        data = pd.read_csv(file_pth)

        # populate POI map
        for _, row in data.iterrows():
            x, y, c, k = int(row['x']), int(row['y']), int(row['category']), row['POI_count']
            mtrx[id, x-1, y-1, c-1] = k


    cities = [1, 2, 3, 4]
    matrices = np.array([{'map': arr, 'city': i} for arr, i in zip([mtrx[i,:,:,:] for i in range(4)], cities)])
        
    np.save('maps/maps.npy', matrices)


def create_poi_matrix_coords(file_path, cats):
    """
    Transform POI data into a matrix
    """

    if not isinstance(file_path, list):
        file_path = [file_path]

    mtrx = np.zeros((len(file_path), 200, 200, cats))

    # define coords
    coords = np.tile(np.arange(200), (200,1))
    coords = np.stack((coords.transpose(), coords), axis=-1)
    coords = np.tile(coords, (len(file_path), 1, 1, 1))

    for (file_pth, id) in zip(file_path, range(len(file_path))):

        data = pd.read_csv(file_pth)

        # populate POI map
        for _, row in data.iterrows():
            x, y, c, k = int(row['x']), int(row['y']), int(row['category']), row['POI_count']
            mtrx[id, x-1, y-1, c-1] = k

    mtrx = np.concat((coords, mtrx), axis=-1)

    cities = [1, 2, 3, 4]

    assert len(cities) == len(file_path)

    matrices = np.array([{'map': arr, 'city': i} for arr, i in zip([mtrx[i,:,:,:] for i in range(4)], cities)])
        
    np.save('maps/maps_coords.npy', matrices)


if __name__ == '__main__':

    # number of used POI categories
    cats = 85

    # paths to raw data
    files = ['raw/POIdata_cityA.csv',
    'raw/POIdata_cityB.csv',
    'raw/POIdata_cityC.csv',
    'raw/POIdata_cityD.csv']

    os.makedirs('maps', exist_ok=True)

    create_poi_matrix(files, cats)
    create_poi_matrix_coords(files, cats)