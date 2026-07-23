"""Melt-pool geometry and OTI support-function sensitivity extraction.

This is the publication workflow's self-contained copy of the original
``OptimizeRasters/simulate.py`` helper module. Study drivers pass explicit
case and snapshot paths; legacy defaults remain for compatibility.
"""

import os
import subprocess
import pandas as pd
import numpy as np
from skimage import measure
from scipy import ndimage

# Resolve defaults independently of the caller's working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEMOS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_DEMOS)

# 3DThesis binary and the case directory the optimizer reads/writes. Both are
# overridable via environment variables:
#   THESIS_BIN  - path to the 3DThesis executable. Default is the plain-double
#                 build (fastest); point it at ../build-oti/bin/3DThesis to get
#                 the OTI derivative columns (dT_dQ, ...) in the snapshot CSV.
#   THESIS_CASE - case directory containing ParamInput.txt and the static
#                 inputs; the optimizer writes Path.txt and Domain.txt here and
#                 3DThesis writes Data/TestSim.Snapshot.00.csv here.
THESIS_BIN = os.path.abspath(os.environ.get(
    "THESIS_BIN", os.path.join(_ROOT, "build", "bin", "3DThesis")))
CASE_DIR = os.path.abspath(os.environ.get(
    "THESIS_CASE", os.path.join(_DEMOS, "raster", "cases", "snapcase")))

def Run(stdout=False):
    '''
    Run 3DThesis on the case directory. Reads the Path.txt and Domain.txt the
    optimizer just wrote there and produces Data/TestSim.Snapshot.00.csv.
    Inputs:
        stdout - if True, let 3DThesis print to the terminal (default: silence)
    '''
    out = None if stdout else subprocess.DEVNULL
    subprocess.run([THESIS_BIN, "./ParamInput.txt"],
                   cwd=CASE_DIR, stdout=out, stderr=out, check=True)

def ExtractMPDims(inFile = "3DThesis/TestInputs/Data/TestSim.Snapshot.00.csv"):
    '''
    Extract the Meltpool Width and Depth
    Inputs:
        inFile - Location of Data
    Outputs:
        width, depth - Width and Depth of the Meltpool
    '''
    # Get data from file
    data = pd.read_csv(inFile)

    # Read in Coordinate Values
    x,y,z = data['x'], data['y'], data['z']

    # Get maximum width
    width = np.max(y)-np.min(y)

    # Get maximum depth
    depth = np.max(np.abs(z))

    return width, depth

def ExtractMPDims_Interp(inFile = "3DThesis/TestInputs/Data/TestSim.Snapshot.00.csv"):
    '''
    Extract the Meltpool Width and Depth
    Inputs:
        inFile - Location of Data
    Outputs:
        width, depth - Width and Depth of the Meltpool
    '''
    # Get data from file
    data = pd.read_csv(inFile)

    # Set the value for the level set
    isovalue = 1733  # Replace with the melting temperature

    # Read in Coordinate Values
    x,y,z,T = data['x'].values, data['y'].values, data['z'].values, data['T'].values

    # If no values are melting, width and depth are zero
    if (np.sum(T>isovalue)==0):
        width, depth = 0,0
        return width, depth


    # Get mesh info
    x_num, y_num, z_num = np.unique(x).size, np.unique(y).size, np.unique(z).size
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)
    x_res, y_res, z_res = (x_max - x_min) / (x_num - 1), (y_max - y_min) / (y_num - 1), (z_max - z_min) / (z_num - 1)

    # Reshape temperature field
    T = T.reshape(x_num,y_num,z_num)


    # Find the level set using Marching Cubes
    vertices, _, _, _ = measure.marching_cubes(T, isovalue)

    # Get widths and depth
    width_1 = y_min + np.min(vertices[:,1])*y_res
    width_2 = y_min + np.max(vertices[:,1])*y_res
    depth =  -(z_min + np.min(vertices[:, 2])*z_res)

    # Set width to me minimum of the 2 widths
    width = min(abs(width_1),abs(width_2))

    return width, depth

def ExtractMPDims_Span(inFile = "3DThesis/TestInputs/Data/TestSim.Snapshot.00.csv"):
    '''
    Extract melt-pool geometry as a *single asymmetric pool*.

    The connected-component test (merge_check.py) showed the T>isovalue region
    stays one connected blob along a tight hatch; as neighbouring tracks preheat
    one side, the pool leans/balloons toward the hot track rather than splitting.
    A single min()/max() edge therefore misrepresents "width": the cold edge is
    nearly constant while the hot edge runs away. We report:

        width  = (y_right - y_left)/2   -- half-span, a radius-like scalar that
                                           is the lateral analogue of depth (both
                                           are one-sided distances from a centre)
        depth  =  downward extent of the isosurface
        asym   = (y_right + y_left)     -- signed lateral shift of the pool
                                           centre off the track centreline; the
                                           diagnostic for directional ballooning.
                                           asym~=0 => symmetric; asym>0 => leaning
                                           toward +y (the preheated neighbour).

    Inputs:
        inFile - Location of Data
    Outputs:
        width, depth, asym  (all in metres; asym signed)
    '''
    data = pd.read_csv(inFile)
    isovalue = 1733  # melting temperature used throughout the study

    x,y,z,T = data['x'].values, data['y'].values, data['z'].values, data['T'].values

    # No melt -> everything zero
    if (np.sum(T>isovalue)==0):
        return 0.0, 0.0, 0.0

    x_num, y_num, z_num = np.unique(x).size, np.unique(y).size, np.unique(z).size
    x_min, y_min, z_min = np.min(x), np.min(y), np.min(z)
    x_max, y_max, z_max = np.max(x), np.max(y), np.max(z)
    y_res = (y_max - y_min) / (y_num - 1)
    z_res = (z_max - z_min) / (z_num - 1)

    T = T.reshape(x_num,y_num,z_num)
    vertices, _, _, _ = measure.marching_cubes(T, isovalue)

    # Signed lateral edges of the isosurface (metres, relative to y=0 centreline)
    y_left  = y_min + np.min(vertices[:,1])*y_res
    y_right = y_min + np.max(vertices[:,1])*y_res

    width = 0.5*(y_right - y_left)     # half-span
    asym  = (y_right + y_left)         # centre offset (signed)
    depth = -(z_min + np.min(vertices[:,2])*z_res)

    return width, depth, asym

def ExtractIsoSupportSensitivities(inFile = "3DThesis/TestInputs/Data/TestSim.Snapshot.00.csv",
                                   isovalue = 1733, directions = None):
    '''
    Generic isosurface support-function sensitivities. Nothing about the
    control set or the probed directions is hardcoded here.

    For a unit direction e, the support point x*(e) is the point of the
    T = isovalue surface extremal in e (the argmax of e.x over the surface).
    There the surface is tangent to the plane normal to e -- grad T is
    parallel to e -- and the implicit-function/envelope theorem gives, for
    ANY scalar model parameter p carried by the snapshot as an OTI column
    dT_dp, the exact first-order sensitivity of the support value
    h(e) = e . x*(e):

        dh(e)/dp = - (dT/dp) / (grad T . e)     at x*(e).

    The spatial gradient (dT_dx, dT_dy, dT_dz) comes from the same snapshot,
    and the controls are DISCOVERED as every remaining dT_d<name> column, so
    a 3DThesis build with more or different seeded design variables is picked
    up with no change here.

    Inputs:
        inFile     - snapshot CSV (OTI build)
        isovalue   - isotherm defining the surface
        directions - {key: direction vector}; default is the melt-pool set
                     {'y+': +y, 'y-': -y, 'z-': -z}
    Outputs: (h, dh, controls)
        h[key]        - support value e . x*  (metres)
        dh[key][p]    - d h(e) / d p for every control p
        controls      - sorted list of discovered control names
    Raises KeyError if the CSV has no OTI columns (plain-double build).
    '''
    data = pd.read_csv(inFile)

    x,y,z,T = data['x'].values, data['y'].values, data['z'].values, data['T'].values
    if (np.sum(T>isovalue)==0):
        raise ValueError("no melt: T never exceeds isovalue")

    if directions is None:
        directions = {'y+': (0.0, 1.0, 0.0),
                      'y-': (0.0, -1.0, 0.0),
                      'z-': (0.0, 0.0, -1.0)}

    mins = np.array([np.min(x), np.min(y), np.min(z)])
    nums = np.array([np.unique(x).size, np.unique(y).size, np.unique(z).size])
    res  = (np.array([np.max(x), np.max(y), np.max(z)]) - mins) / (nums - 1)

    shape = tuple(nums)
    T3 = T.reshape(shape)
    # Spatial gradient grids + discovered control grids (KeyError on the
    # gradient columns => not an OTI snapshot).
    grad = [data['dT_d'+ax].values.reshape(shape) for ax in ('x','y','z')]
    controls = sorted(c[len('dT_d'):] for c in data.columns
                      if c.startswith('dT_d') and c not in ('dT_dx','dT_dy','dT_dz'))
    if not controls:
        raise KeyError("no control derivative columns (dT_d*) in snapshot")
    ctrl_grids = {p: data['dT_d'+p].values.reshape(shape) for p in controls}

    # Support point per direction: isosurface vertex (fractional-index space)
    # extremal in e. The argmax must be taken in METRIC space (index space is
    # anisotropic when the axis resolutions differ), hence the e*res scaling.
    vertices, _, _, _ = measure.marching_cubes(T3, isovalue)
    keys  = list(directions)
    evecs = {k: np.asarray(directions[k], float) for k in keys}
    sup   = {k: vertices[np.argmax(vertices @ (evecs[k]*res))] for k in keys}

    # Trilinear sampling of the derivative grids at the (fractional) support
    # points, all points in one call per field.
    pts = np.stack([sup[k] for k in keys], axis=1)   # (3 coords, npts)
    def at_pts(F):
        return ndimage.map_coordinates(F, pts, order=1)
    g  = np.stack([at_pts(G) for G in grad])         # (3, npts)
    cs = {p: at_pts(ctrl_grids[p]) for p in controls}

    h, dh = {}, {}
    for j, k in enumerate(keys):
        e = evecs[k]
        h[k]  = float(e @ (mins + sup[k]*res))
        gn    = float(e @ g[:, j])                   # grad T . e at x*(e)
        dh[k] = {p: float(-cs[p][j]/gn) for p in controls}
    return h, dh, controls

def ExtractMPSensitivities(inFile = "3DThesis/TestInputs/Data/TestSim.Snapshot.00.csv"):
    '''
    Melt-pool geometry AND its exact sensitivities w.r.t. the process
    controls: the QoI layer over ExtractIsoSupportSensitivities.

    QoIs match ExtractMPDims_Span, expressed in support values
    h(e) = e . x*(e) of the T = isovalue surface:

        width = (y_right - y_left)/2 = (h(+y) + h(-y))/2
        depth = -z_bottom            =  h(-z)
        asym  = (y_right + y_left)   =  h(+y) - h(-y)

    and the sensitivities are the same linear combinations of dh(e)/dp.
    The control set is whatever the snapshot carries (dT_dQ and dT_dsig for
    the current 3DThesis build), producing dwidth_dp, ddepth_dp, dasym_dp
    per control p.  Raises KeyError if the CSV has no OTI columns
    (plain-double build).
    '''
    h, dh, controls = ExtractIsoSupportSensitivities(inFile)

    out = {
        'width': 0.5*(h['y+'] + h['y-']),
        'asym' : h['y+'] - h['y-'],
        'depth': h['z-'],
    }
    for p in controls:
        out['dwidth_d'+p] = 0.5*(dh['y+'][p] + dh['y-'][p])
        out['dasym_d'+p]  = dh['y+'][p] - dh['y-'][p]
        out['ddepth_d'+p] = dh['z-'][p]
    return out

def UpdateDomain(x,y,res,buf,outFile = "3DThesis/TestInputs/Domain.txt"):
    '''
    Update the domain to center around the last point in a scan path
    Inputs:
        x,y - Center of Domain      (m)
        res - Resolution of Domain  (m)
        buf   - Buffer of Domain      (m)
        outFile - Location of Domain Output
    '''

    # Calulate minnimum and maximum
    # x_min, x_max = x - buf, x + buf
    # y_min, y_max = y - buf, y + buf
    # z_min, z_max = -buf, 0

    x_min, x_max = x - 2*buf, x
    y_min, y_max = y - buf, y + buf
    z_min, z_max = -buf/2, 0

    # Open the file for writing
    f = open(outFile, 'w')

    # Write the X Part of the File
    f.write('X\n')
    f.write('{\n')
    f.write('\tMin\t' + str(x_min) + "\n")
    f.write('\tMax\t' + str(x_max) + "\n")
    f.write('\tRes\t' + str(res) + "\n")
    f.write('}\n')

    # Write the Y Part of the File
    f.write('Y\n')
    f.write('{\n')
    f.write('\tMin\t' + str(y_min) + "\n")
    f.write('\tMax\t' + str(y_max) + "\n")
    f.write('\tRes\t' + str(res) + "\n")
    f.write('}\n')

    # Write the Y Part of the File
    f.write('Z\n')
    f.write('{\n')
    f.write('\tMin\t' + str(z_min) + "\n")
    f.write('\tMax\t' + str(z_max) + "\n")
    f.write('\tRes\t' + str(res) + "\n")
    f.write('}\n')

    # Close the file
    f.close()

def ExtractData_2D(inFile,outFile,keys="",surface=""):
    if (len(keys)!=len(surface)):
        print("keys and surface not same length")
        exit(9)

    simData = pd.read_csv(inFile)
    simData = simData.groupby(['x', 'y'])

    simDataTemp = simData.first()  # .apply(lambda df:df.z.argmin())
    simDataTemp.reset_index(inplace=True)
    x, y = simDataTemp['x'].values, simDataTemp['y'].values

    data2D = pd.DataFrame()
    data2D['x'] = x
    data2D['y'] = y

    for i in range(len(keys)):
        if (surface[i]=="top"):
            simDataTemp = simData.last()  #.apply(lambda df:df.z.argmin())
            simDataTemp.reset_index(inplace=True)
            data2D[keys[i]+".top"] = simDataTemp[keys[i]].values
        elif (surface[i]=="bot"):
            simDataTemp = simData.first()  # .apply(lambda df:df.z.argmin())
            simDataTemp.reset_index(inplace=True)
            data2D[keys[i]+".bot"]=simDataTemp[keys[i]].values

    data2D.to_csv(outFile,sep=",",index=False,index_label=False)
    return
