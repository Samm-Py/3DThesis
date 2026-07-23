import numpy as np
from math import sqrt, sin, cos
import copy

def LoadScan(inFile = "3DThesis/TestInputs/Path.txt"):
    '''
    Load a scan file and get the lines from it
    Inputs:
        inFile - Location of Scan Path Input
    OutPuts:
        lines - The lines of the Scan Path Input
    '''

    # Open file for reading
    f = open(inFile, 'r')

    # Read all lines into a list
    lines = f.readlines()

    return lines

def ExportScan(lines, outFile = "3DThesis/TestInputs/Path.Final.txt"):
    '''
    Exports a scan file with the given lines
    Inputs:
        lines   - Scan path lines to print out
        outFile - Location of Scan Path Output
    '''

    # Open file for reading
    f = open(outFile, 'w')

    # Read all lines into a list
    f.writelines(lines)

def SegmentScan(lines, segmentSize=0.5):
    '''
    Divide the rasters of a scan path up into smaller rasters
    Inputs:
        lines - Text lines of the scan path
        segmentSize - Maximum size before segmenting (mm)
    Outputs:
        newLines - Text lines of the divided up scan path
    '''

    # Make newLines container
    newLines = []

    # Make begining and end of segment holders
    x0, y0 = 0, 0
    x1, y1 = 0, 0

    # Loop over lines
    for line in lines:
        # Turn line into array of values
        vals = line.strip().split("\t")
        # If the line is a raster with the power on, it should be split
        if (vals[0]=='0' and vals[4]=='1'):
            x1, y1 = float(vals[1]), float(vals[2])
            rasterDistance = sqrt((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0))
            # While the segment still needs to be cut
            while (rasterDistance>segmentSize):
                xp = x0 + (x1 - x0) * segmentSize / rasterDistance
                yp = y0 + (y1 - y0) * segmentSize / rasterDistance
                vals[1], vals[2] = str(xp), str(yp)
                # Make new line
                newLine = "\t".join(vals) + "\n"
                # Append to list of newLines
                newLines.append(newLine)
                # Set new x0, y0
                x0, y0 = xp, yp
                # Recalculate raster distance
                rasterDistance = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
            # Add on the smaller final segment
            vals[1], vals[2] = str(x1), str(y1)
            # Make new line
            newLine = "\t".join(vals) + "\n"
            # Append to list of newLines
            newLines.append(newLine)

        # If it isn't, just add it in
        else:
            newLines.append(line)

        # Set the next beginning to previous position
        if (vals[0]=='0' or vals[0]=='1'):
            x0, y0 = float(vals[1]), float(vals[2])

    return newLines

def MakeSquareRaster(V, H, S, R = 1e-6, outFile ="3DThesis/TestInputs/Path.txt"):
    '''
    Make a raster scan pattern in a cube.
    Inputs:
        V - Velocity (m/s)
        H - Hatch Spacing (mm)
        S - Square Length/Width (mm)
        R - Rest Time Between Consecutive Rasters (s)
        outFile - Location of Scan Path Output
    '''
    f = open(outFile, 'w')
    f.write('Mode\tX(mm)\tY(mm)\tZ(mm)\tPmod\tVel(m/s)/Time\n')
    scan_dir = 1

    # The x and y positions
    x, y = 0, 0
    while (y < S):
        # Write resting at begining of segment (short rest time
        f.write('{:0.0f}'.format(1) + '\t' + '{:0.3f}'.format(x) + '\t' + '{:0.3f}'.format(y) + '\t' + '{:0.0f}'.format(0) + '\t' + '{:0.0f}'.format(0) + '\t' + '{:0.6f}'.format(R) + '\n')
        # Move X to end of square
        x += S*scan_dir
        # Write the line scan
        f.write('{:0.0f}'.format(0) + '\t' + '{:0.3f}'.format(x) + '\t' + '{:0.3f}'.format(y) + '\t' + '{:0.0f}'.format(0) + '\t' + '{:0.0f}'.format(1) + '\t' + '{:0.6f}'.format(V) + '\n')
        # Increment Y
        y += H
        # Switch scan direction
        scan_dir*=-1

    f.close()

def MakeHalfSquareRaster(V, H, S, R = 1e-6, outFile = "3DThesis/TestInputs/Path.txt"):
    '''
    Make a raster scan pattern in a cube.
    Inputs:
        V - Velocity (m/s)
        H - Hatch Spacing (mm)
        S - Square Length/Width (mm)
        R - Rest Time Between Consecutive Rasters (s)
        outFile - Location of Scan Path Output
    '''
    f = open(outFile, 'w')
    f.write('Mode\tX(mm)\tY(mm)\tZ(mm)\tPmod\tVel(m/s)/Time\n')
    scan_dir = 1

    # The x and y positions
    x, y = 0, 0
    while (y < S/2.0):
        # Write resting at begining of segment (short rest time
        f.write('{:0.0f}'.format(1) + '\t' + '{:0.3f}'.format(x) + '\t' + '{:0.3f}'.format(y) + '\t' + '{:0.0f}'.format(0) + '\t' + '{:0.0f}'.format(0) + '\t' + '{:0.6f}'.format(R) + '\n')
        # Move X to end of square
        x += S*scan_dir
        # Write the line scan
        f.write('{:0.0f}'.format(0) + '\t' + '{:0.3f}'.format(x) + '\t' + '{:0.3f}'.format(y) + '\t' + '{:0.0f}'.format(0) + '\t' + '{:0.0f}'.format(1) + '\t' + '{:0.6f}'.format(V) + '\n')
        # Increment Y
        y += H
        # Switch scan direction
        scan_dir*=-1

    # Write resting at begining of segment (short rest time
    f.write('{:0.0f}'.format(1) + '\t' + '{:0.3f}'.format(x) + '\t' + '{:0.3f}'.format(y) + '\t' + '{:0.0f}'.format(0) + '\t' + '{:0.0f}'.format(0) + '\t' + '{:0.6f}'.format(R) + '\n')
    # Move X to end of square
    x += (S/2.0)*scan_dir
    # Write the line scan
    f.write('{:0.0f}'.format(0) + '\t' + '{:0.3f}'.format(x) + '\t' + '{:0.3f}'.format(y) + '\t' + '{:0.0f}'.format(0) + '\t' + '{:0.0f}'.format(1) + '\t' + '{:0.6f}'.format(V) + '\n')

    f.close()

def MakeTriangleRaster(V, H, S, R = 1e-6, outFile ="3DThesis/TestInputs/Path.txt"):
    '''
    Make a raster scan pattern in a cube.
    Inputs:
        V - Velocity (m/s)
        H - Hatch Spacing (mm)
        S - Equilateral Triangle Side Length (mm)
        R - Rest Time Between Consecutive Rasters (s)
        outFile - Location of Scan Path Output
    '''
    f = open(outFile, 'w')
    f.write('Mode\tX(mm)\tY(mm)\tZ(mm)\tPmod\tVel(m/s)/Time\n')
    scan_dir = 1

    # The x and y positions
    x, y = 0, 0
    while (y < S*sqrt(3.0)/2.0):
        # Write resting at begining of segment (short rest time
        f.write('{:0.0f}'.format(1) + '\t' + '{:0.3f}'.format(x) + '\t' + '{:0.3f}'.format(y) + '\t' + '{:0.0f}'.format(0) + '\t' + '{:0.0f}'.format(0) + '\t' + '{:0.6f}'.format(R) + '\n')
        # Move X to end of square
        x += scan_dir*(S-2*y/sqrt(3))
        # Write the line scan
        f.write('{:0.0f}'.format(0) + '\t' + '{:0.3f}'.format(x) + '\t' + '{:0.3f}'.format(y) + '\t' + '{:0.0f}'.format(0) + '\t' + '{:0.0f}'.format(1) + '\t' + '{:0.6f}'.format(V) + '\n')
        # Increment Y
        y += H
        # Switch scan direction
        scan_dir*=-1
        # Also adjust x accordingly
        x += scan_dir*H/sqrt(3)

    f.close()

def GetEndPosition(inFile = "3DThesis/TestInputs/Path.txt"):
    '''
    Get the last position for the beam in a scan file
    Inputs:
        inFile - Location of Scan Path Input
    OutPuts:
        x, y - 2D Position of Heat Source
    '''

    # Open file for reading
    f = open(inFile, 'r')

    # Read all lines into a list
    lines = f.readlines()

    # Access last line
    endLine = lines[-1]

    # Split up last line
    cols = endLine.split('\t')

    # Get x,y position
    x, y = float(cols[1]), float(cols[2])

    return x, y

def RotateTranslateLastRasterToX0(lines):
    '''
    Rotate a scan pattern so the last raster is in the x-direction (for width and depth collection)
    Inputs:
        lines - lines to rotate
    Output:
        rotLines - rotates lines
    '''

    # Get the last two lines
    prevLine, lastLine = lines[-2], lines[-1]

    # Get the last two values
    prevVals, lastVals = prevLine.strip().split("\t"), lastLine.strip().split("\t")

    # Get x0, y0
    x0, y0 = float(prevVals[1]), float(prevVals[2])

    # Get x1, y1
    x1, y1 = float(lastVals[1]), float(lastVals[2])

    # Calculate the angle of rotation in radians
    angle = -np.arctan2(y1 - y0, x1 - x0)

    rotLines = []
    for line in lines:
        # Set rotLine to line
        rotLine = line

        # Read values from line
        vals = rotLine.strip().split("\t")

        # As long as it isn't the header line
        if(vals[0] == '0' or vals[0] == '1'):
            # Get the original x,y position
            x, y = float(vals[1]) - x1, float(vals[2]) - y1

            # Apply the rotation to all points
            xRot, yRot = x * cos(angle) - y * sin(angle), x * sin(angle) + y * cos(angle)

            # Set new values
            vals[1], vals[2] = str(xRot), str(yRot)

            # Make new line
            rotLine = "\t".join(vals)+"\n"

        # Append line
        rotLines.append(rotLine)

    return rotLines

def ChangeLastRaster(lines,power,speed):
    '''
    Change the power and velocity of the last line in a raster patter
    Inputs:
        lines   - Input scan lines
        power   - New power
        speed   - New speed
    Outputs:
        adjLines    - Scan lines with new power and speed
    '''

    # Deep copy the lines
    adjLines = copy.deepcopy(lines)

    # Get last line (line to adjust)
    adjLine = adjLines[-1]

    # Get values to adjust
    adjVals = adjLine.strip().split("\t")

    # Adjust values
    adjVals[4], adjVals[5] = str(power), str(speed)

    # Add to lines
    newLine = "\t".join(adjVals)+"\n"

    # Change lines
    adjLines[-1] = newLine

    return adjLines

def ConvertToSpotRest(lines):
    '''
    Converts a scan path so that rests (beam off) are in mode 1
    Inputs:
        lines - Text lines of the scan path
    Outputs:
        newLines - Text lines of the modified scan path
    '''

    # Make newLines container
    newLines = []

    # Make begining and end of segment holders
    x0, y0 = 0, 0
    x1, y1 = 0, 0

    # Loop over lines
    for line in lines:
        # Turn line into array of values
        vals = line.strip().split("\t")
        # Make sure all z-coords are 0

        # If the line is a raster with the power off, it should be turned into a spot rest
        if (vals[0] == '0' and vals[4] == '0'):
            # Get everything in mm and mm/s
            x1, y1, speed = float(vals[1]), float(vals[2]), float(vals[5])*1000.0
            restDistance = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
            restTime = restDistance/speed
            # Adjust the segment
            vals[0], vals[5] = '1', str(restTime)

        # Set the next beginning to previous position
        if (vals[0] == '0' or vals[0] == '1'):
            x0, y0 = float(vals[1]), float(vals[2])
            vals[3] = '0'

        # Make new line
        newLine = "\t".join(vals) + "\n"
        # Append to list of newLines
        newLines.append(newLine)

    return newLines
def AddTurnDwell(lines, dwellTime):
    '''
    Add extra beam-off dwell time to every rest (mode 1) segment that follows
    a powered raster -- i.e. the turnarounds of a serpentine pattern. The
    initial positioning spot (before any raster) is left untouched.

    Purpose: at tight hatch spacings the beam revisits material adjacent to
    the previous track before that track's melt has solidified, which makes a
    fixed melt-pool width target infeasible there (the "pool" includes
    still-liquid neighbour residue). Dwelling at the turn lets the residue
    solidify before the next track starts.

    Inputs:
        lines     - Text lines of the scan path (after ConvertToSpotRest)
        dwellTime - Extra dwell to add at each turnaround (s)
    Outputs:
        newLines  - Text lines of the modified scan path
    '''

    # Deep copy the lines
    newLines = copy.deepcopy(lines)

    # Only rests AFTER the first powered raster are turnarounds
    seenRaster = False

    # Loop over lines
    for i in range(len(newLines)):
        vals = newLines[i].strip().split("\t")

        # Skip the header or malformed lines
        if (vals[0] != '0' and vals[0] != '1'):
            continue

        # Note when the first raster has been passed
        if (vals[0] == '0'):
            seenRaster = True

        # Extend the dwell of every subsequent rest
        if (vals[0] == '1' and seenRaster):
            vals[5] = str(float(vals[5]) + dwellTime)
            newLines[i] = "\t".join(vals) + "\n"

    return newLines
