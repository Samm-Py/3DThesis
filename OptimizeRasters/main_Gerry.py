import numpy as np
import pandas as pd
import subprocess
import scan as Scan
import simulate as Sim
from math import sqrt, log10
import time
'''
In this section:
- Define things
'''
# Iterative Search Depth for Velocity
iterDepth = 6

# Size of segments to optimize (mm)
segSize = 0.5

# Starting resolution for power and speed
powRes = 0.5
speedRes = 0.1

# Velocity (m/s), Hatch Spacing (mm), and Square Size (mm) found to be "good"
V = 0.7
H = 0.14
S = 10

# Sim Resolution and Domain Buffer
res = 10.0e-6
buf = 0.4e-3

# Define cost function
def Cost(width,depth,width_target,depth_target):

    # Cost is simple distance
    deepCost = (1-depth/depth_target)**2

    # Depth Weight
    widthCost = (1-width/width_target)**2

    # Total Cost weights depth more heavily
    cost = widthCost + 2*deepCost

    return cost

'''
In this section:
- Want to get ideal MP dimensions from a simulated raster
'''

# # Make Raster Patter with Desired Parameters
# Scan.MakeHalfSquareRaster(V=V,H=H,S=S)

# Load up desired scan path to change
scanPath_target = Scan.LoadScan(inFile = "Path_Reference.txt")

# Change scan path to correct format
scanPath_target = Scan.ConvertToSpotRest(scanPath_target)

# Rotate the scan path so that the direction of motion is +x
scanPath_target_transformed = Scan.RotateTranslateLastRasterToX0(scanPath_target)

# Output the scan path to be simulated
Scan.ExportScan(scanPath_target_transformed, outFile="3DThesis/TestInputs/Path.txt")

# Get end position of scan
x, y = Scan.GetEndPosition()

# Change x,y to meters
x, y = x/1000.0, y/1000.0

# Update the domain to snapshot just around the end of the scan path
Sim.UpdateDomain(x=x,y=y,res=res,buf=buf)

# Run the simulation
Sim.Run()

# Extract the width and depth
width_target, depth_target = Sim.ExtractMPDims_Interp()

# Print targets
print("Targets")
print(width_target,depth_target)

'''
In this section:
- Want to iteratively optimize new scan pattern to match those dimensions
'''

# Directories for optimization
optDirs = ['ScanPaths/51/',
           'ScanPaths/52/',
           'ScanPaths/53/',
           'ScanPaths/54/']

for optDir in optDirs:

    # Load up desired scan path to change
    scanPath = Scan.LoadScan(inFile = optDir + 'scanpath.txt')

    # Change scan path to correct format
    scanPath = Scan.ConvertToSpotRest(scanPath)

    # Divide into smaller rasters
    scanPath_divided = Scan.SegmentScan(scanPath, segmentSize=segSize)

    # Container for optimized pathing
    scanPath_optimized = []

    # Previous optimal values
    prev_speed = 0
    prev_power = 0
    prev_cost = 0
    prev_i = -10

    # For each of these smaller segments, optimize
    for i in range(len(scanPath_divided)):
        # Print out what we are on right now
        print(i,len(scanPath_divided))
        # Get the values for the i'th line
        lastLine = scanPath_divided[i]
        lastVals = lastLine.strip().split("\t")
        # Only optimize if it is rastering and powered
        if (lastVals[0]=='0' and lastVals[4]=='1'):
            # Get the partial scan path
            scanPath_divided_partial = scanPath_divided[:(i+1)]

            # Rotate the scan path so that the direction of motion is +x
            scanPath_divided_partial_rotated = Scan.RotateTranslateLastRasterToX0(scanPath_divided_partial)

            # Output the scan path to be simulated
            Scan.ExportScan(scanPath_divided_partial_rotated, outFile="3DThesis/TestInputs/Path.txt")

            # Get end position of scan
            x, y = Scan.GetEndPosition()

            # Change x,y to meters
            x, y = x/1000.0, y/1000.0

            # Update the domain to snapshot just around the end of the scan path
            Sim.UpdateDomain(x=x,y=y,res=res,buf=buf)

            # While we search velocities
            iter = 0

            # Set up cost function
            minCost = 1e6

            # Containers for optimal power and velocity
            optimal_power = 1.0
            # optimal_speed = V
            optimal_speed = float(lastVals[5])
            optimal_cost = 1e6

            # If the previous one was optimized, used that as starting point
            if (prev_i == i-1):
                optimal_speed = prev_speed
                optimal_power = prev_power
                optimal_cost = prev_cost

                prevVals = scanPath_divided[i-1].strip().split("\t")
                x0, y0 = float(prevVals[1]), float(prevVals[2])
                x1, y1 = float(lastVals[1]), float(lastVals[2])
                dist = sqrt(((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)))

                # If the current one is "short" as well, just copy previous it and continue
                if (abs(dist-segSize)>segSize/10):
                    iter+=1e6

            while (iter < iterDepth):
                # Output iteration
                print("\t",iter)

                # Set up temporary variables for optimal power and speed
                minCost_speed, minCost_power = optimal_speed, optimal_power

                # New logic counters for improved search
                d = (iter != 0) * 1
                j = (iter != 0) * 1

                # Set up the new speeds and power for search
                while (True):
                    # Set up speed and power for this simulation
                    speed = optimal_speed #optimal_speed + (j * d) * speedRes / (2 ** iter)
                    power = optimal_power + (j * d) * powRes / (2 ** iter) # 1

                    # Change the scan path to the new power and speed
                    tempScan = Scan.ChangeLastRaster(scanPath_divided_partial_rotated, power=power, speed=speed)

                    # Output the scan path to be simulated
                    Scan.ExportScan(tempScan, outFile="3DThesis/TestInputs/Path.txt")

                    # Run the simulation
                    Sim.Run()

                    # Extract the width and depth
                    try:
                        width, depth = Sim.ExtractMPDims_Interp()
                    except:
                        buf*=2
                        Sim.UpdateDomain(x=x, y=y, res=res, buf=buf)
                        print("Increasing Domain Size")
                        continue

                    # Get the cost function
                    cost = Cost(width=width, depth=depth, width_target=width_target, depth_target=depth_target)

                    # Output more stuff
                    print("\t\t", (j * d), power, speed, width, depth, cost)

                    # If the cost is very small, just set power, speed and proceed
                    if (cost<1e-9):
                        minCost_speed = speed
                        minCost_power = power
                        minCost = optimal_cost
                        iter += 1e6
                        break

                    # Otherwise, set min cost and keep going
                    if (cost < minCost):
                        minCost = cost
                        minCost_speed = speed
                        minCost_power = power

                        # If 0, set to 1
                        if (j == 0): d = 1

                    # If the new one isn't better
                    else:
                        # If first step, and stepping positive, change direction
                        if (j == 1 and d == 1):
                            d = -1
                            j -= 1
                        # Otherwise, end this search
                        else:
                            break

                    j += 1

                # Increment iteration
                iter += 1

                # Set the new optimal power and speed
                optimal_power, optimal_speed = minCost_power, minCost_speed

                # Set optimal cost
                optimal_cost = minCost

            # Set previous values
            prev_speed = optimal_speed
            prev_power = optimal_power
            prev_cost = optimal_cost
            prev_i = i

            # Get the final adjusted velocity and change the scan path
            lastVals[4], lastVals[5] = str(optimal_power), str(optimal_speed)

            # Update lastLine
            lastLine = "\t".join(lastVals)+"\n"

        # Adjust divided scan path with new speed
        scanPath_divided[i] = lastLine

        # Output a checkpoint
        Scan.ExportScan(scanPath_optimized, outFile = "ckpt/Path_Optimized_ckpt_"+str(i).zfill(int(1+log10(len(scanPath_divided))))+".txt")

        # Append to the optimized scan path
        scanPath_optimized.append(lastLine)

    # Output the scan path to be simulated
    Scan.ExportScan(scanPath_optimized, outFile = optDir + 'scanpath_optimized.txt')