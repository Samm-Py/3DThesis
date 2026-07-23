# Load up desired scan path to change
scanPath = Scan.LoadScan(inFile = optDir + 'scanpath.txt')

# Change scan path to correct format
scanPath = Scan.ConvertToSpotRest(scanPath)

# Divide into smaller rasters
scanPath_divided = Scan.SegmentScan(scanPath, segmentSize=segSize)

# Container for optimized pathing
scanPath_optimized = []

# For each of these smaller segments, optimize
for i in range(len(scanPath_divided)):
    # Print out what we are on right now
    print(i,len(scanPath_divided))

    # Get the values for the i'th line
    lastLine = scanPath_divided[i]
    lastVals = lastLine.strip().split("\t")

    # Get the partial scan path
    scanPath_divided_partial = scanPath_divided[:(i+1)]

    # Rotate the scan path so that the direction of motion is +x
    scanPath_divided_partial_rotated = Scan.RotateTranslateLastRasterToX0(scanPath_divided_partial)

    # Output the scan path to be simulated
    Scan.ExportScan(scanPath_divided_partial_rotated, outFile="3DThesis/TestInputs/Path.txt")