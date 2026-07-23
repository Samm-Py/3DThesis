d = 0
j = (iter!=0)*1

# Set up the new speeds and power for (quintary search)
while (True):
    # Set up speed and power for this simulation
    speed = optimal_speed + (j*d)*speedRes/(4**iter)
    power = 1

    # Change the scan path to the new power and speed
    tempScan = Scan.ChangeLastRaster(scanPath_divided_partial_rotated,power=power,speed=speed)

    # Output the scan path to be simulated
    Scan.ExportScan(tempScan, outFile="3DThesis/TestInputs/Path.txt")

    # Run the simulation
    Sim.Run()

    # Extract the width and depth
    width, depth = Sim.ExtractMPDims()

    # Get the cost function
    cost = Cost(width=width,depth=depth,width_target=width_target, depth_target=depth_target)

    # Output more stuff
    print("\t\t", j, power, speed, width, depth, cost)

    # If the cost is close enough (10%) to the optimal cost, just set power, speed and proceed
    if (abs(1-cost/optimal_cost)<0.1):
        minCost_speed = speed
        minCost_power = power
        iter += 1e6
        break

    # Otherwise, set min cost and keep going
    if (cost<minCost):
        minCost = cost
        minCost_speed = speed
        minCost_power = power
        minCost_j = j

        # If 0, set to 1
        if (j == 0): d = 1

    # If the new one isn't better
    else:
        # If first step, and stepping positive, change direction
        if (j == 1 and d==1):
            d = -1
            j -= 1
        # Otherwise, end this search
        else:
            break

    j += 1
    # If j=1 and