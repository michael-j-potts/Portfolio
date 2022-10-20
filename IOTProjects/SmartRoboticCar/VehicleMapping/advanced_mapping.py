import picar_4wd as fc
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

VAL = 10
speed = 25
us_val = VAL
cur_ang = 0
scan_l = []
scan_a = np.array([])
temparray = np.array([])
statusarr = ([])
xyarray = ([])
max_ang = 90
min_ang = -90
count = 0

#For a grid with higher accuracy, decreasing VAL will increase the number of readings
#taken, and will remove some of the mathematical average estimation. For increased
#speed of scan, increase VAL. This will decrease scan accuracy. As well, the current
#reference distance (within main()) is set to 50cm. If you would like to observe farther distances
#edit it appropriately.

def grid(ref, array):
    #This function is designed to take in the reference distance, and array of
    #angles and distances, convert them to x and y coordinates, and plot
    #them on a visible graph.
    global xyarray, statusarr, count
    #Calculates our x and y coordinates. x is times by negative one to
    #correct for switched sides.
    x = ((array[:,1] * (np.sin(np.deg2rad(array[:,0]))))* -1)
    y = ((array[:,1] * (np.cos(np.deg2rad(array[:,0])))))
    #Round the x y coordinates then convert to ints. Load into
    #a pandas dataframe for better graph object manipulation.
    x = np.around(x).astype(int)
    y = np.around(y).astype(int)
    z = np.array(statusarr)
    #used to correct for how the scan switches perspective when it pans
    #from right to left instead of left to right
    if count == 1:
        z = z[::-1]
    df = pd.DataFrame({"x":x, "y":y, "z":z})
    
    #print statements to ensure accuracy for testing point data
    #print(array)
    #print(df)
    
    #plot our array with respect to a status of 2 (no object)
    #or 1, object detected, and set the values to not a number
    #to only display the proper values with discontinuity, avoiding
    #connecting the line graphs together.
    dfobj = df.copy()
    dfobj[dfobj.z == 2] = np.nan
    dfclear = df.copy()
    dfclear[dfclear.z == 1] = np.nan
    plt.plot(dfobj.x, dfobj.y, color = 'red')
    plt.plot(dfclear.x, dfclear.y, 'o', color = 'green')
    plt.axis([-100, 100, 0, 100])
    plt.show()
    if count == 0:
        count += 1
    else:
        count = 0
    

def scan_step_rev(ref):
    #Roughly the same as scan_step(), but I wanted to receive
    #distance and angle values, so I rewrote scan_step() and
    #changed the variable names to avoid introducing errors.
    global scan_l, cur_ang, us_val, scan_a, temparray
    cur_ang += us_val
    if cur_ang >= max_ang:
        cur_ang = max_ang
        us_val = -VAL
    elif cur_ang <= min_ang:
        cur_ang = min_ang
        us_val = VAL
    stat = int(fc.get_status_at(cur_ang, ref1=ref))
    distance = int(fc.get_distance_at(cur_ang))
    scan_a = np.append(scan_a, np.array([cur_ang, distance]))
    #scan_a is a global list that gathers the angle, and distance
    #and then sets it inside temparray before being wiped clean.
    
    scan_l.append(stat)
    if cur_ang == min_ang or cur_ang == max_ang:
        if us_val < 0:
            scan_l.reverse()
        temp = scan_l.copy()
        temparray = scan_a
        scan_a = ([])
        scan_l = []
        return temp
    else:
        return False


def main():
    global statusarr, dist, temparray
    while True:
        ref = 50
        status = scan_step_rev(ref)
        if not status:
            continue
        
        if temparray.size >= (180/VAL + 2):
            #This is to isolate only the rows of data I want
            #which will be fed into grid(), since the first set
            #only every delivers half the items of data due to its half pan.
            #temparray is only the vehicle to help it escape scan_step_rev
            scan_data = temparray.reshape(int(180/VAL),2)
            if status != False:
                statusarr = np.array(status)
            temparray = ([])
            grid(ref, scan_data)
        #print(status)
        
        

if __name__ == "__main__":
    try:
        main()
    finally:
        fc.stop()
