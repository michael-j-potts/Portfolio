import picar_4wd as fc
import time
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
blankgrid = np.zeros([500,500])
max_ang = 90
min_ang = -90
count = 0
location = np.array([0,0])
destination = np.array([50,50])
facing = "N"


#Since we want to see as far as possible, the ref distance has been set to 5m.
#This will allow us to map everything the car sees and work to institute a
#navigation system based on clear points to travel through.

def gridmap(ref, array):
    global blankgrid, location
    objarr = grid(ref, array)

def grid(ref, array):
    #This function is designed to take in the reference distance, and array of
    #angles and distances, convert them to x and y coordinates, and plot
    #them on a visible graph.
    global xyarray, statusarr, count, gridmap
    #Calculates our x and y coordinates. x is times by negative one to
    #correct for switched sides.
    x = ((array[:,1] * (np.sin(np.deg2rad(array[:,0]))))* -1)
    y = ((array[:,1] * (np.cos(np.deg2rad(array[:,0])))))
    #movess the car to the center of the grid
    #Round the x y coordinates then convert to ints. Load into
    #a pandas dataframe for better graph object manipulation.
    x = np.around(x).astype(int)
    y = np.around(y).astype(int)
    z = np.array(statusarr)
    x = np.reshape(x, (int(180/VAL), 1))
    y = np.reshape(y, (int(180/VAL), 1))
    z = np.reshape(z, (int(180/VAL), 1))
    xyarray = np.concatenate((x, y, z), axis = 1)
    #used to correct for how the scan switches perspective when it pans
    #from right to left instead of left to right
    if count == 1:
        z = z[::-1]
    
    #print statements to ensure accuracy for testing point data
    #print(array)
    #print(df)
    
    #Instead of plotting, we isolate the obstacle x,y values and
    #deliver them to our grid for mapping.
    obj = xyarray[xyarray[:,2] == 1]
    print(obj)
    
    if count == 0:
        count += 1
    else:
        count = 0
    return obj
    

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

def move(): # moves roughly 25cm forward, and returns the xcm that were moved
    speed4 = fc.Speed(25)
    speed4.start()
    fc.forward(30)
    x = 0
    for i in range(8):
        time.sleep(0.0705)
        speed = speed4()
        x += speed * 0.1
        #print("%smm/s"%speed)
    print("%scm"%x)
    speed4.deinit()
    fc.stop()
    return x
    
def left(): #turns roughly 90 degrees to the left
    fc.turn_left(100)
    time.sleep(0.53)
    fc.stop()

def right(): #turns roughly 90 degrees to the right
    fc.turn_right(100)
    time.sleep(0.48)
    fc.stop()

def flip(): #rotates 180 degrees
    right()
    right()

def GoNorth():
    global facing
    print("North")
    if facing != "N":
        if facing == "E":
            left()
        elif facing == "W":
            right()
        elif facing == "S":
            left()
            left()
    facing = "N"

def MoveNorth():
    print("Move north")
    y = move()
    location[1] = int(location[1]) + y
    
def GoSouth():
    global facing
    print("South")
    if facing != "S":
        if facing == "E":
            right()
        if facing == "W":
            left()
        if facing == "N":
            right()
            right()
    facing = "S"    

def MoveSouth():
    print("Move south")
    y = move()
    location[1] = int(location[1]) - y
    
def GoEast():
    global facing
    print("East")
    if facing != "E":
        if facing == "N":
            right()
        if facing == "S":
            left()
        if facing == "W":
            right()
            right()
    facing = "E"
    
def MoveEast():
    print("Move east")
    x = move()
    location[0] = int(location[0]) + x

def GoWest():
    global facing
    print("west")
    if facing != "W":
        if facing == "N":
            left()
        if facing == "S":
            right()
        if facing == "E":
            left()
            left()
    facing = "W"
    
def MoveWest():
    print("Move west")
    x = move()
    location[0] = int(location[0]) - x
  


def main():
    global statusarr, dist, temparray, location, distance, destination, facing
    while True:
        ref = 30
        status = scan_step_rev(ref) #scan
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
            gridmap(ref, scan_data) 
            
            temp = status[7:11] #status ahead of the car
            carl = status[0:4] #status left of the car
            carr = status[14:18] #status right of the car
            clear = [2,2,2,2] #clear
            
            print(location)
            print(status)
            
            if (int(destination[1]) - 10) > location[1]: #if destination is north
                if facing != "N": #and not facing north
                    GoNorth() # face north
                    status = scan_step_rev(ref) #rescan
                    if not status:
                        continue
                    temp = status[7:11]
                if temp != clear: #if north not clear
                    if (int(destination[0]) - 10) > location[0]: #and destination is east
                        if carr == clear: #if clear, face and go east
                            GoEast()
                            MoveEast()
                        else: #else go west
                            GoWest()
                            MoveWest()
                    else: #destination is west, the go west or east
                        if carl == clear:
                            GoWest()
                            MoveWest()
                        else:
                            GoEast()
                            MoveEast()
                else: #move north if the north is clear
                    MoveNorth()
            
            elif (int(destination[1]) + 10) < location[1]: #South 
                if facing != "S":
                    GoSouth()
                    status = scan_step_rev(ref)
                    if not status:
                        continue
                    temp = status[7:11]
                if temp != clear:
                    if (int(destination[0]) - 10) > location[0]:
                        if carl == clear:
                            GoEast()
                            MoveEast()
                        else:
                            GoWest()
                            MoveWest()
                    else:
                        if carr == clear:
                            GoWest()
                            MoveWest()
                        else:
                            GoEast()
                            MoveEast()
                else:
                    MoveSouth()
                    
            elif (int(destination[0]) - 10) > location[0]: #East
                if facing != "E":
                    GoEast()
                    status = scan_step_rev(ref)
                    if not status:
                        continue
                    temp = status[7:11]
                if temp != clear:
                    if (int(destination[1]) - 10) > location[1]:
                        if carl == clear:
                            GoNorth()
                            MoveNorth()
                        else:
                            GoSouth()
                            MoveSouth()
                    else:
                        if carr == clear:
                            GoSouth()
                            MoveSouth()
                        else:
                            GoNorth()
                            MoveNorth()
                else:
                    MoveEast()
            
            elif (int(destination[0]) + 10) < location[0]: #West
                if facing != "W":
                    GoWest()
                    status = scan_step_rev(ref)
                    if not status:
                        continue
                    temp = status[7:11]
                if temp != clear:
                    if (int(destination[1]) - 10) > location[1]:
                        if carl == clear:
                            GoSouth()
                            MoveSouth()
                        else:
                            GoNorth()
                            MoveNorth()
                    else:
                        if carl == clear:
                            GoNorth()
                            MoveNorth()
                        else:
                            GoSouth()
                            MoveSouth()
                else:
                    MoveWest()                        

            else: #once we have arrived at our destination
                fc.stop
                print("Arrived at destination")
                print("New destination?")
                x = int(input("x: "))
                y = int(input("y: "))
                destination = np.array([x, y])
        

if __name__ == "__main__":
    try:
        main()
    finally:
        fc.stop()
