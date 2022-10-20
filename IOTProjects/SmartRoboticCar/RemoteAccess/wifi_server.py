import socket
import sys
import picamera
import os
import time

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2

sys.path.insert(0, 'home/pi/picar-4wd')
import picar_4wd as fc

HOST = "192.168.0.14" # IP address of your Raspberry PI
PORT = 65432          # Port to listen on (non-privileged ports are > 1023)
power_val = 50
CAM_WIDTH = 640
CAM_HEIGHT = 480


def cameraimg():
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.rotation = 180
    rawCapture = PiRGBArray(camera)

    #grab an image from the camera
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array
    return image
    

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    
    while 1:
        client, clientInfo = s.accept()
        print("server recv from: ", clientInfo)
        data = client.recv(1024)      # receive 1024 Bytes of message in binary format
        if data != b"":
            print(data)
            if data == b"quit\r\n":
                break
            elif data == b"cam\r\n":
                image = cameraimg()
                cv2.imshow("camera", image)
                cv2.waitKey(5)
                client.sendall(image)
            elif data == b"temp\r\n":
                cpu_temp = os.popen("vcgencmd measure_temp").readline()
                tem = cpu_temp.replace("temp=", "")
                temp = tem.encode()
                client.sendall(temp)
            elif data == b"bat\r\n":
                battery = (str(fc.power_read())).encode()
                client.sendall(battery)
            elif data == b"69\r\n":
                client.sendall("power up 10".encode())
                if power_val < 100:
                    power_val = power_val + 10
            elif data == b"81\r\n":
                client.sendall("power down 10".encode())
                if power_val > 0:
                    power_val = power_val - 10
            elif data == b"87\r\n":
                client.sendall("forward".encode())
                fc.forward(power_val)
                time.sleep(0.05)
            elif data == b"83\r\n":
                client.sendall("backwards".encode())
                fc.backward(power_val)
                time.sleep(0.05)
            elif data == b"65\r\n":
                client.sendall("left".encode())
                fc.turn_left(power_val)
                time.sleep(0.05)
            elif data == b"68\r\n":
                client.sendall("right".encode())
                fc.turn_right(power_val)
                time.sleep(0.05)
            else:
                client.sendall(data);
                fc.stop()
                
 
    print("Closing socket")
    client.close()
    s.close()  
