import bluetooth
import sys
import os
import time

sys.path.insert(0, 'home/pi/picar-4wd')
import picar_4wd as fc

hostMACAddress = "E4:5F:01:2B:8E:26" # The address of Raspberry PI Bluetooth adapter on the server. The server might have multiple Bluetooth adapters.
port = 0
backlog = 1
size = 1024
s = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
s.bind((hostMACAddress, port))
s.listen(backlog)
print("listening on port ", port)
try:
    client, clientInfo = s.accept()
    while 1:   
        print("server recv from: ", clientInfo)
        data = client.recv(size)
        if data:
            print(data)
            if data == b"temp":
                cpu_temp = os.popen("vcgencmd measure_temp").readline()
                temp = cpu_temp.encode()
                client.send(temp)
            elif data == b"battery":
                percent = str(round((((fc.power_read() - 5.25) / 3.15) * 100), 2))
                print(percent)
                battery = ("battery = " + percent).encode()
                client.send(battery)
            else:
                client.send(data)
except: 
    print("Closing socket")
    client.close()
    s.close()
