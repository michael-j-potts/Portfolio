from datetime import date, time, datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import sys
sys.path.append("/bluepy")
from bluepy import btle
import os
import os.path


class MyDelegate(btle.DefaultDelegate):
	def __init__(self):
		btle.DefaultDelegate.__init__(self)

	def handleNotification(self, cHandle, data):
		#decode our nano data
		parsed_data = data.decode("utf-8")
		#create a timestamp
		day = date.today()
		now = datetime.now()
		curTime = time(now.hour, now.minute, now.second)
		timestamp = datetime.combine(day, curTime)
		if cHandle == int(12):
			file.write("\n" + str(timestamp) + ",")
			print("Toe temperature:  ", parsed_data, "C")
			file.write(parsed_data + ",")
		if cHandle == int(16):
			print("Humidity:         ", parsed_data, "%")
			file.write(parsed_data + ",")
		if cHandle == int(19):
			print("Shoe temperature: ", parsed_data, "C")
			file.write(parsed_data + ",")
		if cHandle == int(23):
			print("Toe impact, check your foot!")
			impact = "impact"
			file.write(impact + ",")
        

#Our loop to receive all notification data that will continue until stopped
def record_data():
	counter = 20 #records 10 sets of data
	while counter > 0:
		if footMonitor.waitForNotifications(1.0):
			continue
		counter = counter - 1
		print()
	file.close()

def view_averages():
	file = open("Foot Monitor Data.csv") #open file and initialize variables
	toeTempAve = 0.0
	humAve = 0.0
	shoeTempAve = 0.0
	impactAve = 0
	lineCount = 0
	next(file)
	for line in file:## file[-40:] will only read the last 40 lines etc....
		lineCount = lineCount + 1
		separ = line.split(",")
		toeTempAve = toeTempAve + float(separ[1])
		humAve = humAve + float(separ[2])
		shoeTempAve = shoeTempAve + float(separ[3])
		if separ[4] == "impact":
			impactAve = impactAve + 1
	toeTempAve = round(toeTempAve / lineCount, 2)
	humAve = round(humAve / lineCount, 2)
	shoeTempAve = round(shoeTempAve / lineCount, 2)
	print("Your average toe temperature is ", toeTempAve, " degrees C")
	if toeTempAve < 26.0:
		print("Your toes are a bit cold! Try wiggling them to warm them up.")
	elif toeTempAve > 32.0:
		print("Your toes are warm!")
	print("The average humidity in your shoe is ", humAve, "%")
	print("Your average shoe temperature is ", shoeTempAve, " degrees C")
	if toeTempAve < 20.0:
		print("Your shoes are a bit cold! Try to find a warmer spot.")
	elif toeTempAve > 30.0:
		print("Your shoes are warm!")
	print("You stubbed your toes ", impactAve, " times.\n")

def graphs():
	file = open("Foot Monitor Data.csv") #open file and initialize variables
	next(file)
	timestamp = []
	toeTemp = []
	hum = []
	shoeTemp = []
	lineCount = 0
	for line in file:## file[-40:] will only read the last 40 lines etc...
		lineCount = lineCount + 1
		separ = line.split(",")
		timestamp.append(separ[0])
		toeTemp.append(float(separ[1]))
		hum.append(float(separ[2]))
		shoeTemp.append(float(separ[3]))
		
	#temp
	dates = pd.read_csv("Foot Monitor Data.csv", parse_dates = ['date'], index_col = ['date'],
						 names = ["date", "Toe_temp", "Humidity", "Shoe_temp", "Impact"], skiprows = [0])
	x = dates.index.values
	y = toeTemp
	fig, ax = plt.subplots()
	ax.plot(x, y)
	ax.set(xlabel='time', ylabel='Temperature in C', title='Toe Temperature',
			ylim=(10, 40))
	plt.setp(ax.get_xticklabels(), rotation = 45)
	ax.grid()
	plt.show()

	#humidity
	dates = pd.read_csv("Foot Monitor Data.csv", parse_dates = ['date'], index_col = ['date'],
						 names = ["date", "Toe_temp", "Humidity", "Shoe_temp", "Impact"], skiprows = [0])
	x2 = dates.index.values
	y2 = hum
	fig, ax = plt.subplots()
	ax.plot(x2, y2)
	ax.set(xlabel='time', ylabel='Humidity %', title='Humidity',
			ylim = (40, 80))
	ax.grid()
	plt.show()
	
	#module temperature
	dates = pd.read_csv("Foot Monitor Data.csv", parse_dates = ['date'], index_col = ['date'],
						 names = ["date", "Toe_temp", "Humidity", "Shoe_temp", "Impact"], skiprows = [0])
	x3 = dates.index.values
	y3 = shoeTemp
	fig, ax = plt.subplots()
	ax.plot(x3, y3)
	ax.set(xlabel='time', ylabel='Temperature in C', title='Shoe Temperature',
			ylim=(10, 40))
	plt.setp(ax.get_xticklabels(), rotation = 45)
	ax.grid()
	plt.show()


def print_menu():
	print(31 * "-", "Menu", 31* "-")
	print("1. Record data")
	print("2. View averages")
	print("3. View graph trends")
	print("4. Exit")
	print(68 * "-", "\n")

#Initialize a linked list
Device = "D9:89:A6:02:E4:62"
file = open("Foot Monitor Data.csv", "a")
if os.stat("Foot Monitor Data.csv").st_size == 0:
	file.write("date, Toe_temp, Humidity, Shoe_temp, Impact")

#Set the Device and call our delegate program
footMonitor = btle.Peripheral(Device)
footMonitor.setDelegate(MyDelegate())

#Set our services and characteristic UUID's, and retrieve the first item from the characteristics
Thermistor = footMonitor.getServiceByUUID("7389b987-270d-52b0-ab6e-3c1dc968dc1a")
Foot_Temp = Thermistor.getCharacteristics('7389b988-270d-52b0-ab6e-3c1dc968dc1a')[0]
Humidity_module = footMonitor.getServiceByUUID("2fe4f1a2-8631-5441-8a33-c40e5f756197")
Humidity = Humidity_module.getCharacteristics("2fe4f1a3-8631-5441-8a33-c40e5f756197")[0]
MTemperature = Humidity_module.getCharacteristics("2fe4f1a4-8631-5441-8a33-c40e5f756197")[0]
Switch = footMonitor.getServiceByUUID("0ab29215-7e21-5f14-b931-c1dee6c6e25f")
Switch_button = Switch.getCharacteristics("0ab29216-7e21-5f14-b931-c1dee6c6e25f")[0]

#Write our characteristics
setup_data = b"\x01\00"
footMonitor.writeCharacteristic(Foot_Temp.valHandle+1, setup_data)
footMonitor.writeCharacteristic(Humidity.valHandle+1, setup_data)
footMonitor.writeCharacteristic(MTemperature.valHandle+1, setup_data)
footMonitor.writeCharacteristic(Switch_button.valHandle+1, setup_data)

loop = True

while loop:
	print_menu()
	choice = input("Which item would you like to select?")
	if choice.isdigit() != True: #Catch statement. Ints only
		print("Sorry, invalid entry.")

	else:
		choice = int(choice)
		if choice == 1:
			print("Recording new data entries")
			record_data()
			pass
		elif choice == 2:
			print("\nViewing data averages:")
			view_averages()
			pass
		elif choice == 3:
			graphs()
			pass
		elif choice == 4: #exit menu
			loop = False
		else: #second catch, only a listed int is allowed
			print("Sorry, you entered an invalid option, please try again.")

