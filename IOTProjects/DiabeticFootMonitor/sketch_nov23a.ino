#include <ArduinoBLE.h>
#include <math.h>
#include <Adafruit_AHTX0.h>
#include <String>
//#include "HX711.h"

#define Switch D3
#define Therm A0
//#define LCDAT D5 //load cell
//#define LCCLK D4 //load cell
//#define calibration 2230 //load cell


//Thermistor BLE
BLEService THERMISTOR("7389b987-270d-52b0-ab6e-3c1dc968dc1a");
BLEStringCharacteristic TEMPERATURE("7389b988-270d-52b0-ab6e-3c1dc968dc1a", BLERead | BLENotify, 13);
//Humidity BLE
BLEService HUMIDITY("2fe4f1a2-8631-5441-8a33-c40e5f756197");
BLEStringCharacteristic RHUMIDITY("2fe4f1a3-8631-5441-8a33-c40e5f756197", BLERead | BLENotify, 13);
BLEStringCharacteristic MTEMPERATURE("2fe4f1a4-8631-5441-8a33-c40e5f756197", BLERead | BLENotify, 13);
//Impact sensor
BLEService SWITCH("0ab29215-7e21-5f14-b931-c1dee6c6e25f");
BLEBoolCharacteristic IMPACT("0ab29216-7e21-5f14-b931-c1dee6c6e25f", BLERead | BLENotify);
//Load Cell
//BLEService LOADCELL("")
//BLEStringCharacteristic WEIGHT("", BLERead | BLENotify)
//Pulse Ox
//BLEService PULSEOX("");
//BLEStringCharacteristic OXYGEN("", BLERead | BLENotify);
//BLEStringCharacteristic PULSE("", BLERead | BLENotify);


//Switch BLE

//HX711 scale; //load cell
Adafruit_AHTX0 aht; //humidity
bool impact = false; //leaf switch
int lastState = HIGH; //leaf
int currentState; //leaf

double Thermistor(int RawADC) { //formula for temperature based on "Temperature Measurement with a Thermistor and an Arduino" by Gerald Recktenwald*
  double fTemp;
  fTemp = log(10000.0 * ((1024.0 / RawADC - 1)));
  fTemp = (1 / (0.001129148 + (0.000234125 + (0.0000000876741 * fTemp * fTemp)) * fTemp)) - 273.15;
  return fTemp;
}

void press()  { //leaf switch
  currentState = digitalRead(Switch);
  if (lastState == LOW && currentState == HIGH)  {
    Serial.print("Impact");
    impact = true;
  }
  lastState = currentState;
}

void startBLE() { //Initialize ble if available
  if (!BLE.begin()) {
    Serial.println("BLE failed!");
    while(1);
  }
}

void setup() {
  Serial.begin(115200);
  //humidity
  if(! aht.begin()) {
    Serial.print("Humidity sensor not started");
  }
  
  //Switch
  pinMode(Switch, INPUT_PULLUP); //leaf switch
  attachInterrupt(digitalPinToInterrupt(Switch), press, CHANGE);

  //LoadCell
  //scale.begin(LCDAT, LCCLK);
  //scale.read_average();
  //scale.set_scale(calibration);
  //scale.tare();

  //BLE
  startBLE();
  //Thermistor
  BLE.setLocalName("Foot Monitor");
  BLE.setAdvertisedService(THERMISTOR);
  THERMISTOR.addCharacteristic(TEMPERATURE);
  BLE.addService(THERMISTOR);
  //Humidity
  BLE.setAdvertisedService(HUMIDITY);
  HUMIDITY.addCharacteristic(RHUMIDITY);
  HUMIDITY.addCharacteristic(MTEMPERATURE);
  BLE.addService(HUMIDITY);
  //Impact leaf switch
  BLE.setAdvertisedService(SWITCH);
  SWITCH.addCharacteristic(IMPACT);
  BLE.addService(SWITCH);
  //Load cell
  //BLE.setAdvertisedService(LOADCELL);
  //LOADCELL.addCharacteristic(SCALE);
  //BLE.addService(LOADCELL);
  //Pulse ox
  //BLE.setAdvertisedService(PULSEOX);
  //PULSEOX.addCharacteristic(OXYGEN);
  //PULSEOX.addCharacteristic(PULSE);
  //BLE.addService(PULSEOX);
  
  BLE.advertise();
  Serial.println("Bluetooth waiting for connections.");
  
}

void loop() {
  BLEDevice central = BLE.central();

  if(central) {
    Serial.print("Connected");
    Serial.println(central.address());
    digitalWrite(LED_BUILTIN, HIGH);


    while(central.connected())  {
      //Impact sensor
      if (impact == true) {
        IMPACT.writeValue(impact);
        impact = false;
      }
      
      //foot temp
      double val = analogRead(Therm);
      double fTemp = Thermistor(val);
      Serial.print("Foot Temperature = ");
      Serial.print(fTemp);
      Serial.println(" C");
      TEMPERATURE.writeValue(String(fTemp));
      delay(100);

      //Humitidy and sensor temp
      sensors_event_t humidity, temp;
      aht.getEvent(&humidity, &temp);
      Serial.print("Humidity module temp: "); Serial.print(temp.temperature); Serial.println(" degrees C");
      Serial.print("Humidity: "); Serial.print(humidity.relative_humidity); Serial.println("% relative humidity");
      RHUMIDITY.writeValue(String(humidity.relative_humidity));
      MTEMPERATURE.writeValue(String(temp.temperature));
      delay(100);

      //Load Cell
      //Serial.print(scale.get_units(), 1);
      //Serial.println(" lbs");
      //SCALE.writeValue(String(weight));

      //Pulse Ox

      
      //general
      delay(1000);
      }
    
    digitalWrite(LED_BUILTIN, LOW);
    Serial.print("Disconnected");
  }

}
