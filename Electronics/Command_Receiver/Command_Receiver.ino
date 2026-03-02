String incomingByte ;    
int commandPin = PA0;



void setup() {

  Serial.begin(9600);

  pinMode(commandPin, OUTPUT);
  digitalWrite(commandPin, HIGH);

}
void loop() {

  if (Serial.available() > 0) {
  
  incomingByte = Serial.read();

    if (incomingByte == "1"){
      digitalWrite(commandPin, HIGH);

      Serial.write("Led on\n");
    }

    if (incomingByte == "0") {

      digitalWrite(commandPin, LOW);

      Serial.write("Led off\n");

    }




  }

}
