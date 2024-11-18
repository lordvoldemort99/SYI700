import paho.mqtt.client as mqtt
import time

# MQTT Broker settings
broker_address = "127.0.0.1"
broker_port = 1883
handshake_topic = "handshakeMV"  # Subscribed topic for handshake from PLC
response_topic = "ResponseToPLC"  # Published topic to send response to PLC
ready_topic = "ready"  # Published topic to send ready signal
x_coord_topic = "coord_X"  # Published topic for x coordinate
y_coord_topic = "coord_Y"  # Published topic for y coordinate
orientation_topic = "Orientation"  # Published topic for orientation
bottle_type_topic = "BottleType"  # Published topic for bottle type

#! Shiva start here 
# Initialize initial values
x_coordinate = 100  # Starting X coordinate
y_coordinate = 200  # Starting Y coordinate
orientation = 0  # Starting orientation angle
bottle_types = ["Red", "Blue", "Yellow"]  # Cycle through these bottle types
bottle_index = 0  # Start with the Red bottle type
#! Shive stop here 
handshake_status = ""

# Initialize MQTT client and set callbacks
client = mqtt.Client("MachineVisionSystem")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        # Subscribe to handshake topic from PLC
        client.subscribe(handshake_topic)
    else:
        print("Connection failed with code", rc)

def on_message(client, userdata, msg):
    global handshake_status
    print(f"Message received from PLC: {msg.topic} - {msg.payload.decode()}")
    
    # Update handshake status based on the message from PLC
    if msg.topic == handshake_topic:
        handshake_status = msg.payload.decode()

# Set up client and callbacks
client.on_connect = on_connect
client.on_message = on_message
client.connect(broker_address, broker_port)
client.loop_start()

# Step 1: Send "ready" signal to PLC
print("Sending ready signal to PLC...")
client.publish(ready_topic, "MV_ready")
time.sleep(1) 

# Step 2: Update coordinates and bottle type when requested
def update_coordinates(handshake_status, x_coordinate, y_coordinate, orientation, bottle_type):

    while True:
        # Wait for specific update command from PLC
        if handshake_status == "update_variable":
            # Send all values once when response is "update_variable"
            while handshake_status != "rec_handshake":
                print(f"Sending x coordinate: {x_coordinate}")
                client.publish(x_coord_topic, str(x_coordinate))
                print(f"Sending y coordinate: {y_coordinate}")
                client.publish(y_coord_topic, str(y_coordinate))
                print(f"Sending orientation: {orientation}")
                client.publish(orientation_topic, str(orientation))
                bottle_type = bottle_types[bottle_index]
                print(f"Sending bottle type: {bottle_type}")
                client.publish(bottle_type_topic, bottle_type)
                time.sleep(1)  # Delay before checking handshake again

            # Increment values after each complete set is sent and confirmed
            x_coordinate += 10
            y_coordinate += 10
            orientation += 30
            bottle_index = (bottle_index + 1) % len(bottle_types)

            print(f"Updated and confirmed values: X={x_coordinate}, Y={y_coordinate}, Orientation={orientation}, BottleType={bottle_type}")
            handshake_status = "" 
        else:
            time.sleep(1)

try:
    update_coordinates()
except KeyboardInterrupt:
    print("Disconnecting from broker...")
finally:
    client.loop_stop()
    client.disconnect()
