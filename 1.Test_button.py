from gpiozero import Button
from time import sleep
button = Button(21)

while True:
    if button.is_pressed:
        print("Button is pressed")
        sleep(1)
        
    else:
        print("Button is not pressed")
        sleep(1)
