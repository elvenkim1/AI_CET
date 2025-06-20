from gpiozero import LED
from time import sleep
gpio = LED(17)

while True:
    gpio.on()
    sleep(1)
    gpio.off()
    sleep(1)
