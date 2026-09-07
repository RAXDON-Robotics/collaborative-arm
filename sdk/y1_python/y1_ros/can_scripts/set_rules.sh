#!/bin/bash

sudo cp raxdon_y1_can.rules /etc/udev/rules.d/

sudo chmod +x /etc/udev/rules.d/raxdon_y1_can.rules

sudo udevadm control --reload-rules && sudo udevadm trigger