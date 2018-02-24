#!/bin/bash

# A string of ssh commands that tunnel directly
# to anubis and port forward the port used by
# tensorboard so that the tensorboard server can
# be accessed locally

# To connect to the tensorboard server (it must
# be running first) type the following into your
# web browser:
#    localhost:6006

ssh -A -t -l vp3yp1 robots.ox.ac.uk \
-L 6006:localhost:6006 \
ssh -A -t -l vp3yp1 anubis.robots.ox.ac.uk -p 10007 \
-L 6006:localhost:6006
