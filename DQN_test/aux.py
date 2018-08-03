from gamestate import *
from DQL import *
from gtpinterface import gtpinterface
import sys

def random_play(gamestate):
    while True:
        try:
            a = random.randrange(gamestate.size)
            b = random.randrange(gamestate.size)
            gamestate.play((a, b))
            break
        except ValueError:
            print ("Oops!  That was no valid number.  Try again")

