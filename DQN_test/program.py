#!/usr/bin/env python
from gtpinterface import gtpinterface
import sys

def main():
	"""
	Main function, simply sends user input on to the gtp interface and prints
	responses.
	"""
	interface = gtpinterface()
	print("Loading finished")
	while True:
		command = input()
		if (command == ""):
			print("Enter valid command")
			continue
		success, response = interface.send_command(command)
		print ("= ") if success else print("? ")
		print(response, "\n")
		sys.stdout.flush()

if __name__ == "__main__":
	main()
