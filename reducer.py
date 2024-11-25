#!/usr/bin/env python3
import sys

def reducer():
    # Read data from mapper's output line by line
    for line in sys.stdin:
        print(line)

if __name__ == "__main__":
    reducer()