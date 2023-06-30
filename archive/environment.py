# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import socket
import os
import sys
import pandas as pd

def handler():
    # Use a breakpoint in the code line below to debug your script.
    print(f'Server: {socket.gethostname()}')
    print(f'Executable: {sys.executable}')

    directory = '/cpnet/projects/sikuttiaq/pond_inlet/'
    files = os.listdir(directory)
    for file in files:
        print(file)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    handler()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
