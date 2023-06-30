import pandas as pd
import socket
import sys

def handler():
    # Use a breakpoint in the code line below to debug your script.
    print(f'Server: {socket.gethostname()}')
    print(f'Executable: {sys.executable}')

    s2catalog_file = '/cpnet/projects/sikuttiaq/pond_inlet/Sentinel_2/CATALOG/index_Sentinel.csv'
    s2catalog = pd.read_csv(s2catalog_file)

    for column in s2catalog:
        print(column)
        print(s2catalog[column][0])
        print(s2catalog[column][1])
        print()

    # catalog = pd.read_csv('/cpnet/projects/sikuttiaq/pond_inlet/Sentinel_2/CATALOG/index_Sentinel.csv')
    # print(catalog)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    handler()