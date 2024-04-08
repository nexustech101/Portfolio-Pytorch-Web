import os
import time
import timeit
import logging
import requests
import argparse
import numpy as np
from typing import List, Dict, Union
from abc import ABC, abstractmethod

class PasswordManager(ABC):
    @abstractmethod
    def createManager(self):
        pass


def parse_req_data(url):
    with requests.get(url) as req:
        print(req.headers)

def passcodes() -> List[str]:
    """ Generates random passcodes, appends them to a list, 
        and returns the list of passcodes.
        
        Output:
            ['0000', '0001', '0002',..., '9999']
    """
    n = []  # Array that holds the uniquely generated passcode
    # Each loop generates a unique digit between 0 and 9
    for h in range(10):
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    # Append resulting combos
                    nums = f'{h}{i}{j}{k}'
                    n.append(nums)
                    print(nums)
    return n  # Return the list


def mutate_password(pass_phrase: str, min_idx: Union[int, None] = None, max_idx: Union[int, None] = None) -> List[str]:
    """ Mutates a password string within a specified range.

        Args:
            pass_phrase (str): The password string to be mutated.
            min_idx (int or None): The minimum index of the mutation range (inclusive).
            max_idx (int or None): The maximum index of the mutation range (inclusive).

        Returns:
            List[str]: A list of mutated password strings.
    """
    if not isinstance(pass_phrase, str): raise TypeError(f"{pass_phrase} is not a string")
    if not isinstance(min_idx, int): raise TypeError(f"{min_idx} is not an integer")
    if not isinstance(max_idx, int): raise TypeError(f"{max_idx} is not an integer")
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV0123456789!@#$%^&*()_-=+/|'
    pass_list = [char for char in pass_phrase]
    mutations = []
    for i in range(len(pass_list)):
        original_pass_char = pass_list[i]
        # modify the index value i if it's btween the minimum and maximum values
        if min_idx <= i <= max_idx:
            for char in chars:
                # time.sleep(0.1)       # delay for 200ms
                pass_list[i] = char   # assign char to the index of the string
                print(pass_list)      # print results in the console
                mutations.append(''.join(pass_list))
        # restart pass_phrase after every iteration
        pass_list[i] = original_pass_char
        
    return [i for i in mutations if i != pass_phrase]


def generate_ssn():
    x = [] # contains first sequence of SSN
    y = [] # contains second sequnce of SSN
    z = [] # contains third sequence of SSN
    w = [] # conatains all possible combos for SSN's
    
    for j in range(10):
        for k in range(10):
            y.append(f"{j}{k}")
            for l in range(10):
                x.append(f"{j}{k}{l}")
                for m in range(10):
                    z.append(f'{j}{k}{l}{m}')
                    
    for a in range(len(x)):
        for b in range(len(y)):
            for c in range(len(z)):
                if x[a] == '000' or x[a] == '666' or y[b] == '00' or z[c] == '0000': continue
                sequnce = f"{x[a]}-{y[b]}-{z[c]}"
                print(sequnce)
                w.append(sequnce)
                
    # return (w, len(w))


def copy_passfile(filename: str) -> None:
    if not str(filename) in os.listdir(): 
        raise FileNotFoundError(
            f'{filename} does not exist in the current working directory.'
        )
    # open the file and read it's contents
    with open(str(filename), 'r', encoding='utf-8') as original_file:
        # append results to avoid overwriting a file
        with open(f'copy_{filename}', 'a', encoding='utf-8') as new_file:
            words = original_file.readlines()
            new_file.writelines(words)


def write_passcodes_to_file(filename: str) -> None:
    try:
        with open(filename, 'a', encoding='utf-8') as file:
            for i in passcods():
                file.write(i + '\n')
                
        print(f'Passcode written to {filename} successfully.')
                
    except FileNotFoundError as e:
        raise FileNotFoundError(f'File not found: {e}')
    except Exception as e:
        raise Exception(f'Error reading file: {e}')


def get_passfile_contents(filename: str) -> List[str]:
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            passcodes = file.readlines()
        
        return [code.strip() for code in passcodes]
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f'File not found: {e}')
    except Exception as e:
        raise Exception(f'Error reading file: {e}')


def print_file_contents(filename: str) -> None:
    contents = ''
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            contents = file.read()
            print(contents)
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f'File not found: {e}')
    except Exception as e:
        raise Exception(f'Error reading file: {e}')


def serialize(array: List) -> Dict:
    return dict(zip(range(1, len(array) + 1), array))

            
def main():
    """ Main function to parse arguments and call file operations.

        This function sets up argument parsing for command-line usage,
        initializes logging, and calls associated functions with the parsed arguments.
        Any exceptions during the process are logged.
    """
    # Logging and Argument parsing setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description='Copy a text file')
    
    parser.add_argument('filename', help='Input filename as first argument and the script will copy the file')
    parser.add_argument('--stdout', action='store_true', help='Optionally log results into the console')
    parser.add_argument('--stdin', action='store_true', help='Optionally log results into the console')
    parser.add_argument('--file', action='', help='Optionally log results into the console')
    args = parser.parse_args()
    
    try:
        copy_passfile(args.filename)
        if args.stdout:
            print_file_contents(args.filename)
            
    except Exception as e:
        raise Exception(e)
    

def function(n):
    return n**2 / (n + 1)


if __name__ == '__main__':
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV0123456789!@#$%^&*()_-=+/|'
    pass_phrase = 'testing'
    
    # print(write_passcodes_to_file('passcodes.txt'))
    # print(get_passfile_contents('passcodes.txt'))
    # print(mutate_password(pass_phrase, 0, len(pass_phrase)))
    # print(serialize(['a', 'b', 'c', 'd']))
    # print(generate_ssn())
    # print(passcodes())
    # print(copy_passfile('passcodes.txt'))
    
    states = {
        'nj': [i for i in range(100, 120)],
        'pa': [i for i in range(121, 140)]
    }
    
    parse_req_data('https://kbpizza.com')

    # for i in range(1, len(states['pa']) + 1):
    #     if i >= 5:
    #         states['pa'].pop()
    # print(states['pa'])
    
    # main()
    