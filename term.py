from __future__ import print_function
import colorama
from colorama import Fore, Back, Style
import contextlib
import sys
from tqdm import tqdm
colorama.init()

'''
colorama supports:
    Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
    Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
    Style: DIM, NORMAL, BRIGHT, RESET_ALL
'''
### white 
def white(s: str) -> str:
    return "{}{}{}".format(Fore.WHITE, s, Style.RESET_ALL)

def light_white(s: str) -> str:
    return "{}{}{}".format(Fore.LIGHTWHITE_EX, s, Style.RESET_ALL)

def white_bg(s: str) -> str:
    return "{}{}{}".format(Back.WHITE, s, Style.RESET_ALL)

def light_white_bg(s: str) -> str:
    return "{}{}{}".format(Back.LIGHTWHITE_EX, s, Style.RESET_ALL)

### cyan 
def cyan(s: str) -> str:
    return "{}{}{}".format(Fore.CYAN, s, Style.RESET_ALL)

def light_cyan(s: str) -> str:
    return "{}{}{}".format(Fore.LIGHTCYAN_EX, s, Style.RESET_ALL)

def cyan_bg(s: str) -> str:
    return "{}{}{}".format(Back.CYAN, s, Style.RESET_ALL)

def light_cyan_bg(s: str) -> str:
    return "{}{}{}".format(Back.LIGHTCYAN_EX, s, Style.RESET_ALL)

### purple 
def purple(s: str) -> str:
    return "{}{}{}".format(Fore.MAGENTA, s, Style.RESET_ALL)

def light_purple(s: str) -> str:
    return "{}{}{}".format(Fore.LIGHTMAGENTA_EX, s, Style.RESET_ALL)

def purple_bg(s: str) -> str:
    return "{}{}{}".format(Back.MAGENTA, s, Style.RESET_ALL)

def light_purple_bg(s: str) -> str:
    return "{}{}{}".format(Back.LIGHTMAGENTA_EX, s, Style.RESET_ALL)

### blue
def blue(s: str) -> str:
    return "{}{}{}".format(Fore.BLUE, s, Style.RESET_ALL)

def light_blue(s: str) -> str:
    return "{}{}{}".format(Fore.LIGHTBLUE_EX, s, Style.RESET_ALL)

def blue_bg(s: str) -> str:
    return "{}{}{}".format(Back.BLUE, s, Style.RESET_ALL)

def light_blue_bg(s: str) -> str:
    return "{}{}{}".format(Back.LIGHTBLUE_EX, s, Style.RESET_ALL)

### yellow
def yellow(s: str) -> str:
    return "{}{}{}".format(Fore.YELLOW, s, Style.RESET_ALL)

def light_yellow(s: str) -> str:
    return "{}{}{}".format(Fore.LIGHTYELLOW_EX, s, Style.RESET_ALL)

def yellow_bg(s: str) -> str:
    return "{}{}{}".format(Back.YELLOW, s, Style.RESET_ALL)

def light_yellow_bg(s: str) -> str:
    return "{}{}{}".format(Back.LIGHTYELLOW_EX, s, Style.RESET_ALL)

### red
def red(s: str) -> str:
    return "{}{}{}".format(Fore.RED, s, Style.RESET_ALL)

def light_red(s: str) -> str:
    return "{}{}{}".format(Fore.LIGHTRED_EX, s, Style.RESET_ALL)

def red_bg(s: str) -> str:
    return "{}{}{}".format(Back.RED, s, Style.RESET_ALL)

def light_red_bg(s: str) -> str:
    return "{}{}{}".format(Back.LIGHTRED_EX, s, Style.RESET_ALL)

### black
def black(s: str) -> str:
    return "{}{}{}".format(Fore.BLACK, s, Style.RESET_ALL)

def light_black(s: str) -> str:
    return "{}{}{}".format(Fore.LIGHTBLACK_EX, s, Style.RESET_ALL)

def black_bg(s: str) -> str:
    return "{}{}{}".format(Back.BLACK, s, Style.RESET_ALL)

def light_black_bg(s: str) -> str:
    return "{}{}{}".format(Back.LIGHTBLACK_EX, s, Style.RESET_ALL)



### green
def green(s: str) -> str:
    return "{}{}{}".format(Fore.GREEN, s, Style.RESET_ALL)

def light_green(s: str) -> str:
    return "{}{}{}".format(Fore.LIGHTGREEN_EX, s, Style.RESET_ALL)

def green_bg(s: str) -> str:
    return "{}{}{}".format(Back.GREEN, s, Style.RESET_ALL)

def light_green_bg(s: str) -> str:
    return "{}{}{}".format(Back.LIGHTGREEN_EX, s, Style.RESET_ALL)

class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        # sys.stdout = sys.stderr = DummyTqdmFile(orig_out_err[0])
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err



