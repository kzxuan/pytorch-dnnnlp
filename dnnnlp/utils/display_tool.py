#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some tools for optimizing output
Last update: KzXuan, 2019.08.12
"""
import os
import sys
import tqdm
import time
import dnnnlp
import threading


def sysprint(_str):
    """
    Print without '\n
    * _str [str]: string to output
    """
    sys.stdout.write(_str)
    sys.stdout.flush()


def seconds_str(seconds):
    """
    Convert seconds to time string
    * seconds [int]: number of seconds
    - _time [str]: time format string
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return "{:02d}:{:02d}:{:02d}".format(int(h), int(m), int(s))
    else:
        return "{:02d}:{:02d}".format(int(m), int(s))


class timer(object):
    """Record the run time.
    """
    def __str__(self):
        return f"{self.__class__} records run time. Use timer.start()/timer.stop()."

    @classmethod
    def start(cls, desc=None, verbose=0):
        """Start timer.

        Args:
            desc [str]: a description string
            verbose [int]: verbose level
        """
        cls.verbose = verbose
        cls.begin = time.time()
        if desc and dnnnlp.verbose.check(cls.verbose):
            print(desc)

    @classmethod
    def stop(cls, desc=None):
        """Stop timer.

        Args:
            desc [str]: a description string
        """
        assert hasattr(cls, "begin"), "Need timer.start() first."

        cls.end = time.time()
        if dnnnlp.verbose.check(cls.verbose):
            if desc:
                sysprint(desc)
                sysprint(' ')
            print(seconds_str(cls.end - cls.begin))


class _wait(object):
    """Base class for wait animation.
    """
    def __call__(self, func, *args, **kwargs):
        """Decorator function.

        Args:
            func [func]: function
        """
        def wrapper(*args, **kwargs):
            self.start(self.desc, self.verbose)
            func(*args, **kwargs)
            self.stop()
        return wrapper

    def start(self, desc, verbose):
        pass

    def stop(self):
        pass


class slash(_wait):
    """Print '/-\|' when the program is running.
    """
    def __output(self):
        while self.flag:
            sysprint("\b/")
            time.sleep(0.1)
            sysprint("\b-")
            time.sleep(0.1)
            sysprint("\b\\")
            time.sleep(0.1)
            sysprint("\b|")
            time.sleep(0.1)

    @classmethod
    def start(cls, desc=None, verbose=1):
        """Start slash.

        Args:
            desc [str]: a description string
        """
        cls.verbose = verbose
        if not dnnnlp.verbose.check(cls.verbose):
            return

        cls.flag = 1
        if desc:
            sysprint(desc + "  ")
        cls.pr = threading.Thread(target=cls.__output, args=(cls, ))
        cls.pr.setDaemon(True)
        cls.pr.start()
        cls.begin = time.time()

    @classmethod
    def stop(cls):
        """Stop slash.

        Args:
            desc [str]: a description string
        """
        if not dnnnlp.verbose.check(cls.verbose):
            return

        assert hasattr(cls, "begin"), "Need slash.start() first."
        cls.flag = 0
        cls.pr.join()
        cls.end = time.time()
        print("\b" + u"\u2713", seconds_str(cls.end - cls.begin))


class dot(_wait):
    """Print '...' when the program is running.
    """
    def __output(self):
        while self.flag:
            if self.count == 6:
                sysprint('\b' * 6 + ' ' * 6 + '\b' * 6)
                self.count = 0
                time.sleep(0.5)
            else:
                sysprint(".")
                self.count += 1
                time.sleep(0.5)

    @classmethod
    def start(cls, desc=None, verbose=1):
        """Start dot.

        Args:
            desc [str]: a description string
            verbose [int]: verbose level
        """
        cls.verbose = verbose
        if not dnnnlp.verbose.check(cls.verbose):
            return

        cls.flag, cls.count = 1, 0
        if desc:
            sysprint(desc + " ")
        cls.pr = threading.Thread(target=cls.__output, args=(cls, ))
        cls.pr.setDaemon(True)
        cls.pr.start()
        cls.begin = time.time()

    @classmethod
    def stop(cls):
        """Stop dot.

        Args:
            desc [str]: a description string
        """
        if not dnnnlp.verbose.check(cls.verbose):
            return

        assert hasattr(cls, "begin"), "Need dot.start() first."
        cls.flag = 0
        cls.pr.join()
        cls.end = time.time()
        print("\b" * cls.count + u"\u2713", seconds_str(cls.end - cls.begin))


def bar(iterable, desc=None, leave=True, verbose=2):
    """Show progress bar and time in a loop.

    Args:
        iterable [int/list/dict/set/range]: a list for loop
        desc [str]: a description string
        leave [bool]: leave the bar or not after loop
        verbose [int]: verbose level
    """
    if isinstance(iterable, int):
        iterable = list(range(iterable))
    elif isinstance(iterable, dict):
        iterable = list(iterable.items())
    elif isinstance(iterable, set) or isinstance(iterable, range):
        iterable = list(iterable)
    elif isinstance(iterable, list):
        pass
    else:
        raise TypeError("Type error of 'iterable', wants int/list/dict/set/range, gets {}.".format(type(iterable)))

    if dnnnlp.verbose.check(verbose):
        for i in tqdm.tqdm(iterable, desc=desc, leave=leave, ascii=True):
            yield i
    else:
        for i in iterable:
            yield i


class table(object):
    """Print table style.
    """
    def __init__(self, col, place='^', sep='|', ndigits=4, verbose=2):
        """Initialize.

        Args:
            col [int/list]: number of columns or list of column names
            place [str/list]: one alignment mark '^'/'<'/'>' or list of each col mark
            sep [str]: separate mark like ' '/'|'/'*'
            ndigits [int]: decimal number for float
            verbose [int]: verbose level
        """
        self.verbose = verbose
        if not dnnnlp.verbose.check(self.verbose):
            return None
        self._col(col)
        self._place(place)
        self._sep(sep)
        self.ndigits = ndigits
        self.width = [0] * self.n_col
        self.cache = []
        self.falg = False
        print()
        self._header()

    def __str__(self):
        return f"{self.__class__} prints table row-by-row."

    def _col(self, col):
        """Initialize column number or name.

        Args:
            col [int/list]: number of columns or list of column names
        """
        if isinstance(col, list):
            self.col = col
            self.n_col = len(col)
        elif isinstance(col, int):
            self.col = None
            self.n_col = col
        else:
            raise TypeError("Type error of 'col', wants int/list, gets {}.".format(type(col)))

    def _place(self, place):
        """Initialize alignment mark.

        Args:
            place [str/list]: one alignment mark '^'/'<'/'>' or list of each col mark
        """
        if isinstance(place, str):
            self.place = [place for i in range(self.n_col)]
        elif isinstance(place, list):
            if len(place) != self.n_col:
                raise ValueError("Length error of 'place'.")
            self.place = place
        else:
            raise TypeError("Type error of 'place', wants str/list, gets {}.".format(type(place)))

    def _sep(self, sep):
        """Initialize separate mark.

        Args:
            sep [str]: separate mark like ' '/'|'/'*'
        """
        if sep != ' ':
            sep = ' ' + sep + ' '
        self.sep = [sep] * (self.n_col + 1)

    def _header(self):
        """Print header line if need.
        """
        if self.col:
            self.row(self.col)

    def _str_row(self, value):
        """Convert a row to string.

        Args:
            value [list]: list of column value
        """
        _str = []
        for ind, val in enumerate(value):
            if isinstance(val, float):
                val = "{:.{}f}".format(val, self.ndigits)
            elif not isinstance(val, str):
                val = str(val)
            if len(val) > self.width[ind]:
                self.width[ind] = len(val)
                self.flag = True

            _str.append(val)
        return _str

    def _print_row(self, value):
        """Print a row.

        Args:
            value [list]: list of column value
        """
        for ind, val in enumerate(value):
            sysprint(self.sep[ind])
            sysprint("{:{}{}}".format(val, self.place[ind], self.width[ind]))
        print(self.sep[-1])

    def _flash_cache(self):
        """Re-print all rows.
        """
        sysprint('\x1b[{}A'.format(len(self.cache)))
        for _str in self.cache:
            if _str != "---TABLE INSIDE LINE---":
                self._print_row(_str)
            else:
                values = ['-' * self.width[i] for i in range(self.n_col)]
                self._print_row(values)
        self.flag = False

    def line(self):
        """Print a line.
        """
        if not dnnnlp.verbose.check(self.verbose):
            return
        values = ['-' * self.width[i] for i in range(self.n_col)]
        self._print_row(values)
        self.cache.append("---TABLE INSIDE LINE---")

    def row(self, values):
        """Process and print a row.

        Atgs:
            values [list/dict]: list/dict of column value
        """
        if not dnnnlp.verbose.check(self.verbose):
            return
        if isinstance(values, list):
            assert len(values) == self.n_col, ValueError("Length error of the input row.")
            _str = self._str_row(values)
        elif isinstance(values, dict):
            assert self.col, TypeError("No column names for query.")
            list_values = []
            for cn in self.col:
                if cn in values.keys():
                    list_values.append(values[cn])
                else:
                    list_values.append('-')
            _str = self._str_row(list_values)
        else:
            raise TypeError("Type error of 'values', wants list/dict, gets {}.".format(type(values)))

        if self.flag:
            self._flash_cache()
        self._print_row(_str)
        self.cache.append(_str)
