#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Print the progress of the program
Last update: KzXuan, 2018.12.11
"""
import re
import sys
import time
import threading
import numpy as np


def sysprint(str_words):
    sys.stdout.write(str_words)
    sys.stdout.flush()


class slash(object):
    """
    Print /-\| when the program is running
    """
    def __init__(self):
        self.flag = 1

    def __output(self):
        while(1):
            if self.flag == 0:
                break
            sysprint("\b/")
            time.sleep(0.1)
            sysprint("\b-")
            time.sleep(0.1)
            sysprint("\b\\")
            time.sleep(0.1)
            sysprint("\b|")
            time.sleep(0.1)

    def start(self, print_str):
        self.flag = 1
        sysprint(print_str + "  ")
        self.pr = threading.Thread(target=self.__output)
        self.pr.start()
        self.begin = time.time()

    def stop(self):
        self.flag = 0
        self.pr.join()
        self.end = time.time()
        sysprint("\b" + u"\u2713" + "  %4.1fs" % (self.end - self.begin) + "\n")


class percent(object):
    """
    Print percentage when the program is running
    """
    def __init__(self, print_str, total, show_time=True):
        self.print_str = print_str
        self.total = total
        self.count = 0
        self.show_time = show_time
        if self.show_time:
            self.begin = time.time()
        sysprint(print_str + " ")

    def change(self, num=1):
        self.count += num
        if self.count > self.total:
            raise OverflowError("Too many loop for percent function.")

        sysprint('\r' + self.print_str + " ")
        sysprint("%6.2f%%" % (float(self.count) / self.total * 100))
        if self.show_time:
            now_time = time.time() - self.begin
            left_time = (now_time / self.count) * (self.total - self.count)

        if self.count == self.total:
            sysprint("  %4.1fs   " % now_time)
            print()
        else:
            sysprint("  %4.1fs   " % left_time)

    def change_to(self, to):
        self.count = to - 1
        self.change()


class run_time(object):
    """
    Record the run time of the program
    """
    def __init__(self, print_str=None):
        self.begin = time.time()
        if print_str is not None:
            print(print_str)

    def stop(self, time_str="* Run time:", end_str="* Done !"):
        self.end = time.time()
        print((time_str + ' %.2f' + 's') % (self.end - self.begin))
        if end_str is not None:
            print(end_str)


class table_print(object):
    """
    Print table style
    """
    def __init__(self, col, width, place='^', sep="space", double_sep=-1,
                 header=True, header_line=False, digits=4):
        self.col = col
        self.width = width
        self._place(place)
        self._check()

        self.sep = self._style(sep)
        self.seps = [self.sep for i in range(len(col) + 1)]
        if double_sep != -1:
            self._double(double_sep)
        self.header_line = header_line
        if header:
            self._header()
        self.digits = digits

    def _place(self, place):
        if type(place).__name__ == 'str':
            self.place = [place for i in range(len(self.col))]
        elif type(place).__name__ == 'list':
            self.place = place
        else:
            raise ValueError("Woring type for parameter 'place', only allow 'str' or 'int'")

    def _check(self):
        if not len(self.col) == len(self.width) == len(self.place):
            raise Exception("Wrong length for some parameters.")
        for ind, wid in enumerate(self.width):
            if wid < len(self.col[ind]):
                raise ValueError("Wrong length for {}.".format(self.col[ind]))

    def _style(self, sep):
        styles = {
            "space": ' ',
            "vertical": ' | ',
            "star": ' * '
        }
        return styles[sep]

    def _double(self, double_sep):
        mid = len(self.sep) // 2
        char = self.sep[mid]
        d_sep = self.sep[:mid] + char + self.sep[mid:]
        if type(double_sep).__name__ == "list":
            for loc in double_sep:
                self.seps[loc] = d_sep
        elif type(double_sep).__name__ == "int":
            self.seps[double_sep] = d_sep
        else:
            raise ValueError("Woring type for parameter 'double_sep', only allow 'int' or 'list'.")

    def _header(self):
        for ind, name in enumerate(self.col):
            sysprint(self.seps[ind])
            sysprint(self._fixed_width(name, self.width[ind], self.place[ind]))
        print(self.seps[-1])
        if self.header_line:
            for ind, name in enumerate(self.col):
                sysprint(self.seps[ind])
                sysprint('-' * self.width[ind])
            print(self.seps[-1])

    def _fixed_width(self, value, width, place, padding=False, digits=4):
        if type(value).__name__[:5] == "float":
            if width - digits < 2:
                raise ValueError("Woring format for parameters 'width' & 'digits'.")
            form = "{:0{}{}.{}f}" if padding else "{:{}{}.{}f}"
            get = form.format(value, place, width, digits)
            while len(get) > width:
                digits -= 1
                get = form.format(value, place, width, digits)

        elif type(value).__name__[:3] == 'int':
            form = "{:0{}{}d}" if padding else "{:{}{}d}"
            get = str(value)
            min_width = 7 if value >= 0 else 8
            if len(get) > width and width < min_width:
                raise ValueError("No more place for show a int number.")
            if len(get) > width:
                get = "{:{}.{}e}".format(value, place, width - min_width + 1)
            elif len(get) < width and place != '<':
                get = ' ' + form.format(value, place, width - 1)
            else:
                get = form.format(value, place, width)

        elif type(value).__name__ == 'str':
            if len(value) > width and width > 3:
                value = value[:width - 3] + '...'
            elif len(value) > width and width <= 3:
                raise ValueError("No more place for show a string.")
            if len(value) < width:
                get = ' ' + "{:{}{}s}".format(value, place, width - 1)
            else:
                get = "{:{}{}s}".format(value, place, width)

        else:
            try:
                value = str(value)
            except:
                raise TypeError("Wrong type for output.")
            get = self._fixed_width(value, width, place, padding=padding, digits=digits)

        return get

    def print_row(self, values, padding=False):
        if type(values).__name__ == 'list':
            if len(values) != len(self.col):
                raise ValueError("Wrong length of the input row.")
            for ind, wid in enumerate(self.width):
                sysprint(self.seps[ind])
                sysprint(self._fixed_width(values[ind], wid, self.place[ind],
                                        padding=padding, digits=self.digits))
            print(self.seps[-1])
        elif type(values).__name__ == 'dict':
            for ind, wid in enumerate(self.width):
                sysprint(self.seps[ind])
                value = values[self.col[ind]] if self.col[ind] in values else '-'
                sysprint(self._fixed_width(value, wid, self.place[ind],
                                           padding=padding, digits=self.digits))
            print(self.seps[-1])
