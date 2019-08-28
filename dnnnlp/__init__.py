#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep neural model
Ubuntu 16.04 & PyTorch 1.2
Last update: KzXuan, 2019.08.28
"""
__name__ = "dnnnlp"
__version__ = "1.1.0"


# verbose for output portion
class verbose():
    @classmethod
    def config(cls, level):
        """Set verbose level for the whole package.

        Args:
            level [int]: verbose level, needs 0/1/2, higher level means more output
        """
        cls.level = level

    @staticmethod
    def check(level):
        """Check a set level of one output.

        Args:
            level [int]: verbose level of a function, needs 0/1/2

        Returns:
            True or False
        """
        if verbose.level >= level:
            return True
        else:
            return False


# set default verbose level
verbose.config(2)
