'''
Created on Mar 21, 2016

@author: Robert Chan
'''
from unittest import TestCase
from Driver import Driver


class TestSomeModule(TestCase):


    def testRun(self):
        sm = Driver()
        sm.run()