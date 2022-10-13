#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This code runs no-adaptation condition.

Author: Busra Tugce Gurbuz

Dependencies:
1.PsychoPy 3.0.1.
2. numpy, os, sys, pandas
3. set_parameters.py, welcome_screen.py, no_adaptation.py, end_screen.py
'''


from __future__ import absolute_import, division
from psychopy import visual, event, gui, data, logging, clock, core
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
								STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np
import os
import sys
import pandas as pd

import set_parameters
import welcome_screen
import no_adaptation
import end_screen

################## EXP INFO #######################################
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Get expInfo from UI and store
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel

expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp

################# SETUP #########################################
##Window
win = visual.Window(
    size=windowSize, fullscr=True, screen=0,
    allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, units = 'deg')
win.mouseVisible=False
##Timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine

###################  EXPERIMENTAL ROUTINE #############################
# Welcome Screen to experiment
welcome_screen(win, routineTimer)
# Test routine
data = no_adaptation(sc_states, location_ind_list)
# Save the data
data_path = './'+ str(expInfo['participant']) +'_No_adaptation.csv'
data.to_csv(data_path)
# End screen
end_screen(win, routineTimer)