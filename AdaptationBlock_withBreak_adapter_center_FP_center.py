#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This experiment was created using PsychoPy.
This code tests the TAE magnitude for a central adapter and central fixation point condition.
'''

from __future__ import absolute_import, division
from psychopy import visual, event, gui, data, logging, clock, core
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
								STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np
from numpy.random import random, randint, normal, shuffle
import os
import sys
import pandas as pd
import random


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '3.0.1'
expName = 'Adaptation_adapter_center_FP_center'
expInfo = {'participant': '', 'session': '', 'vf': ''}

dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
	core.quit()  # user pressed cancel

expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem
filename = _thisDir + os.sep + u'data/%s_%s' % (expInfo['participant'], expName)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen
endExpNow = False  # flag for 'escape' or other condition => quit the exp

# Setup the Window
win = visual.Window(
	size=[1920, 1200], fullscr=True, screen=0,
	allowGUI=False, allowStencil=False,
	monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
	blendMode='avg', useFBO=True, units = 'deg')
win.mouseVisible=False
#win.setRecordFrameIntervals(True)


# Initialize components for Routine "Welcome_Screen"
Welcome_ScreenClock = core.Clock()
welcome_text = visual.TextStim(win=win, name='welcome_text',
	text='''Welcome to the experiment!\n\n
In this part of experiment you are expected to fixate at the cross sign at center of the screen when the gratings appear.\n
You need to pay attention to gratings WITHOUT LOOKING AT THEM and indicate the direction of the tilt of the grating \n
If you perceive the grating as tilted to;\n
Right/Clockwise:press right arrow key ->\n
Left/Counter-clockwise: press left arrow key <-\n
You will have two break screens. You are expected to stay in the experiment room and rest your eyes.n
Once you are ready you can continue to the other half of the experiment by pressing SPACE key.\n\n
Feel free to ask any questions before you start to experiment...\n
If you are ready, please press SPACE to start.\n''',
	font='Arial',
	pos=(0, 0), height=0.7, wrapWidth=None, ori=0,
	color='black', colorSpace='rgb', opacity=1,
	languageStyle='LTR',
	depth=0.0, units = 'deg');

#Initialize components for Routine "Adaptation"
AdaptationClock = core.Clock()
fixationPoint = visual.ShapeStim(
	win=win, name='fixationPoint', vertices='cross',
	size=(0.5, 0.5),
	ori=0, pos=(0, 0),units='deg',
	lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
	fillColor=[-1,-1,-1], fillColorSpace='rgb',
	opacity=1, depth=-1.0, interpolate=True)
adaptorGrating = visual.GratingStim(
	win=win, name='adaptorGrating',units='deg',
	tex='sin', mask='gauss',
	ori=15.0, pos=[0,0], size=(3, 3), sf=1.44, phase=0.0,
	color=[1,1,1], colorSpace='rgb', opacity=1.0 ,blendmode='avg',
	texRes=256, interpolate=True, depth=-2.0)

# Initialize components for Routine "Blank_1"
Blank_1Clock = core.Clock()
blankScreen1 = visual.TextStim(win=win, name='blankScreen1',
	text=None,
	font='Arial',
	pos=(0, 0), height=0.1, wrapWidth=None, ori=0,
	color='white', colorSpace='rgb', opacity=1,
	languageStyle='LTR',
	depth=0.0);

# Initialize components for Routine "Test"
TestClock = core.Clock()
testGrating = visual.GratingStim(
	win=win, name='grating',units='deg',
	tex='sin', mask='gauss', pos=[0,0], size=(3, 3), sf=1.44, phase=0.0,
	color=[1,1,1], colorSpace='rgb', opacity=1.0 ,blendmode='avg',
	texRes=256, interpolate=True, depth=-1.0)


# Initialize components for Routine "Blank_2"
Blank_2Clock = core.Clock()
textBlank = visual.TextStim(win=win, name='blankScreen2',
	text=' ',
	font='Arial',
	pos=(0, 0), height=0.1, wrapWidth=None, ori=0,
	color='white', colorSpace='rgb', opacity=1,
	languageStyle='LTR',
	depth=0.0);

# Initialize components for Routine "Thanks_Screen"
Thanks_ScreenClock = core.Clock()
thanksText = visual.TextStim(win=win, name='thanksText',
	text='''This part of the experiment is complete!\n
		  Please press SPACE to continue.''',
	font='Arial', units = 'deg',
	pos=(0, 0), height = 0.8, wrapWidth=None, ori=0,
	color='black', colorSpace='rgb', opacity=1,
	languageStyle='LTR',
	depth=0.0);

# Initialize components for Routine "Break_Screen"
Break_ScreenClock = core.Clock()
break_text = visual.TextStim(win=win, name='breakText',
	text='''1/3 of the experiment is complete!\n
		  You can take a break now and rest your eyes.\n
		  You are expected to stay in the experiment room. \n
		  When you are ready to continue, please press SPACE to continue.''',
	font='Arial', units = 'deg',
	pos=(0, 0), height = 0.8, wrapWidth=None, ori=0,
	color='black', colorSpace='rgb', opacity=1,
	languageStyle='LTR',
	depth=0.0);

Break_Screen2Clock = core.Clock()
break_text2 = visual.TextStim(win=win, name='thanksText',
	text='''2/3 of the experiment is complete!\n
		  You can take a break now and rest your eyes.\n
		  You are expected to stay in the experiment room. \n
		  When you are ready to continue, please press SPACE to continue.''',
	font='Arial', units = 'deg',
	pos=(0, 0), height = 0.8, wrapWidth=None, ori=0,
	color='black', colorSpace='rgb', opacity=1,
	languageStyle='LTR',
	depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine


# ------Prepare to start Routine "Welcome_Screen"-------
t = 0
Welcome_ScreenClock.reset()  # clock
frameN = -1
continueRoutine = True

# update component parameters for each repeat
welcome_resp = event.BuilderKeyResponse()

# keep track of which components have finished
Welcome_ScreenComponents = [welcome_text, welcome_resp]

for thisComponent in Welcome_ScreenComponents:
	if hasattr(thisComponent, 'status'):
		thisComponent.status = NOT_STARTED

# -------Start Routine "Welcome_Screen"-------
while continueRoutine:
	# get current time
	t = Welcome_ScreenClock.getTime()
	frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
	# update/draw components on each frame

	# *welcome_text* updates
	if t >= 0.0 and welcome_text.status == NOT_STARTED:
		# keep track of start time/frame for later
		welcome_text.tStart = t
		welcome_text.frameNStart = frameN  # exact frame index
		welcome_text.setAutoDraw(True)

	# *welcome_resp* updates
	if t >= 0.0 and welcome_resp.status == NOT_STARTED:
		# keep track of start time/frame for later
		welcome_resp.tStart = t
		welcome_resp.frameNStart = frameN  # exact frame index
		welcome_resp.status = STARTED
		# keyboard checking is just starting
		event.clearEvents(eventType='keyboard')
	if welcome_resp.status == STARTED:
		theseKeys = event.getKeys(keyList=['space'])

		# check for quit:
		if "escape" in theseKeys:
			endExpNow = True
		if len(theseKeys) > 0:  # at least one key was pressed
			# a response ends the routine
			continueRoutine = False

	# check for quit (typically the Esc key)
	if endExpNow or event.getKeys(keyList=["escape"]):
		core.quit()

	# check if all components have finished
	if not continueRoutine:  # a component has requested a forced-end of Routine
		break
	continueRoutine = False  # will revert to True if at least one component still running
	for thisComponent in Welcome_ScreenComponents:
		if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
			continueRoutine = True
			break  # at least one component has not yet finished

	# refresh the screen
	if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
		win.flip()

# -------Ending Routine "Welcome_Screen"-------
for thisComponent in Welcome_ScreenComponents:
	if hasattr(thisComponent, "setAutoDraw"):
		thisComponent.setAutoDraw(False)

# the Routine "Welcome_Screen" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()


def mainRoutine(sc_states, location_ind_list):
	'''
	Staircase has fixed trial number = 25.
	So, it will repeat 25 times at each spatial location.
	'''

	endExpNow = False

	#Initialize the components of data frame
	staircase_no = []
	testLoc_xs = []
	testLoc_ys = []
	raw_test_locs = []
	correct_answers = []
	test_keys = []
	response_corrs = []
	rts = []
	orientations = []
	vf = expInfo['vf']

	#grating location
	if vf == 'r':
		grating_location = (10.5, 0)
	else:
		grating_location = (-10.5, 0)

	#If test grating is on the left side of the screen
	if grating_location < (0, 0):
		for l in range(0, len(sc_states)):
			temp_x = sc_states[l][0][0]
			new_x = -temp_x
			sc_states[l][0][0] = new_x

	#shuffle staircase sc_states
	random.shuffle(sc_states)

	#index list
	index_list = 25 * location_ind_list
	random.shuffle(index_list)

	#total number of trials: 25 * [2staircases * x locations]
	for i in range(0, 25 * len(location_ind_list)):

		# -------Start Routine "Adaptation"-------
		continueRoutine = True

		adaptorGrating.setPos(grating_location)

		#time variables
		t = 0
		AdaptationClock.reset()  # clock
		frameN = -1
		routineTimer.add(5.000000)

		# keep track of which components have finished
		AdaptationComponents = [fixationPoint, adaptorGrating]

		for thisComponent in AdaptationComponents:
			if hasattr(thisComponent, 'status'):
				thisComponent.status = NOT_STARTED

		while continueRoutine and routineTimer.getTime() > 0:
			# get current time
			t = AdaptationClock.getTime()
			frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
			# update/draw components on each frame

			# *fixationPoint* updates
			if t >= 0.0 and fixationPoint.status == NOT_STARTED:
				fixationPoint.setAutoDraw(True)

			# *adaptorGrating* updates
			if t >= 0.0 and adaptorGrating.status == NOT_STARTED:
				# keep track of start time/frame for later
				adaptorGrating.tStart = t
				adaptorGrating.frameNStart = frameN  # exact frame index
				#flickering
				t1 = 0
				flickeringPeriod = 0.1
				t1 = globalClock.getTime()
				if t1 % flickeringPeriod < flickeringPeriod / 1.2:
					stim = adaptorGrating
				else:
					adaptorGrating.color *=-1
					stim = adaptorGrating
				stim.draw()

			frameRemains = 0.0 + 5- win.monitorFramePeriod * 0.75  # most of one frame period left
			if adaptorGrating.status == STARTED and t >= frameRemains:
				stim.setAutoDraw(False)

			# check for quit (typically the Esc key)
			if endExpNow or event.getKeys(keyList=["escape"]):
				core.quit()

			# check if all components have finished
			if not continueRoutine:  # a component has requested a forced-end of Routine
				break
			continueRoutine = False  # will revert to True if at least one component still running
			for thisComponent in AdaptationComponents:
				if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
					continueRoutine = True
					break  # at least one component has not yet finished

			# refresh the screen
			if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
				win.flip()

		# -------Ending Routine "Adaptation"-------
		for thisComponent in AdaptationComponents:
			if hasattr(thisComponent, "setAutoDraw"):
				thisComponent.setAutoDraw(False)

		# -------Start Routine "Blank_1"-------
		continueRoutine = True

		#time variables
		t = 0
		Blank_1Clock.reset()  # clock
		frameN = -1
		routineTimer.add(0.200000)

		# update component parameters for each repeat
		# keep track of which components have finished
		Blank_1Components = [blankScreen1, fixationPoint]

		for thisComponent in Blank_1Components:
			if hasattr(thisComponent, 'status'):
				thisComponent.status = NOT_STARTED

		while continueRoutine and routineTimer.getTime() > 0:
			# get current time
			t = Blank_1Clock.getTime()
			frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
			# update/draw components on each frame

			# *fixationPoint* updates
			if t >= 0.0 and fixationPoint.status == NOT_STARTED:
				fixationPoint.setAutoDraw(True)

			# *blankScreen1* updates
			if t >= 0.0 and blankScreen1.status == NOT_STARTED:
				# keep track of start time/frame for later
				blankScreen1.tStart = t
				blankScreen1.frameNStart = frameN  # exact frame index
				blankScreen1.setAutoDraw(True)
			frameRemains = 0.0 + 0.2- win.monitorFramePeriod * 0.75  # most of one frame period left
			if blankScreen1.status == STARTED and t >= frameRemains:
				blankScreen1.setAutoDraw(False)

			# check for quit (typically the Esc key)
			if endExpNow or event.getKeys(keyList=["escape"]):
				core.quit()

			# check if all components have finished
			if not continueRoutine:  # a component has requested a forced-end of Routine
				break
			continueRoutine = False  # will revert to True if at least one component still running
			for thisComponent in Blank_1Components:
				if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
					continueRoutine = True
					break  # at least one component has not yet finished

			# refresh the screen
			if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
				win.flip()

		# -------Ending Routine "Blank_1"-------
		for thisComponent in Blank_1Components:
			if hasattr(thisComponent, "setAutoDraw"):
				thisComponent.setAutoDraw(False)

		# ------Start Routine "Test"-------
		continueRoutine = True

		#time variables
		t = 0
		TestClock.reset()  # clock
		frameN = -1

		#we have 10 different staircase situations and we need to randomly show each
		#25 times
		ind = index_list[i]
		this_sc = sc_states[ind][0]
		next_ori = 0

		#get location and store
		testLoc_x = this_sc[0]
		testLoc_xs.append(testLoc_x)
		testLoc_y = this_sc[1]
		testLoc_ys.append(testLoc_y)
		rawLoc = this_sc[3]
		raw_test_locs.append(rawLoc)

		#set test grating location
		testLoc = (testLoc_x, testLoc_y)
		testGrating.setPos(testLoc)


		#get staircase number and store
		sc_num = this_sc[2]
		staircase_no.append(sc_num)

		#set test grating orientation and correct answers
		ori_list = sc_states[ind][1]
		ori = ori_list[-1]
		orientations.append(ori)
		testGrating.setOri(ori)

		if ori > 0:
			corrTestAns = 'right'
		elif ori < 0:
			corrTestAns = 'left'
		else:
			corrTestAns = 'vertical'

		correct_answers.append(corrTestAns)

		#Initialize test response
		testResponse = event.BuilderKeyResponse()

		# keep track of which components have finished
		TestComponents = [fixationPoint, testGrating, testResponse]
		for thisComponent in TestComponents:
			if hasattr(thisComponent, 'status'):
				thisComponent.status = NOT_STARTED

		while continueRoutine:
			# get current time
			t = TestClock.getTime()
			frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
			# update/draw components on each frame

			# *fixationPoint* updates
			if t >= 0.0 and fixationPoint.status == NOT_STARTED:
				fixationPoint.setAutoDraw(True)

			# *testGrating* updates
			if t >= 0.0 and testGrating.status == NOT_STARTED:
				# keep track of start time/frame for later
				testGrating.tStart = t
				testGrating.frameNStart = frameN  # exact frame index
				testGrating.setOpacity(1)
				testGrating.setAutoDraw(True)
			frameRemains = 0.0 + 0.1- win.monitorFramePeriod * 0.75  # most of one frame period left
			if testGrating.status == STARTED and t >= frameRemains:
				testGrating.setAutoDraw(False)

			# *testResponse* updates
			if t >= 0.3 and testResponse.status == NOT_STARTED:
				# keep track of start time/frame for later
				testResponse.tStart = t
				testResponse.frameNStart = frameN  # exact frame index
				testResponse.status = STARTED

				# keyboard checking is just starting
				win.callOnFlip(testResponse.clock.reset)  # t=0 on next screen flip
				event.clearEvents(eventType='keyboard')

			if testResponse.status == STARTED:
				theseKeys = event.getKeys(keyList=['right', 'left'])

				# check for quit:
				if "escape" in theseKeys:
					endExpNow = True
				if len(theseKeys) > 0:  # at least one key was pressed
					testResponse.keys = theseKeys[-1]  # just the last key pressed
					testResponse.rt = testResponse.clock.getTime()


					# was this 'correct'?
					if (testResponse.keys == str(corrTestAns)) or (testResponse.keys == corrTestAns):
						testResponse.corr = 1
					else:
						testResponse.corr = 0

					response_list = sc_states[ind][2]
					accuracy = testResponse.corr
					response_list.append(accuracy)
					num_occurance = response_list.count(accuracy)

					stepsize_list = [2, 1, 0.5]

					if (len(response_list) > 1) and (sc_states[ind][3] < 2):
						if num_occurance == 1:
							sc_states[ind][3] += 1
							response_list.clear()

					stepsize = stepsize_list[sc_states[ind][3]]

					if accuracy == 1:
						if ori < 0:
							next_ori = ori + stepsize
						else:
							next_ori = ori - stepsize
					else:
						if ori < 0:
							next_ori = ori - stepsize
						elif ori > 0:
							next_ori = ori + stepsize

					# a response ends the routine
					continueRoutine = False


			#append next ori to sc_states
			sc_states[ind][1].append(next_ori)

			# check for quit (typically the Esc key)
			if endExpNow or event.getKeys(keyList=["escape"]):
				core.quit()

			# check if all components have finished
			if not continueRoutine:  # a component has requested a forced-end of Routine
				break
			continueRoutine = False  # will revert to True if at least one component still running
			for thisComponent in TestComponents:
				if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
					continueRoutine = True
					break  # at least one component has not yet finished

			# refresh the screen
			if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
				win.flip()

		# -------Ending Routine "Test"-------
		for thisComponent in TestComponents:
			if hasattr(thisComponent, "setAutoDraw"):
				thisComponent.setAutoDraw(False)

		# check responses
		if testResponse.keys in ['', [], None]:  # No response was made
			testResponse.keys="None"
			# was no response the correct answer?!
			if str(corrTestAns).lower() == 'none':
			   testResponse.corr = 1;  # correct non-response
			else:
			   testResponse.corr = "None";  # failed to respond (incorrectly)

		#store response correlations
		response_corrs.append(testResponse.corr)

		#store test keys
		test_keys.append(testResponse.keys)

		#store rts
		rts.append(testResponse.rt)

		# the Routine "Test" was not non-slip safe, so reset the non-slip timer
		routineTimer.reset()

		# ------Prepare to start Routine "Blank_2"-------
		t = 0
		Blank_2Clock.reset()  # clock
		frameN = -1
		continueRoutine = True
		routineTimer.add(2.000000)

		# update component parameters for each repeat
		# keep track of which components have finished
		Blank_2Components = [textBlank, fixationPoint]

		for thisComponent in Blank_2Components:
			if hasattr(thisComponent, 'status'):
				thisComponent.status = NOT_STARTED

		# -------Start Routine "Blank_2"-------
		while continueRoutine and routineTimer.getTime() > 0:
			# get current time
			t = Blank_2Clock.getTime()
			frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
			# update/draw components on each frame

			# *fixationPoint* updates
			if t >= 0.0 and fixationPoint.status == NOT_STARTED:
				fixationPoint.setAutoDraw(True)

			# *textBlank* updates
			if t >= 0.0 and textBlank.status == NOT_STARTED:
				# keep track of start time/frame for later
				textBlank.tStart = t
				textBlank.frameNStart = frameN  # exact frame index
				textBlank.setAutoDraw(True)
			frameRemains = 0.0 + 2- win.monitorFramePeriod * 0.75  # most of one frame period left

			if textBlank.status == STARTED and t >= frameRemains:
				textBlank.setAutoDraw(False)

			# check for quit (typically the Esc key)
			if endExpNow or event.getKeys(keyList=["escape"]):
				core.quit()

			# check if all components have finished
			if not continueRoutine:  # a component has requested a forced-end of Routine
				break
			continueRoutine = False  # will revert to True if at least one component still running

			for thisComponent in Blank_2Components:
				if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
					continueRoutine = True
					break  # at least one component has not yet finished

			# refresh the screen
			if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
				win.flip()

		# -------Ending Routine "Blank_2"-------
		for thisComponent in Blank_2Components:
			if hasattr(thisComponent, "setAutoDraw"):
				thisComponent.setAutoDraw(False)

		routineTimer.reset()


	df = pd.DataFrame({'StaircaseNo': staircase_no,
		  'TestLoc_x':testLoc_xs,
		  'TestLoc_y':testLoc_ys,
		  'rawTestLoc': raw_test_locs,
		  'Orientation': orientations,
		  'CorrectAns': correct_answers,
		  'TestKey': test_keys,
		  'ResponseCorr': response_corrs,
		  'RT': rts
		  })

	# this function returns dataframe
	return df


start_angle_sc1 = 15.0
start_angle_sc2 = -15.0

sc_states = [
	[[6.5, 0, 1, 'cNear2'], [start_angle_sc1], [], 0],
	[[6.5, 0, 2, 'cNear2'], [start_angle_sc2], [], 0],
	[[10.5, 0, 1, 'cCenter'], [start_angle_sc1], [], 0],
	[[10.5, 0, 2, 'cCenter'], [start_angle_sc2], [], 0],
	[[14.5, 0, 1, 'cFurther1'], [start_angle_sc1], [], 0],
	[[14.5, 0, 2, 'cFurther1'], [start_angle_sc2], [], 0],
	[[6.5, -4, 1, 'dNear2'], [start_angle_sc1], [], 0],
    [[6.5, -4, 2, 'dNear2'], [start_angle_sc2], [], 0],
    [[10.5, -4, 1, 'dCenter'], [start_angle_sc1], [], 0],
    [[10.5, -4, 2, 'dCenter'], [start_angle_sc2], [], 0],
    [[14.5, -4, 1, 'dFurther1'], [start_angle_sc1], [], 0],
    [[14.5, -4, 2, 'dFurther1'], [start_angle_sc2], [], 0],
    [[6.5, 4, 1, 'uNear2'], [start_angle_sc1], [], 0],
    [[6.5, 4, 2, 'uNear2'], [start_angle_sc2], [], 0],
    [[10.5, 4, 1, 'uCenter'], [start_angle_sc1], [], 0],
    [[10.5, 4, 2, 'uCenter'], [start_angle_sc2], [], 0],
    [[14.5, 4, 1, 'uFurther1'], [start_angle_sc1], [], 0],
    [[14.5, 4, 2, 'uFurther1'], [start_angle_sc2], [], 0]
	]

# Get indexes to randomly sample the sc_states for breaks
inds=[0, 1, 2, 3, 4, 5, 6, 7, 8]
inds1 = random.sample(range(0, 9), 3)
inds_2 = [i for i in inds if i not in inds1]
inds2 = inds_2[0:3]
inds3 = inds_2[3:6]

# Init sc_state1 and sc_state2
sc_state1 = []
sc_state2 = []
sc_state3 = []

for i in inds1:
  sc_state1.append(sc_states[i*2])
  sc_state1.append(sc_states[i*2+1]) #looks ugly because of indexing list

for i in inds2:
  sc_state2.append(sc_states[i*2])
  sc_state2.append(sc_states[i*2+1]) #looks ugly because of indexing list

for i in inds3:
  sc_state3.append(sc_states[i*2])
  sc_state3.append(sc_states[i*2+1]) #looks ugly because of indexing list

routine_list = [sc_state1, sc_state2, sc_state3]
random.shuffle(routine_list)

# Run mainRoutine for sc_state1: for random 3 locations x 2 staircases
df1 = mainRoutine(routine_list[0], [0, 1, 2, 3, 4, 5])

# Save the first part to be safe
data_path = './'+ str(expInfo['participant']) +'_first_part_of_AdaptationBlock_adapter_center_FP_center.xlsx'
with pd.ExcelWriter(data_path) as writer:
	df1.to_excel(writer, sheet_name = '1_adaptation_center_FP_center', index = False)

# Break screen
# ------Prepare to start Routine "Break_Screen"-------
t = 0
Break_ScreenClock.reset()  # clock
frameN = -1
continueRoutine = True
# update component parameters for each repeat
break_resp = event.BuilderKeyResponse()
# keep track of which components have finished
Break_ScreenComponents = [break_text, break_resp]
for thisComponent in Break_ScreenComponents:
	if hasattr(thisComponent, 'status'):
		thisComponent.status = NOT_STARTED

# -------Start Routine "Break_Screen"-------
while continueRoutine:
	# get current time
	t = Break_ScreenClock.getTime()
	frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
	# update/draw components on each frame

	# *thanksText* updates
	if t >= 0.0 and break_text.status == NOT_STARTED:
		# keep track of start time/frame for later
		break_text.tStart = t
		break_text.frameNStart = frameN  # exact frame index
		break_text.setAutoDraw(True)

	# *thanks_resp* updates
	if t >= 0.0 and break_resp.status == NOT_STARTED:
		# keep track of start time/frame for later
		break_resp.tStart = t
		break_resp.frameNStart = frameN  # exact frame index
		break_resp.status = STARTED
		# keyboard checking is just starting
		event.clearEvents(eventType='keyboard')
	if break_resp.status == STARTED:
		theseKeys = event.getKeys(keyList=['space'])

		# check for quit:
		if "escape" in theseKeys:
			endExpNow = True
		if len(theseKeys) > 0:  # at least one key was pressed
			# a response ends the routine
			continueRoutine = False

	# check for quit (typically the Esc key)
	if endExpNow or event.getKeys(keyList=["escape"]):
		core.quit()

	# check if all components have finished
	if not continueRoutine:  # a component has requested a forced-end of Routine
		break
	continueRoutine = False  # will revert to True if at least one component still running
	for thisComponent in Break_ScreenComponents:
		if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
			continueRoutine = True
			break  # at least one component has not yet finished

	# refresh the screen
	if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
		win.flip()

# -------Ending Routine "Break_Screen"-------
for thisComponent in Break_ScreenComponents:
	if hasattr(thisComponent, "setAutoDraw"):
		thisComponent.setAutoDraw(False)
routineTimer.reset()

# Run mainRoutine for sc_state2: for random 3 locations x 2 staircases
df2 = mainRoutine(routine_list[1], [0, 1, 2, 3, 4, 5])

# Save the first part to be safe
data_path = './'+ str(expInfo['participant']) +'_second_part_AdaptationBlock_adapter_center_FP_center.xlsx'
with pd.ExcelWriter(data_path) as writer:
	df2.to_excel(writer, sheet_name = '_2_adapter_center_FP_center', index = False)

# Break screen
# ------Prepare to start Routine "Break_Screen"-------
t = 0
Break_Screen2Clock.reset()  # clock
frameN = -1
continueRoutine = True
# update component parameters for each repeat
break_resp2 = event.BuilderKeyResponse()
# keep track of which components have finished
Break2_ScreenComponents = [break_text2, break_resp2]
for thisComponent in Break2_ScreenComponents:
	if hasattr(thisComponent, 'status'):
		thisComponent.status = NOT_STARTED
print('k')
# -------Start Routine "Break_Screen"-------
while continueRoutine:
	# get current time
	t = Break_Screen2Clock.getTime()
	frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
	# update/draw components on each frame

	# *thanksText* updates
	if t >= 0.0 and break_text2.status == NOT_STARTED:
		# keep track of start time/frame for later
		break_text2.tStart = t
		break_text2.frameNStart = frameN  # exact frame index
		break_text2.setAutoDraw(True)

	# *thanks_resp* updates
	if t >= 0.0 and break_resp2.status == NOT_STARTED:
		# keep track of start time/frame for later
		break_resp2.tStart = t
		break_resp2.frameNStart = frameN  # exact frame index
		break_resp2.status = STARTED
		# keyboard checking is just starting
		event.clearEvents(eventType='keyboard')
	if break_resp2.status == STARTED:
		theseKeys = event.getKeys(keyList=['space'])

		# check for quit:
		if "escape" in theseKeys:
			endExpNow = True
		if len(theseKeys) > 0:  # at least one key was pressed
			# a response ends the routine
			continueRoutine = False

	# check for quit (typically the Esc key)
	if endExpNow or event.getKeys(keyList=["escape"]):
		core.quit()

	# check if all components have finished
	if not continueRoutine:  # a component has requested a forced-end of Routine
		break
	continueRoutine = False  # will revert to True if at least one component still running
	for thisComponent in Break2_ScreenComponents:
		if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
			continueRoutine = True
			break  # at least one component has not yet finished

	# refresh the screen
	if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
		win.flip()

# -------Ending Routine "Break_Screen"-------
for thisComponent in Break2_ScreenComponents:
	if hasattr(thisComponent, "setAutoDraw"):
		thisComponent.setAutoDraw(False)
routineTimer.reset()

df3 = mainRoutine(routine_list[2], [0, 1, 2, 3, 4, 5])
#Concatanate the data frames
dfs = pd.concat([df1, df2, df3])

#Save the dataframes
data_path = './'+ str(expInfo['participant']) +'_AdaptationBlock_adapter_center_FP_center.xlsx'
with pd.ExcelWriter(data_path) as writer:
	dfs.to_excel(writer, sheet_name = 'adaptation_center_FP_center', index = False)

# ------Prepare to start Routine "Thanks_Screen"-------
t = 0
Thanks_ScreenClock.reset()  # clock
frameN = -1
continueRoutine = True
# update component parameters for each repeat
thanks_resp = event.BuilderKeyResponse()
# keep track of which components have finished
Thanks_ScreenComponents = [thanksText, thanks_resp]
for thisComponent in Thanks_ScreenComponents:
	if hasattr(thisComponent, 'status'):
		thisComponent.status = NOT_STARTED

# -------Start Routine "Thanks_Screen"-------
while continueRoutine:
	# get current time
	t = Thanks_ScreenClock.getTime()
	frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
	# update/draw components on each frame

	# *thanksText* updates
	if t >= 0.0 and thanksText.status == NOT_STARTED:
		# keep track of start time/frame for later
		thanksText.tStart = t
		thanksText.frameNStart = frameN  # exact frame index
		thanksText.setAutoDraw(True)

	# *thanks_resp* updates
	if t >= 0.0 and thanks_resp.status == NOT_STARTED:
		# keep track of start time/frame for later
		thanks_resp.tStart = t
		thanks_resp.frameNStart = frameN  # exact frame index
		thanks_resp.status = STARTED
		# keyboard checking is just starting
		event.clearEvents(eventType='keyboard')
	if thanks_resp.status == STARTED:
		theseKeys = event.getKeys(keyList=['space'])

		# check for quit:
		if "escape" in theseKeys:
			endExpNow = True
		if len(theseKeys) > 0:  # at least one key was pressed
			# a response ends the routine
			continueRoutine = False

	# check for quit (typically the Esc key)
	if endExpNow or event.getKeys(keyList=["escape"]):
		core.quit()

	# check if all components have finished
	if not continueRoutine:  # a component has requested a forced-end of Routine
		break
	continueRoutine = False  # will revert to True if at least one component still running
	for thisComponent in Thanks_ScreenComponents:
		if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
			continueRoutine = True
			break  # at least one component has not yet finished

	# refresh the screen
	if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
		win.flip()

# -------Ending Routine "Thanks_Screen"-------
for thisComponent in Thanks_ScreenComponents:
	if hasattr(thisComponent, "setAutoDraw"):
		thisComponent.setAutoDraw(False)
# the Routine "Thanks_Screen" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()
