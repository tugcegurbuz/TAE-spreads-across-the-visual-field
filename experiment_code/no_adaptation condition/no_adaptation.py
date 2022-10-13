import set_parameters
from psychopy import visual, event, clock, core
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
								STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import pandas as pd
import numpy as np

def no_adaptation(sc_states, location_ind_list):
    '''
    Input:
    -sc_states: the tensor that stores the information of the staircase states
    -location_ind_list: index list of how many staircases will be shown.
        This list will be replicated 25 times and shuffled to provide gratings randomly
        in each staircase conidition.
        The staircase has fixed trial number = 25 for every spatial location.
        There are two staircases for each spatial locations

    Output:
    -df: data frame of experimental data
    '''

    endExpNow = False

    ######################## SETUP ################################

    ## Initialize the components of data frame where experimental data will be stored
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

    ## Staircases
    # Initialize staircase states based on the visual field selected for the experiment
    if vf != 'r':
        for l in range(0, len(sc_states)):
            temp_x = sc_states[l][0][0]
            new_x = -temp_x
            sc_states[l][0][0] = new_x

    # Shuffle staircase sc_states
    random.shuffle(sc_states)
    # Create index list which will be used to select which sc states will be visible
    index_list = numTrials * location_ind_list
    # Shuffle the indexes for randomness
    random.shuffle(index_list)

    ## Components
    # Fixation Point
    fixationPoint = visual.ShapeStim(
        win=win, name='fixationPoint', vertices='cross',
        size=(0.5, 0.5),
        ori=0, pos=(0, 0),units='deg',
        lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
        fillColor=[-1,-1,-1], fillColorSpace='rgb',
        opacity=1, depth=-1.0, interpolate=True)

    # Blank Screens
    Blank_1Clock = core.Clock()
    blankScreen1 = visual.TextStim(win=win, name='blankScreen1',
        text=None,
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0,
        color='white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0)
    Blank_2Clock = core.Clock()
    textBlank = visual.TextStim(win=win, name='blankScreen2',
        text=' ',
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0,
        color='white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0)
    
    # Test
    TestClock = core.Clock()
    testGrating = visual.GratingStim(
        win=win, name='grating',units='deg',
        tex='sin', mask='gauss', size=(3, 3), sf=1.44, phase=0.0,
        color=[1,1,1], colorSpace='rgb', opacity=1.0 ,blendmode='avg',
        texRes=256, interpolate=True, depth=-2.0)

    for i in range(0, numTrials * len(location_ind_list)):
        ###Test routine
        # Start Routine "Test"
        continueRoutine = True

        #time variables
        t = 0
        TestClock.reset()  # clock
        frameN = -1

        # Select which staircase will be showed
        ind = index_list[i]
        this_sc = sc_states[ind][0]
        next_ori = 0

        # Get location and store it
        testLoc_x = this_sc[0]
        testLoc_xs.append(testLoc_x)
        testLoc_y = this_sc[1]
        testLoc_ys.append(testLoc_y)
        rawLoc = this_sc[3]
        raw_test_locs.append(rawLoc)

        # Set test grating location
        testLoc = (testLoc_x, testLoc_y)
        testGrating.setPos(testLoc)

        # Get staircase number and store
        sc_num = this_sc[2]
        staircase_no.append(sc_num)

        # Set test grating orientation and correct answers
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
        # Store the correct answers
        correct_answers.append(corrTestAns)

        # Initialize test response
        testResponse = event.BuilderKeyResponse()

        # Keep track of which components have finished
        TestComponents = [fixationPoint, testGrating, testResponse]
        for thisComponent in TestComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED

        while continueRoutine:
            # get current time
            t = TestClock.getTime()
            #frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
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
            frameRemains = 0.0 + 0.1- win.monitorFramePeriod * 0.75  # show test grating for 100 ms
            if testGrating.status == STARTED and t >= frameRemains:
                testGrating.setAutoDraw(False)

            # *testResponse* is taken for the next 200  ms
            if t >= 0.3 and testResponse.status == NOT_STARTED:
                # keep track of start time/frame for later
                testResponse.tStart = t
                testResponse.frameNStart = frameN  # exact frame index
                testResponse.status = STARTED

                # keyboard checking is just starting
                win.callOnFlip(testResponse.clock.reset)
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
                    # Store the accuracy
                    response_list = sc_states[ind][2]
                    accuracy = testResponse.corr
                    response_list.append(accuracy)
                    num_occurance = response_list.count(accuracy)

                    # Step size for the staircase
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

            # Append next ori to sc_states
            sc_states[ind][1].append(next_ori)

            # Check for quit (typically the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                core.quit()

            # Check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in TestComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            # refresh the screen
            if continueRoutine:
                win.flip()

        # Ending Routine "Test"
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

        # Store response correlations
        response_corrs.append(testResponse.corr)
        # Store test keys
        test_keys.append(testResponse.keys)
        # Store rts
        rts.append(testResponse.rt)
        routineTimer.reset()

        ###Blank Screen 2
        # Prepare to start Routine "Blank_2"
        t = 0
        Blank_2Clock.reset()
        frameN = -1
        continueRoutine = True
        routineTimer.add(2.000000)

        Blank_2Components = [textBlank, fixationPoint]

        for thisComponent in Blank_2Components:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED

        # Start Routine "Blank_2"
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = Blank_2Clock.getTime()
            frameN = frameN + 1

            # *fixationPoint* updates
            if t >= 0.0 and fixationPoint.status == NOT_STARTED:
                fixationPoint.setAutoDraw(True)

            # *textBlank* updates
            if t >= 0.0 and textBlank.status == NOT_STARTED:
                # keep track of start time/frame for later
                textBlank.tStart = t
                textBlank.frameNStart = frameN  # exact frame index
                textBlank.setAutoDraw(True)
            frameRemains = 0.0 + 2- win.monitorFramePeriod * 0.75  # show blank screen for 2 seconds

            if textBlank.status == STARTED and t >= frameRemains:
                textBlank.setAutoDraw(False)

            # check for quit (typically the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                core.quit()

            # check if all components have finished
            if not continueRoutine:
                break
            continueRoutine = False

            for thisComponent in Blank_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break
            # refresh the screen
            if continueRoutine:
                win.flip()

        # Ending Routine "Blank_2"
        for thisComponent in Blank_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)

    #####Form the dataframe
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

    return df