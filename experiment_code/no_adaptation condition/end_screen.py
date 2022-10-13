from psychopy import visual, event, clock, core
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
								STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

def end_screen(win, routineTimer):
    
    endExpNow = False

    end_ScreenClock = core.Clock()
    endText = visual.TextStim(win=win, name='endText',
        text='''This part of the experiment is COMPLETE!\n\n
            Please press SPACE to continue.''',
        font='Arial', units = 'deg',
        pos=(0, 0), height = 0.7, wrapWidth=None, ori=0,
        color='black', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0)

    # Prepare to start Routine "end_Screen"
    t = 0
    end_ScreenClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    # update component parameters for each repeat
    end_resp = event.BuilderKeyResponse()
    # keep track of which components have finished
    end_ScreenComponents = [endText, end_resp]
    for thisComponent in end_ScreenComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    # Start Routine "end_Screen"
    while continueRoutine:
        # get current time
        t = end_ScreenClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)

        if t >= 0.0 and endText.status == NOT_STARTED:
            # keep track of start time/frame for later
            endText.tStart = t
            endText.frameNStart = frameN  # exact frame index
            endText.setAutoDraw(True)

        if t >= 0.0 and end_resp.status == NOT_STARTED:
            # keep track of start time/frame for later
            end_resp.tStart = t
            end_resp.frameNStart = frameN  # exact frame index
            end_resp.status = STARTED
            # keyboard checking is just starting
            event.clearEvents(eventType='keyboard')
        if end_resp.status == STARTED:
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
        for thisComponent in end_ScreenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished

        # refresh the screen
        if continueRoutine:
            win.flip()

    # Ending Routine "end_Screen"
    for thisComponent in end_ScreenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    routineTimer.reset()