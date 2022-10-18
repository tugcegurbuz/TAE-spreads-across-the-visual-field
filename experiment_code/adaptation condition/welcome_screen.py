from psychopy import visual, event, clock, core
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
								STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

def welcome_screen(win, routineTimer):
    
    endExpNow = False

    ##Welcome Screen text
    Welcome_ScreenClock = core.Clock()
    welcome_text = visual.TextStim(win=win, name='welcome_text',
        text='''Welcome to the experiment!\n\n
                In this part of experiment you are expected to fixate at the cross sign at center of the screen when the gratings appear.\n
                You need to pay attention to gratings WITHOUT LOOKING AT THEM and indicate the direction of the tilt of the grating that appears after initial flickering grating.\n
                If you perceive the grating as tilted to;\n
                Right/Clockwise:press right arrow key ->\n
                Left/Counter-clockwise: press left arrow key <-\n\n
                Feel free to ask any questions to the experimenter before you start to experiment...\n
                If you are ready, please press SPACE to start.\n''',
        font='Arial',
        pos=(0, 0), height=0.7, wrapWidth=None, ori=0,
        color='black', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0, units = 'deg')

    ## Prepare to start Routine "Welcome_Screen"
    t = 0
    clock.reset()  # clock
    frameN = -1
    continueRoutine = True

    # update component parameters for each repeat
    welcome_resp = event.BuilderKeyResponse()

    # keep track of which components have finished
    Welcome_ScreenComponents = [welcome_text, welcome_resp]

    for thisComponent in Welcome_ScreenComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    #Start Routine "Welcome_Screen"
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
    # Ending Routine "Welcome_Screen"
    for thisComponent in Welcome_ScreenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    routineTimer.reset()