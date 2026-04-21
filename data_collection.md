# Data collection description

## Set up


## Session
This is a description step-by-step how a session for one subject works.

Preperation:
1. Make sure that the table, chair and markers on the table are visible and correct. Make sure that camera is correctly positioned.
2. Verify informed consent.
3. Make sure that the subject is where cloths that does not blend in the background.
4. Explain to the subject what will happen in the session in broad terms.
5. Place the subject in the chair and show the markers on the table.
6. Explain to the subject where the left hand should start and where the right hand should be placed.
7. Explain and point at the markers where the hand will move through the session and where the hand should finish when done.

Trial:
1. Start the data collection script and verify that the pose is detected.
2. Explain that the leader of the experiment will count down 3, 2, 1 and then the recording of the movement starts. Tell the subject that it will record for 8 seconds.
3. Ask if the subject is ready and if yes, start the data collection and do one trial.
4. After the movement of the subject, the leader of the experiment verify a replay of the movement to see that the keypoints are visible and correct. 
5. Make three more trials that are slow, medium and fast. Tell the subject to slow down or speed up depending on how long the movement was during the first trial. Write down what commands where chosen.
6. Repeat 1-5 for the agreed number of instances.

One instance = 4 trials.

## Data
The data are stored depending on the subject, movement and trial according to:

`data/raw/subject_<XX>/<motion>/<>trial_<XXX>`

For example, subject 1 with a moving cup motion and trial 2 would be: `data/raw/subject_01/move_cup/trial_002`
Meta data of the trial is saved according to:
..
..
..


