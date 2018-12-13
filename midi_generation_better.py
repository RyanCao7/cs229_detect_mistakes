'''
* The purpose of this file is to take in an example expert midi segment, and
* modify it using mido to generate random added, subtracted, and edited notes,
* as well as varying tempi and dynamics slightly.
* Vocabulary -
*   1. Metadata_track - the first of the two processed midi tracks. Contains tempo
*       change and other metadata events (key signature, composer, title, etc.)
*   2. Piano_track - the second of the two processed midi tracks. Contains note
*       events and other sound-y events (e.g. pedal changes).
*   3. Message - any event that occurs in a track. Messages store information about
*       themselves, (e.g. pitch and velocity for a note message, bpm for a tempo
*       change message, etc.) and all have a time field which is measured in ticks
*       (these are independent of the tempo), relative to the previous message in
*       that same track.
'''

import mido
import copy
import random
import glob
import math
import os
import time
import numpy as np
import argparse

from tqdm import tqdm

NUM_EXAMPLES = 5000

DATASET_PATH = "/Volumes/Ryan_Cao/cs229_dataset/"
ADDED_PATH = "added/"
REMOVED_PATH = "removed/"
EDITED_PATH = "edited/"
EXPERT_PATH = 'midi/expert.mid'

TEST_FILE = "MIDI_sample.mid"#chpn_op25_e11.mid"
TEST_FILE = "debussy/DEB_PASS.MID"
# TEST_FILE = "chopin/chpn_op25_e11.mid"
TEST_FILE = "bach/bach_846.mid"
TEST_FILE = "examples/new_0_test.mid"

class Message:
    def __init__(self, event, time):
        self.event = event
        self.time = time

def raw_has_offset(preproc_midi, note_event, note_idx):
    for i in range(note_idx + 1, len(preproc_midi.tracks[1])):
        checking_event = preproc_midi.tracks[1][i]
        if checking_event.type in ['note_on', 'note_off']:
            if checking_event.note == note_event.note and (checking_event.type == 'note_off' or checking_event.velocity == 0):
                return True
    return False

def raw_sanity_check(preproc_midi):
    for i, note_event in enumerate(preproc_midi.tracks[1]):
        if note_event.type == 'note_on' and note_event.velocity > 0:
            assert(raw_has_offset(preproc_midi, note_event, i))
    print('passed raw sanity check!')

# Idea: 2 parallel lists for note events; 2 parallel lists for tempo events
# I: Actual midi events (disregard .time instance field)
# II: Absolute time in ticks corresponding to each midi event

# Takes in a normal piano midi file (i.e. one with multiple tracks, including
# left and right hand tracks), and merges them all into two tracks - a single
# track for tempo and other metadata, which is stored in midi_file.tracks[0],
# and another track for note events and pedal events, which is stored in
# midi_tracks[1].
# @return processed midi file with two tracks.
def preprocess_midi(midi):
    # Assume this is the meta-data and tempo track
    meta_data_track = midi.tracks[0]

    # Assume these are the piano left hand and right hand tracks
    right_hand_track = midi.tracks[1]
    left_hand_track = midi.tracks[2]

    # Merge hands into one track
    merged_track = mido.MidiTrack()
    merged_track.name = "Piano"

    # Performs merging of left and right hand tracks
    done = False
    i, j = 0, 0
    i_time, j_time, merge_time = 0, 0, 0
    while not done:
        # Check which item in either list happens first and append that first to the merged events

        # Indices aren't out of bounds
        if i < len(right_hand_track) and j < len(left_hand_track):
            if right_hand_track[i].time + i_time < left_hand_track[j].time + j_time:
                modified_message = right_hand_track[i].copy()
                modified_message.time = right_hand_track[i].time + i_time - merge_time
                merge_time = right_hand_track[i].time + i_time

                merged_track.append(modified_message)
                i_time += right_hand_track[i].time
                i += 1
            else:
                modified_message = left_hand_track[j].copy()
                modified_message.time = left_hand_track[j].time + j_time - merge_time
                merge_time = left_hand_track[j].time + j_time

                merged_track.append(modified_message)
                j_time += left_hand_track[j].time
                j += 1

        # Indices are out of bounds
        elif i == len(right_hand_track) and j < len(left_hand_track):
            modified_message = left_hand_track[j].copy()
            modified_message.time = left_hand_track[j].time + j_time - merge_time
            merge_time = left_hand_track[j].time + j_time

            merged_track.append(modified_message)
            j_time += left_hand_track[j].time
            j += 1
        elif i < len(right_hand_track) and j == len(right_hand_track):
            modified_message = right_hand_track[i].copy()
            modified_message.time = right_hand_track[i].time + i_time - merge_time
            merge_time = right_hand_track[i].time + i_time

            merged_track.append(modified_message)
            i_time += right_hand_track[i].time
            i += 1
        else:
            done = True

    preprocessed_midi = mido.MidiFile()
    preprocessed_midi.ticks_per_beat = midi.ticks_per_beat
    preprocessed_midi.tracks.extend([meta_data_track, merged_track])

    raw_sanity_check(preprocessed_midi)

    return preprocessed_midi

# Converts from ticks to seconds
def ticks_to_seconds(ticks, ticks_per_beat, microseconds_per_beat):
    beats = ticks / ticks_per_beat
    microseconds = beats * microseconds_per_beat
    return microseconds / 1000000.

# Passes back a list of onset messages
def get_onsets(midi_data):
    onsets = []
    for note in midi_data['notes']:
        if note.event.type == 'note_on' and note.event.velocity > 0:
            onsets.append(note)
    return onsets

# For midi_data only.
def has_offset(midi_data, onset_message):
    assert(onset_message.event.type == 'note_on' and onset_message.event.velocity > 0)
    begin_idx = midi_data['notes'].index(onset_message)
    for i in range(begin_idx + 1, len(midi_data['notes'])):
        if midi_data['notes'][i].event.note == onset_message.event.note and (midi_data['notes'][i].event.type == 'note_off' or midi_data['notes'][i].event.velocity == 0):
            return True
    return False

# Appends offsets to un-offset-onsets to the end of the track.
def add_offsets(midi_data):
    last_event_ticks = midi_data['notes'][-1].time
    for onset_message in get_onsets(midi_data):
        if not has_offset(midi_data, onset_message):
            offset_message = Message(mido.Message(type = 'note_off', note = onset_message.event.note, velocity = 0, time = 0), last_event_ticks)
            midi_data['notes'].append(offset_message)
    midi_data['notes'] = sorted(midi_data['notes'], key = lambda message: message.time)

# Takes in a preprocessed midi file and returns a dictionary of
# 'notes'
# 'tempo_changes'
# 'duration_in_ticks'
# 'num_onsets'
# 'min_note_len'
# 'max_note_len'
# 'ticks_per_beat' - int value pulled from original midi file
# Where the first two are parallel lists, as specified earlier,
# and the next pair are also parallel lists as specified earlier.
def convert_to_abs_format(preproc_midi):
    midi_data = {'notes': [],
                'tempo_changes': [],
                'meta_messages': [],
                'duration_in_ticks': 0,
                'num_onsets': 0,
                'min_note_len': 0,
                'max_note_len': 0,
                'ticks_per_beat': preproc_midi.ticks_per_beat}

    abs_ticks = 0
    for event in preproc_midi.tracks[0]:
        abs_ticks += event.time
        if event.type == 'set_tempo':
            midi_data['tempo_changes'].append(Message(event, abs_ticks))
        else:
            midi_data['meta_messages'].append(Message(event, abs_ticks))

    abs_ticks = 0
    for note_event in preproc_midi.tracks[1]:
        abs_ticks += note_event.time

        if note_event.time > 0 and note_event.time < midi_data['min_note_len']:
            midi_data['min_note_len'] = note_event.time

        if note_event.time > midi_data['max_note_len']:
            midi_data['max_note_len'] = note_event.time

        if note_event.type in ['note_on', 'note_off']:
            midi_data['notes'].append(Message(note_event, abs_ticks))
            if note_event.type == 'note_on':
                midi_data['num_onsets'] += 1

    add_offsets(midi_data)

    midi_data['duration_in_ticks'] = abs_ticks

    # print('before')
    # for note_message in midi_data['notes']:
    #     print(note_message.event, note_message.time)
    # print('before', midi_data['tempo_changes'])
    # sanity_check(midi_data)

    return midi_data

# Given midi_data dictionary, converts back into
# midi track file format.
def convert_back_to_midi_format(midi_data):

    sanity_check(midi_data)

    converted = mido.MidiFile()
    converted.ticks_per_beat = midi_data['ticks_per_beat']

    tempo_track = mido.MidiTrack()
    note_track = mido.MidiTrack()

    prev_note_time = 0
    for note_message in midi_data['notes']:
        # Unpacking
        note = note_message.event
        note_time = note_message.time
        # Converting back
        note.time = note_time - prev_note_time
        prev_note_time = note_time
        note_track.append(note)

    prev_tempo_time = 0
    for tempo_change_message in midi_data['tempo_changes']:
        # Unpacking
        tempo_change = tempo_change_message.event
        tempo_change_time = tempo_change_message.time
        # Converting back
        tempo_change.time = tempo_change_time - prev_tempo_time
        prev_tempo_time = tempo_change_time
        tempo_track.append(tempo_change)

    end_of_track_message = mido.MetaMessage(type = 'end_of_track', time = 0)
    note_track.append(end_of_track_message)
    tempo_track.append(end_of_track_message)

    converted.tracks.extend([tempo_track, note_track])
    return converted

# Helper function. Note that message is a Message object.
def insert_note_message(midi_data, message):
    insert_idx = -1
    for i, note_message in enumerate(midi_data['notes']):
        if note_message.time >= message.time:
            insert_idx = i
            break
    if insert_idx == -1:
        midi_data['notes'].append(message)
    else:
        midi_data['notes'].insert(insert_idx, message)
    midi_data['notes'] = sorted(midi_data['notes'], key = lambda m: m.time)

# Helper function. Note that message is a Message object.
def insert_tempo_message(midi_data, message):
    insert_idx = -1
    for i, tempo_message in enumerate(midi_data['tempo_changes']):
        if tempo_message.time >= message.time:
            insert_idx = i
            break
    if insert_idx == -1:
        midi_data['tempo_changes'].append(message)
    else:
        midi_data['tempo_changes'].insert(insert_idx, message)
    midi_data['tempo_changes'] = sorted(midi_data['tempo_changes'], key = lambda m: m.time)

# Given a tick time at the specified time stamp, changes the rest of the absolute times by delta
def change_times(midi_data, time_stamp, delta = 0):
    if delta != 0:
        for tempo_message in midi_data['tempo_changes']:
            if tempo_message.time > time_stamp:
                tempo_message.time += delta

        for note_message in midi_data['notes']:
            if note_message.time > time_stamp:
                note_message.time += delta

        midi_data['tempo_changes'] = sorted(midi_data['tempo_changes'], key = lambda message: message.time)
        midi_data['notes'] = sorted(midi_data['notes'], key = lambda message: message.time)

# Gets the absolute time, in secs, given absolute ticks.
def get_abs_time_from_ticks(midi_data, abs_ticks):
    running_total = 0
    for i in range(len(midi_data['tempo_changes']) - 1):
        if abs_ticks <= midi_data['tempo_changes'][i].time:
            diff = abs_ticks - midi_data['tempo_changes'][i].time
            running_total += ticks_to_seconds(diff, midi_data['ticks_per_beat'], midi_data['tempo_changes'][i].event.tempo)
            return running_total
        else:
            diff = midi_data['tempo_changes'][i + 1].time - midi_data['tempo_changes'][i].time
            running_total += ticks_to_seconds(diff, midi_data['ticks_per_beat'], midi_data['tempo_changes'][i].event.tempo)

    diff = abs_ticks - midi_data['tempo_changes'][-1].time
    running_total += ticks_to_seconds(diff, midi_data['ticks_per_beat'], midi_data['tempo_changes'][-1].event.tempo)
    return running_total

# Given a note onset message, returns the corresponding offset and the index.
def get_offset(midi_data, onset_message):
    assert(onset_message.event.type == 'note_on' and onset_message.event.velocity > 0)
    begin_idx = midi_data['notes'].index(onset_message)
    for i in range(begin_idx + 1, len(midi_data['notes'])):
        if midi_data['notes'][i].event.note == onset_message.event.note and (midi_data['notes'][i].event.type == 'note_off' or midi_data['notes'][i].event.velocity == 0):
            return midi_data['notes'][i], i

    assert(1 == 2) # Should never get here
    return None

# Edits the given note with new value, velocity, and change in ticks/note length.
def edit_note(midi_data, onset_message, new_note_val = -1, new_velocity = -1, new_length_delta_ticks = 0):
    offset_message, offset_index = get_offset(midi_data, onset_message)
    if new_note_val >= 0:
        onset_message.event.note = new_note_val
        offset_message.event.note = new_note_val
    if new_velocity >= 0:
        onset_message.event.velocity = new_velocity
    if new_length_delta_ticks != 0:
        change_times(midi_data, onset_message.time, delta = new_length_delta_ticks)
    # sanity_check(midi_data)

# Removes a single note from the midi file, also taking away its duration
def remove_note(midi_data, note_message):
    # Asserts it's an onset we got
    assert(note_message.event.type == 'note_on' and note_message.event.velocity > 0)
    note_offset_message, note_offset_index = get_offset(midi_data, note_message)
    note_index = midi_data['notes'].index(note_message)
    delta_t = note_message.time - midi_data['notes'][note_index + 1].time

    # Pushes all times back after the note time by delta_t
    change_times(midi_data, note_message.time, delta = delta_t)

    # Removes the note and its offset from note messages and time changes lists
    midi_data['notes'].remove(note_message)
    midi_data['notes'].remove(note_offset_message)
    # sanity_check(midi_data)

# Adds a single note, given an onset, when to add it, and its note length.
def add_note(midi_data, onset_message, note_length, delay = 0):
    offset_message = Message(mido.Message(type = 'note_off', note = onset_message.event.note, velocity = 0, time = 0), onset_message.time + note_length)
    insert_note_message(midi_data, onset_message)
    # Moves the next note over by delay ticks
    change_times(midi_data, onset_message.time, delta = delay)
    insert_note_message(midi_data, offset_message)
    sanity_check(midi_data)

# Retuns the length, in ticks, of a note (number of ticks between onset and offset)
def length_note_ticks(midi_data, onset_message):
    offset_message, idx = get_offset(midi_data, onset_message)
    return offset_message.time - onset_message.time

# Returns the tempo event closest (but before) a note event
def get_current_tempo(midi_data, note_event):
    find_idx = -1
    for i in range(len(midi_data['tempo_changes']) - 1):
        if midi_data['tempo_changes'][i].time > note_event.time:
            find_idx = i
            break
    return midi_data['tempo_changes'][find_idx]

# Performs a sanity check on midi data
def sanity_check(midi_data):
    for i in range(len(midi_data['notes']) - 1):
        # if not (midi_data['note_times'][i] <= midi_data['note_times'][i + 1]):
        #     print(midi_data['note_times'][i], midi_data['note_times'][i + 1])
        assert(midi_data['notes'][i].time <= midi_data['notes'][i + 1].time)
    for i in range(len(midi_data['tempo_changes']) - 1):
        # if not (midi_data['tempo_change_times'][i] <= midi_data['tempo_change_times'][i + 1]):
        #     print(midi_data['tempo_change_times'][i], midi_data['tempo_change_times'][i + 1])
        assert(midi_data['tempo_changes'][i].time <= midi_data['tempo_changes'][i + 1].time)

    for onset in get_onsets(midi_data):
        assert(onset.event.type == 'note_on' and onset.event.velocity > 0)
        assert(has_offset(midi_data, onset))

# ---------------------------------------- Actual use functions --------------------------------------------

# Generates random noise around the velocities of each note.
def perturb_dynamics(midi_data, width = 0, offset = 0):
    for onset in get_onsets(midi_data):
        onset.event.velocity = max(min(onset.event.velocity + round(offset + random.uniform(-width, width)), 127), 1)

# Takes in a midi thingy and edits some number of notes in it randomly.
# Returns the times at the beginning and end of those edits.
def edit_random_notes(midi_data, num_notes = 0):
    mistake_times = []

    for i in range(num_notes):
        note_message_to_edit = random.choice(get_onsets(midi_data))
        next_note_message = midi_data['notes'][midi_data['notes'].index(note_message_to_edit) + 1]
        # Grabs new values randomly and edits note
        new_note_val = round(max(min(random.uniform(note_message_to_edit.event.note - 10, note_message_to_edit.event.note + 10), 127), 1))
        new_velocity = round(max(random.uniform(note_message_to_edit.event.velocity - 20, note_message_to_edit.event.velocity + 20), 1))
        new_length_delta_ticks = 0 # int(random.uniform(0.7, 1.2) * length_note_ticks(midi_data, note_message_to_edit))
        edit_note(midi_data, note_message_to_edit, new_note_val = new_note_val, new_velocity = new_velocity, new_length_delta_ticks = new_length_delta_ticks)

        # Collects labels on begin/end mistake times
        start_mistake_time = note_message_to_edit.time
        end_mistake_time = next_note_message.time
        mistake_times.append((get_abs_time_from_ticks(midi_data, start_mistake_time), get_abs_time_from_ticks(midi_data, end_mistake_time)))

    return mistake_times

# Takes in a midi thingy and adds some number of notes to it in random places.
# Returns the times at the beginning and end of those additions.
def add_random_notes(midi_data, num_notes = 0):
    mistake_times = []
    added_notes = []

    for i in range(num_notes):
        place_to_add = round(random.uniform(0, midi_data['duration_in_ticks']))

        # Grabs new values randomly and creates new note to be added
        random_onset_message = random.choice(get_onsets(midi_data))
        note = int(max(min(random.uniform(random_onset_message.event.note - 10, random_onset_message.event.note + 10), 127), 0))
        velocity = int(max(random.uniform(random_onset_message.event.velocity - 20, random_onset_message.event.velocity + 20), 1))
        duration = int(random.uniform(midi_data['min_note_len'] + 1, midi_data['max_note_len']))
        note_message_to_add = Message(mido.Message(type = 'note_on', velocity = velocity, note = note, time = 0), place_to_add)
        added_notes.append(note_message_to_add)
        add_note(midi_data, note_message_to_add, duration, delay = duration)

    for added_message in added_notes:
        # print('added_message:', added_message.event, added_message.time)
        next_note_message = midi_data['notes'][midi_data['notes'].index(added_message) + 1]
        # Collects labels on begin/end mistake times
        start_mistake_time = added_message.time
        end_mistake_time = next_note_message.time
        mistake_times.append((get_abs_time_from_ticks(midi_data, start_mistake_time), get_abs_time_from_ticks(midi_data, end_mistake_time)))

    return mistake_times

# Takes in a midi thingy and removes some number of notes from it randomly.
# Returns the times at the beginning and end of those removals.
def remove_random_notes(midi_data, num_notes = 0):
    mistake_times = []

    for i in range(num_notes):
        onset_message_to_remove = random.choice(get_onsets(midi_data))
        next_note_message = midi_data['notes'][midi_data['notes'].index(onset_message_to_remove) + 1]
        start_mistake_time = onset_message_to_remove.time
        end_mistake_time = next_note_message.time
        remove_note(midi_data, onset_message_to_remove)
        mistake_times.append((get_abs_time_from_ticks(midi_data, start_mistake_time), get_abs_time_from_ticks(midi_data, end_mistake_time)))

    return mistake_times

# Inserts tempo change events at every onset, with width and offset being in BPM.
def perturb_tempos(midi_data, width = 0, offset = 0):
    onsets = get_onsets(midi_data)
    # print('onsets', onsets)
    for onset_message in onsets:
        tempo_message = get_current_tempo(midi_data, onset_message)
        new_tempo = mido.bpm2tempo(mido.tempo2bpm(tempo_message.event.tempo) + offset + round(random.uniform(-width, width)))
        if tempo_message.time == onset_message.time:
            tempo_message.tempo = new_tempo
        else:
            new_tempo_message = Message(mido.MetaMessage(type = 'set_tempo', tempo = new_tempo, time = 0), onset_message.time)
            insert_tempo_message(midi_data, new_tempo_message)
    # sanity_check(midi_data)

# =========================================================================================================

# Wrapper for changing the length of a certain note by multiplier.
def edit_duration_multiplier(position, multiplier, midi, key_to_info, key_list, chord = True):

    # Grab current note and tempo messages
    current_note_message, current_tempo_message = key_to_info[position]

    # If we're treating the note as a note in a chord, find the last note in the chord.
    if chord:
        position, current_note_message, current_tempo_message = grab_last_chord_note(position, midi, key_to_info, key_list)

    # Grab next note and corresponding tempo change
    next_note_message, next_tempo_message = key_to_info[key_list[key_list.index(position) + 1]]

    # Grab new duration accordingly
    new_duration = next_note_message.time * multiplier

    edit_duration(position, new_duration, midi, key_to_info, key_list)

# Just random stuff
def test_lots():
    for file in glob.iglob("chopin/*.mid"):
        test(file)

# Takes in start and end times (in real seconds[!!!]) and saves the corresponding
# slice of midi under savefile.
def slice_between_times(start_time, end_time, midi, key_to_info, key_list, savefile = "examples/test.mid"):
    print(start_time, end_time)
    midi_copy = mido.MidiFile()

    sliced_meta_data_track = mido.MidiTrack()
    sliced_piano_track = mido.MidiTrack()
    midi_copy.tracks.append(sliced_meta_data_track)
    midi_copy.tracks.append(sliced_piano_track)

    # First copy over everything that's not a tempo change until the first non-zero duration message.
    for i, message in enumerate(midi.tracks[0]):
        if message.time != 0:
            break
        if 'set_tempo' not in message.type:
            sliced_meta_data_track.append(message)

    # Find first and last note_on events after and before the start time and end time stamps, respectively.
    first_onset_pos, first_seen_onset_message, first_tempo_change_message, first_note_abs_time, first_tempo_abs_time = find_nearest_onset(start_time, key_to_info, key_list, midi, False)
    last_onset_pos, last_seen_onset_message, last_tempo_change_message, last_note_abs_time, last_tempo_abs_time = find_nearest_onset(end_time, key_to_info, key_list, midi, True)

    # Find last note event (corresponding note_off event to the last note_on event)
    last_pos, last_message, last_tempo_message = find_offset(last_onset_pos, midi, key_to_info, key_list)

    # for i in range(midi.tracks[1].index(first_seen_onset_message), midi.tracks[1].index(last_message) + 1):
    #     sliced_piano_track.append(message)
    #
    # for i in range(midi.tracks[0].index(first_tempo_change_message), midi.tracks[0].index(last_tempo_message) + 1):
    #     sliced_meta_data_track.append(message)

    # Copy all events, inclusive, between the two.
    begin_copy = False
    for message in midi.tracks[1]:
        if not begin_copy and message is first_seen_onset_message:
            begin_copy = True
        if begin_copy:
            sliced_piano_track.append(message)
            if message is last_message:
                break

    begin_copy = False
    for message in midi.tracks[0]:
        if not begin_copy and message is first_tempo_change_message:
            begin_copy = True
        if begin_copy:
            sliced_meta_data_track.append(message)
            if message is last_tempo_message:
                break

    # Adjust the times of the starting note and tempo changes.
    first_note_pos, first_note_message = first_note_affected(first_onset_pos, midi, key_to_info, key_list)
    delta = diff_in_ticks(first_note_pos, first_onset_pos, midi, key_to_info, key_list)

    print("delta", delta)

    second_tempo_idx = first_onset_pos
    while key_to_info[second_tempo_idx][1] is first_tempo_change_message:
        second_tempo_idx = key_list[key_list.index(second_tempo_idx) + 1]

    pushback(key_to_info[second_tempo_idx], delta, midi_copy, key_to_info, key_list)

    midi_copy.save(savefile)

# Grabs and dumps into a folder the sliced midi segments
# TODO: I think it's still buggy
def slice_file(midi, key_to_info, key_list, seg_avg_len = 10):

    if not os.path.exists("examples/"):
        os.makedirs("examples/")

    # Grabs number of segments we have
    num_segs = math.ceil(midi.length / seg_avg_len)

    for i in range(num_segs):
        slice_between_times(i * seg_avg_len, min((i + 1) * seg_avg_len, int(midi.length)), midi, key_to_info, key_list, "examples/new_{}_test.mid".format(i))

# Mode should be 'add', 'remove', or 'edit'
# num_modified should be a tuple of (low, high)
# examples_per_mistake should be an int; how many examples to generate per mistake
# expert midi path should be the path to, well, the preprocessed (and sliced) expert midi
def generate_mistakes(mode, num_modified_range, examples_per_mistake = 1, expert_midi_path = TEST_FILE, dataset_path = DATASET_PATH):

    assert(num_modified_range[0] <= num_modified_range[1])
    assert(mode in {'add', 'remove', 'edit'})
    assert(examples_per_mistake >= 1)

    if mode == 'add':
        upper_level_dir = ADDED_PATH
    elif mode == 'remove':
        upper_level_dir = REMOVED_PATH
    else: # mode == 'edited'
        upper_level_dir = EDITED_PATH

    # Makes the upper level directories
    if not os.path.exists(dataset_path + upper_level_dir):
        os.makedirs(dataset_path + upper_level_dir)

    # Creates this run's directory
    this_run_dir = dataset_path + upper_level_dir + time.strftime("%d-%m-%Y_at_%H:%M:%S", time.localtime()) + "_with_" + str((num_modified_range[1] - num_modified_range[0]) * examples_per_mistake) + "_examples/"
    os.makedirs(this_run_dir)

    # Sets up expert midi
    expert_midi = mido.MidiFile(expert_midi_path)
    # raw_sanity_check(expert_midi)
    expert_midi_data = convert_to_abs_format(expert_midi)

    # For each specific number of mistakes...
    for i in tqdm(range(num_modified_range[0], num_modified_range[1] + 1)):

        # Create folder for each number of mistakes
        num_mistakes_dir = this_run_dir + "{}_mistakes/".format(i)
        os.mkdir(num_mistakes_dir)

        # Generate mistake examples
        for j in range(examples_per_mistake):

            this_file_dir = num_mistakes_dir + "run_{}/".format(j)
            os.mkdir(this_file_dir)

            # Copies expert midi data first
            student_midi_data = copy.deepcopy(expert_midi_data)
            sanity_check(student_midi_data)
            # Labels for autogenerated mistakes
            mistake_file = open(this_file_dir + "file_{}_with_{}_mistakes.txt".format(j, i), "w")

            # Perturbs some stuff on the student's side. TODO: Randomly do offsets
            perturb_tempos(student_midi_data, width = 2, offset = 0)
            perturb_dynamics(student_midi_data, width = 7, offset = 0)

            # TODO: Implement add_random_notes and edit_random_notes
            mistake_list = None
            if mode == 'add':
                mistake_list = add_random_notes(student_midi_data, num_notes = i)
            elif mode == 'remove':
                mistake_list = remove_random_notes(student_midi_data, num_notes = i)
            else: # mode == 'edited'
                mistake_list = edit_random_notes(student_midi_data, num_notes = i)

            for item in mistake_list:
                mistake_file.write(str(item[0]) + " " + str(item[1]) + "\n")

            mistake_file.close()

            student_midi = convert_back_to_midi_format(student_midi_data)
            student_midi.save(this_file_dir + "run_{}_with_{}_mistakes.mid".format(j, i))

    return this_run_dir

# --------------------------------- ACTUAL RUNNING CODE ---------------------------------
parser = argparse.ArgumentParser(description = "Automagic generation of midi files with randomly added mistakes.")
parser.add_argument("expert_path", type = str, nargs = '?', default = EXPERT_PATH, help= "The full path to the expert midi.")
parser.add_argument("mode", type = str, help = "The mode you're using. Choose from 'add', 'edit', or 'remove'.")
parser.add_argument("low", type = int, help = "Lower bound number of mistakes to generate.")
parser.add_argument("high", type = int, help = "Upper bound number of mistakes to generate.")
parser.add_argument("examples_per_mistake", type = int, help = "Number of examples to generate per mistake. Total number examples_per_mistake * (high - low + 1)")
parser.add_argument("dataset_path", type = str, nargs = '?', default = DATASET_PATH, help = "path to store dataset")
args = parser.parse_args()

# Prints the directory that all of the generated midi got saved into.
# This is the path you can directly pass into all of the corruption.py functions.
print(generate_mistakes(args.mode, (args.low, args.high), args.examples_per_mistake, args.expert_path, args.dataset_path))
# ---------------------------------------------------------------------------------------
