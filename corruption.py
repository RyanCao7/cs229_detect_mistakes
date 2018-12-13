import pydub
import argparse
from pydub import AudioSegment
from tqdm import tqdm
import matplotlib as plot
plot.use('Agg')
import matplotlib.pyplot as plt
import random
import librosa
import librosa.display
import subprocess
import os
import numpy as np
import glob
from multiprocessing import Process

# folder_path = "app/static/White_Noise_Slicing/"
# sound1 = AudioSegment.from_file(folder_path + "Noise_1.wav")
# sound2 = AudioSegment.from_file("app/static/Model_Suite/ABABCDCDEFEFABAB_good1.wav")

# Hard crops the spectrogram to be 256 x 360
class ThreeSixtyCrop(object):

    def __init__(self):
        self.output_size = (256, 360)

    # Resizes spectrogram to be 256 x 360, and label to be 1x360
    def __call__(self, spectrogram, label):
        spectrogram = spectrogram[:, :360]
        label = label[:360]
        assert(spectrogram.shape == (256, 360))
        return spectrogram, label

# Silly wrapper - don't even know why it's here ¯\_(ツ)_/¯
def load_noise(path):
    return AudioSegment.from_file(path)

# Pad the foreground audio with anywhere from 0 to 2 seconds of background noise
# Foreground audio path is the path to the foreground audio (wav) file.
def add_background_noise(foreground_audio_path, background_noise_audio, gain=-15):
    # print(foreground_audio_path)
    foreground_audio = AudioSegment.from_file(foreground_audio_path)

    # Generates how much white noise should happen before/after the actual recording
    pre_noise = round(random.uniform(0, 2000))
    post_noise = round(random.uniform(0, 2000))

    noise_start = round(random.uniform(0, len(background_noise_audio) - (len(foreground_audio) + pre_noise + post_noise + 1)))
    noise_end = noise_start + len(foreground_audio) + pre_noise + post_noise + 1

    # Chooses a random place and cuts out a slice
    slice = background_noise_audio[noise_start : noise_end] + gain

    # Overlays and saves the noisy file
    noisy_audio = slice.overlay(foreground_audio, position = pre_noise, loop=False)
    noisy_file_name = foreground_audio_path.rstrip('.wav') + "_noisy.wav"
    noisy_audio.export(noisy_file_name, format = "wav")

    # Returns the name of the noisy file
    return pre_noise / 1000., noisy_file_name

# Given a midi filepath name, converts it into a .wav and saves it under the same folder,
# but with the .wav extension instead of .mid
def midi_to_wav(path):
    for midi_file in tqdm(glob.iglob(path + "**/**/*.mid")):
        filename = midi_file[:-4]
        subprocess.check_output(["timidity", midi_file, "-Ow", "-o", filename + ".wav"])

# Cleans out all wav files.
def cleanup(path):
    for wav_file in glob.iglob(path + "**/**/*dense_labels.txt"):
        os.remove(wav_file)

# Plots a single mel spectrogram to store_path, given the np.ndarray with the
# spectroram data and the labels.
def plot_mel_spectrogram(store_path, spectrogram, record_len, labels, fmax):
    # print(np.shape(spectrogram), np.shape(labels))
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', fmax=fmax, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    for i, label in enumerate(labels):
        if label > 0:
            plt.axvline(x = record_len * i / len(labels), color = 'g', linewidth = 3, alpha = 0.4)
    plt.savefig(store_path)
    plt.close()

def process_one_debug(midi_file_path, background_noise_path, save = False, n_mels=256, fmax=4096):
    filename = midi_file_path.rstrip('.mid')
    wav_file_path = filename + "_noisy.wav"
    FNULL = open(os.devnull, 'w')
    subprocess.call(["timidity", midi_file_path, "-Ow", "-o", wav_file_path], stdout=FNULL, stderr=subprocess.STDOUT)

    y, sr = librosa.load(wav_file_path)

    # Compute its duration
    duration = y.shape[0]/float(sr)

    # Find the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)

    np.save(wav_file_path.rstrip(".wav") + '_mel_%d_%d_' % (n_mels, fmax), mel_spectrogram)

    # -----------------------------------------------------------------------------------------------
    # Dense labels for every time step in the spectrogram

    noisy_audio_file = AudioSegment.from_file(wav_file_path)
    duration = noisy_audio_file.duration_seconds

    num_time_steps = mel_spectrogram.shape[1]
    time_steps_per_second = num_time_steps/duration

    # To generate dense labels
    metadata_path = glob.glob(wav_file_path[:wav_file_path.rfind("run_")] + "*.txt")[0]

    # Read in the mistake intervals
    with open(metadata_path) as f:

        mistake_intervals = []

        for line in f.readlines():
            processed = line.rstrip("\n").split(" ")
            mistake_intervals.append((float(processed[0]), float(processed[1])))

    # print(mistake_intervals)

    # Creates the labels for this student sample
    gt_labels = np.zeros(num_time_steps)
    for m_low, m_high in mistake_intervals:
        gt_labels[int((m_low) * time_steps_per_second):int((m_high) * time_steps_per_second) + 1] = 1

    # Write the labels to a dense label txt file
    # dense_label_file = open(wav_file_path.rstrip(".wav") + "_dense_labels.txt", "w")
    # for i, label in enumerate(gt_labels):
    #     dense_label_file.write(str(label) + " ") if i < len(gt_labels) - 1 else dense_label_file.write(str(label) + "\n")
    np.save(wav_file_path.rstrip(".wav") + '_dense_labels', gt_labels)


# Given a single midi file path, goes all the way to the .npy mel-spectrogram and saves it in the same directory.
def process_one(midi_file_path, background_noise_path, save = False, n_mels=256, fmax=4096):

    # Performs midi to wave conversion
    filename = midi_file_path.rstrip('.mid')
    wav_file_path = filename + ".wav"
    FNULL = open(os.devnull, 'w')
    subprocess.call(["timidity", midi_file_path, "-Ow", "-o", wav_file_path], stdout=FNULL, stderr=subprocess.STDOUT)
    # -----------------------------------------------------------------------------------------------

    # Performs noise addition on wave file
    background_noise_audio = load_noise(background_noise_path)
    pre_noise, noisy_file_path = add_background_noise(wav_file_path, background_noise_audio, gain = 2)

    # -----------------------------------------------------------------------------------------------

    # Performs the spectrogram generation and saving
    # Load the .wav file
    y, sr = librosa.load(noisy_file_path)

    # Compute its duration
    duration = y.shape[0]/float(sr)

    # Find the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)

    np.save(wav_file_path.rstrip(".wav") + '_mel_%d_%d_' % (n_mels, fmax), mel_spectrogram)

    # -----------------------------------------------------------------------------------------------
    # Dense labels for every time step in the spectrogram

    noisy_audio_file = AudioSegment.from_file(noisy_file_path)
    duration = noisy_audio_file.duration_seconds

    num_time_steps = mel_spectrogram.shape[1]
    time_steps_per_second = num_time_steps / duration

    # To generate dense labels
    metadata_path = glob.glob(wav_file_path[:wav_file_path.rfind("run_")] + "*.txt")[0]

    # Read in the mistake intervals
    with open(metadata_path) as f:

        mistake_intervals = []

        for line in f.readlines():
            processed = line.rstrip("\n").split(" ")
            mistake_intervals.append((float(processed[0]), float(processed[1])))

    # print(mistake_intervals)

    # Creates the labels for this student sample
    gt_labels = np.zeros(num_time_steps)
    for m_low, m_high in mistake_intervals:
        gt_labels[int((m_low + pre_noise) * time_steps_per_second):int((m_high + pre_noise) * time_steps_per_second) + 1] = 1

    # Write the labels to a dense label txt file
    # dense_label_file = open(wav_file_path.rstrip(".wav") + "_dense_labels.txt", "w")
    # for i, label in enumerate(gt_labels):
    #     dense_label_file.write(str(label) + " ") if i < len(gt_labels) - 1 else dense_label_file.write(str(label) + "\n")
    np.save(wav_file_path.rstrip(".wav") + '_dense_labels', gt_labels)

    duration_path = wav_file_path[:wav_file_path.rfind("/") + 1] + 'duration.txt'
    duration_file = open(duration_path, 'w')
    duration_file.write(str(duration) + '\n')
    duration_file.write(str(360 / time_steps_per_second) + '\n')
    duration_file.close()

    # dense_label_file.close()

    # -----------------------------------------------------------------------------------------------

    # Deletes the wav files, so we don't take up too much space
    if not save:
        os.remove(wav_file_path)
        os.remove(noisy_file_path)

# Input: the wav path folder with midi and noise files
# Output: generated spectrograms (saved as .npy files) inside the respective folders
def process_all(path, noise_path, num_threads, save = False):
    all_midi_files = glob.glob(path + "*/*/*.mid")
    # Number of times we have to start the threads and let them run over
    # (because Ryan is too lazy to learn Python multi-thread-safe locking and stuff)
    num_times = int(np.floor(len(all_midi_files) / num_threads))
    extra_times = len(all_midi_files) % num_threads

    # Goes through number of times we need to run all the threads
    for i in tqdm(range(num_times)):
        processes = []
        for j in range(num_threads):

            # Grabs midi file
            midi_file = all_midi_files[i * num_threads + j]

            # Starts the process
            p = Process(target = process_one, args = (midi_file, noise_path, save))
            p.start()
            processes.append(p)

        # Stops all of them before we begin new ones
        for process in processes:
            process.join()

    # Do all the extra ones (if our number isn't perfectly divisble by num_threads)
    processes = []
    for i in range(extra_times):

        # Grabs midi file
        midi_file = all_midi_files[num_times * num_threads + i]

        # Starts the process
        p = Process(target = process_one, args = (midi_file, noise_path, save))
        p.start()
        processes.append(p)

    # Stops all of them before we begin new ones
    for process in processes:
        process.join()

# Assumes the mel spectrogram .npy files are there, and plots them alongside
# the ground-truth labels generated.
def plot_all(path):
    for folder in tqdm(glob.glob(path + "*/*/")):
        try:
            # Loads the recording length, spectrogram, and labels.
            recording_len_file = open(folder + 'duration.txt')
            recording_len = float(recording_len_file.readlines()[0].rstrip('\n'))
            recording_len_file.close()
            spectrogram = np.load(glob.glob(folder + "*mel*.npy")[0])
            labels = np.load(glob.glob(folder + "*dense*.npy")[0])
            plot_mel_spectrogram(folder + '_mel_spectrogram' + '.png', spectrogram, recording_len, labels, fmax=4096)
        except IndexError:
            tqdm.write('error')
            pass

def plot_all_two(path, id):
    for folder in tqdm(glob.glob(path + "*/*/*/")):
        try:
            if not os.path.isfile(folder + id + '/predictions.npy'):
                print('No predictions yet for this id!')
                return
            recording_len_file = open(folder + 'duration.txt')
            recording_len = float(recording_len_file.readlines()[1].rstrip('\n'))
            recording_len_file.close()
            crop = ThreeSixtyCrop()
            spectrogram = np.load(glob.glob(folder + "*mel*.npy")[0])
            labels = np.load(glob.glob(folder + "*dense*.npy")[0])
            predictions = np.load(glob.glob(folder + id + '/*pred*.npy')[0])
            spectrogram, labels = crop(spectrogram, labels)
            plot_mel_spectrogram(folder + '_mel_spectrogram_gt' + '.png', spectrogram, recording_len, labels, fmax=4096)
            plot_mel_spectrogram(folder + '_mel_spectrogram_pred' + '.png', spectrogram, recording_len, predictions, fmax=4096)
        except IndexError:
            tqdm.write('error')
            pass

# Little helper for converting argparse input into boolean
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# --------------------------------- ACTUAL RUNNING CODE ---------------------------------
parser = argparse.ArgumentParser(description="Processes midi files into wavs (intermediate step; not saved) and then finally into .npy mel spectrograms.")
parser.add_argument("path", type=str, help="The path to the run you're processing (should be a time-stamped folder under /'added/', /'removed/', or /'edited/'.)")
parser.add_argument("--noise_path", type=str, default='noise/Noise_1.wav', help="The full path to the noise file you want to sample from.")
parser.add_argument("--num_threads", type=int, default=1, help="The number of threads you want to use for this. Come on now, Ryan ¯\_(ツ)_/¯")
parser.add_argument("-s", "--save", type=str2bool, nargs='?', const=True, default="False", help="Whether you want to save ALL the .wav files. BE CAREFUL!")
parser.add_argument("-p", "--plot", action='store_true', help='Plot/visualize the mistake labelling from ground truth.')
parser.add_argument("-d", "--pred", type=str2bool, nargs='?', const=True, default="False", help="Whether you want to plot predictions.")
args = parser.parse_args()
# Try and visualize some labels
if args.plot:
    if args.pred:
        id = input('What is the id of the plot? ')
        plot_all_two(args.path, id)
    else:
        plot_all(args.path)
else:
    # Call the actual function
    process_all(args.path, args.noise_path, args.num_threads, args.save)

# ---------------------------------------------------------------------------------------
# Useful for mess-ups. Uncomment to clean up all wavs! Change the .wav extension in the actual declaration to delete other filetypes.
# cleanup("dataset/removed/24-08-2018_at_16:03_with_50_examples/")
