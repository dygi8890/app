
#--------------------------------------------------------
#
#
# Author: Spencer Stepanek and team 013_6
# CSCI_3308, Photophonic Web-App Project
# Created on 2/16/21
#
#
# Tones Docs: https://github.com/eriknyquist/tones
# OpenCV Docs: https://docs.opencv.org/master/index.html
#
#
#--------------------------------------------------------


import sys
import numpy as np
import cv2
from tones import SINE_WAVE, SAWTOOTH_WAVE, TRIANGLE_WAVE, SQUARE_WAVE
from tones.mixer import Mixer
import logging
import uuid
import os
from colorlog import ColoredFormatter

# Logging stuff
LOG_LEVEL = logging.DEBUG
LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
log = logging.getLogger('pythonConfig')
log.setLevel(LOG_LEVEL)
log.addHandler(stream)


#-----------< GLOBALS >-----------#


volume = 0.4
duration = 5
notes = ['c','db','d','eb','e','f','gb','g','ab','a','bb','b']

# { "pitchset_name": [ [tonic-pitch, octave], [second-pitch, octave], ... ], ... }
# Pitchsets, when given an offset, can be used to perform their associated emotion from any tonic pitch
# pitchset 1 describes the spread of colors in a given image
primary_pitchsets = {
    "lonely": [ [5,3], [7,3], [9,3], [0,4], [4,5] ],

    # if image has little difference in center of masses
    "powerful": [ [0,3], [7,3], [2,4], [9,4], [4,4] ],
}

# pitchset 2 describes which colors of BGR are most prominent
secondary_pitchsets = { # "bass notes", chosen based on further analysis of the image
    "none": [],  # if image isn't colorful enough
    "virtue": [ [0,2], [0, 3], [2, 3]  ], # if green dominates
    "legacy": [ [0,2], [0, 3], [4, 3] ], # if blue dominates by a measurable margin
    "passion": [ [0,2], [0, 3], [7, 3] ] # if red dominates
    # "legacy" # if orange dominates
}



#----------------< IMAGE PROCESSING DEFINITIONS >----------------#



# params: ( image pixel matrix , channel to keep )
# NOTE: destroys other channels
# returns pixel matrix of a single color
def __STRIP_CHANNELS__(mat, chan):
    ret_mat, row = [], []  # to store/build stripped mat
    for y in range(len(mat)):
        row = []
        for x in range(len(mat[y])):
            row.append(mat[y][x][chan])
        ret_mat.append(row)
    return np.array(ret_mat)


# params: ( image pixel matrix , channel to keep )
# NOTES: retains other empty channels in list as 0's
# returns pixel matrix of a single color
def __STRIP_INPLACE__(mat, chan):
    ret = mat.copy()
    if chan == 0:
        ret[:, :, 1] = 0
        ret[:, :, 2] = 0
    if chan == 1:
        ret[:, :, 0] = 0
        ret[:, :, 2] = 0
    if chan == 2:
        ret[:, :, 0] = 0
        ret[:, :, 1] = 0
    return ret


def image_colorfulness(image):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


# returns ratio of color for each channel
def getColorAmount(mat):
    rats = [0,0,0]
    w, h = __getImageDimensions__(mat)
    for row in mat:
        for p in row:
            for chanIndex in range(3):
                rats[chanIndex] += p[chanIndex]
    for ratIndex in range(3):
        rats[ratIndex] = rats[ratIndex]/(255 * w * h)
    return rats


# params: ( image pixel matrix )
# returns: color channels center of mass
def __COM__(full_mat):
    w, h =__getImageDimensions__(full_mat)
    channels_COM = [] # to store return string
    # strip color channels from each other
    channels = [__STRIP_CHANNELS__(full_mat,0),__STRIP_CHANNELS__(full_mat,1),__STRIP_CHANNELS__(full_mat,2)]

    for channel in range(3):
        # log.debug("Performing center of mass function on color channel {}...".format(channel))
        mat = channels[channel]
        # calculate Xbar, Ybar
        sum_over_columns = [sum(x) for x in zip(*mat)]
        sum_over_rows = [sum(x) for x in mat]
        x_weights, x_masses, y_weights, y_masses = [],[],[],[]
        for i in range(w):
            x_weights.append( i * sum_over_columns[i] ) # mass times position
            x_masses.append( sum_over_columns[i]  ) # mass
        x_bar = sum( x_weights ) / sum( x_masses )

        for i in range(h):
            y_weights.append( i * sum_over_rows[i] ) # mass times position
            y_masses.append( sum_over_rows[i]  ) # mass
        y_bar = sum( y_weights ) / sum( y_masses )

        # print("x_bar: {:.4f}\ny_bar: {:.4f}".format(x_bar,y_bar))
        channels_COM.append( [ int(round(x_bar)), int(round(y_bar)) ] )

    return channels_COM


# params: ( image pixel matrix )
# returns ( width, height )
def __getImageDimensions__(mat):
    w = len(mat[0])
    h = len(mat)
    return (w, h)


# General-use funtion for determining if something is an image
def isImage(filename):
    print(filename)
    mat = cv2.imread("project/static/" + filename)
    if mat is None:
        sys.exit("Must be an image.")
    else:
        return mat


# provides details on individual color means and ranges in a list of values
def analyze_image(image):
    retStats = [] # return stats
    blue = []
    green = []
    red = []
    count = 0
    pixel_sums = []

    for row in image:
        for pixel in row:
            count += 1
            blue.append(pixel[0])
            green.append(pixel[1])
            red.append(pixel[2])
            pixel_sums.append( ( (pixel[0]/3) + (pixel[1]/3) + (pixel[2]/3) ) )

    for color in [ [blue, "blue"], [green, "green"], [red, "red"] ]:
        mean = np.mean(color[0])
        val_max = np.max(color[0])
        val_min = np.min(color[0])
        val_range = val_max - val_min
        retStats.append([color[1], mean, val_range])

    # avg_color = [np.sum(blue) / count, np.sum(green) / count, np.sum(red) / count]
    mean = np.mean(image)
    val_max = np.max(image)
    val_min = np.min(image)
    val_range = val_max - val_min
    retStats.append([ 'shade', mean, val_range ])
    return retStats


#----------------< AUDIO GENERATION DEFINITIONS >----------------#


def initColorSynth(mixer, iden, analysis, color, synth, attack=1, decay=1, v_f=0, v_v=0, octave=4, pitch_1='c' ):

    mixer.create_track(iden, synth, vibrato_frequency=v_f, vibrato_variance=v_v, attack=attack, decay=decay)

    presence = analysis[1]/255  # calculates the presence of each color in an image
                                # (normalize the mean to 0-1 to stuff into amplitude)

    mixer.add_note(iden, note=pitch_1, octave=octave, duration=duration, amplitude=presence)



#----------------< FRONT-END CALLS >----------------#



def colorMark(mat):

    log.info("Analyzing image...")
    circles = __COM__(mat)
    rats = getColorAmount(mat)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    w, h = __getImageDimensions__(mat)
    for chan in range(3):
        mat = cv2.circle(
            mat, (circles[chan][0], circles[chan][1]),
            int(w * rats[chan] * 0.25),
            colors[chan],
            int(w / 150))

    COMs = [] # convert COMs into measurements by image width
    for chan in circles:
        COMs.append( [chan[0]/w, chan[1]/w] )

    return mat, rats, COMs

def makeUUID(f, uploadPath):
    file_path = os.path.join(uploadPath, f.filename)
    f.save(file_path)
    image_mat = cv2.imread(file_path)
    image_id = str(uuid.uuid4())
    imgName = image_id + '.' + "jpg"
    idFilename = 'project/static/uuids/' + imgName
    cv2.imwrite(idFilename, image_mat)
    os.remove(file_path)
    writeAudio(image_id, imgName, uploadPath)
    return image_id, image_mat

def writeAudio(imageID, filename, path):
    img = [imageID, filename]
    file_path = os.path.join( path, filename )
    img.append( cv2.imread(file_path) )

    try:
        img[2][0][0]  # test to get first pixel value
    except TypeError:
        log.error("Inputted path ({}) must be an image".format(filename))
        log.error("Check file names and extensions in image_pool to ensure they match the image (explicitly write out the file extension).")
        sys.exit()

    width, height = __getImageDimensions__(img[2])
    if (width > 1500) or (height > 1500):  # if image is larger than 800 pixels in either dimension
        factor = 0.5  # percent of original size
        width = int(img[2].shape[1] * factor)
        height = int(img[2].shape[0] * factor)
        dimensions = (width, height)

        log.warning("Resized image {} ({}, {}) for quicker analysis".format(img[1], width, height))
        img[2] = cv2.resize(img[2], dimensions, interpolation=cv2.INTER_AREA)
        cv2.imwrite('project/static/uuids/' + img[1], img[2])

    # log.info("Writing audio: {}/{}.wav".format(path, imageID))
    img[2], color_ratios, COMs = colorMark(img[2])  # mark it up yo

    # log.debug("Color ratios: %s", str(color_ratios))
    # log.debug("Center of masses: %s", str(COMs))

    avgDist = 0
    ds = []
    for COM_index in range(len(COMs)):
        # distance between any one to the others is larger than x
        i1 = COM_index % 3
        i2 = (COM_index + 1) % 3

        xs = (COMs[i1][0] - COMs[i2][0]) ** 2
        ys = (COMs[i1][1] - COMs[i2][1]) ** 2
        d = (xs + ys) ** (1 / 2)
        avgDist += d
        ds.append(d)

    avgDist = round(avgDist / 3, 15)
    # log.debug("Average distance between COMs: %s", avgDist)
    outlierCOMs = 0
    for distIndex in range(len(ds)):
        if (ds[distIndex] > avgDist * 1.5):
            outlierCOMs += 1
            # log.debug("Upper outlier COM %s in channel %s detected", str(ds[distIndex]), str(distIndex))
        if (ds[distIndex] < avgDist * 0.5):
            outlierCOMs += 1
            # log.debug("Lower outlier COM %s in channel %s detected", str(ds[distIndex]), str(distIndex))

    if outlierCOMs == 1: primary_pitchset_choice = "lonely" # complexity in this decision leaves much to be desired
    else: primary_pitchset_choice = "powerful"

    log.debug("Primary pitch-set \"%s\" selected", primary_pitchset_choice)

    colorfulness = image_colorfulness(img[2])
    # log.debug("Colorfulness is : %s", str(colorfulness))

    if colorfulness < 50: # determine if image is colorful enough to deserve a second pitchset (bass notes)
        secondary_pitchset_choice = "none"
    else: # determine most prominent color for secondary_pitchset selection

        # log.debug("Max outlier found for 2nd_pitchset in channel %s", str(maxOutlier))
        maxRatIndex = color_ratios.index(max(color_ratios))

        if maxRatIndex == 0:
            secondary_pitchset_choice = "virtue"
        elif maxRatIndex == 1:
            secondary_pitchset_choice = "legacy"
        elif maxRatIndex == 2:
            secondary_pitchset_choice = "passion"
        else:
            secondary_pitchset_choice = "none"

    log.debug("Secondary pitch-set \"%s\" selected", secondary_pitchset_choice)

    log.info("Writing audio...")

    mixer = Mixer(44100, volume)
    pitch_offset = 0  # change to edit key of playback
    saw_pitches = round(colorfulness/10) - 4 # measure of how many sawtooth synths to use on playback
    # log.debug("Sawtooth pitches to be created: %s", str(colorful_pitches+1))
    instrument = SAWTOOTH_WAVE

    ampBase = 0.5
    for pitch_id in range(len(primary_pitchsets[primary_pitchset_choice])):
        pitch_content = primary_pitchsets[primary_pitchset_choice][pitch_id]
        played_pitch = notes[(pitch_content[0] + pitch_offset) % 12]
        if pitch_id > saw_pitches:
            instrument = SINE_WAVE
            ampBase = 0.6
        octave = pitch_content[1]
        if octave > 4: # lower volume on higher notes
            amp = ampBase + 0.2
        else:
            amp = ampBase + 0.3
        mixer.create_track(pitch_id, instrument, vibrato_frequency=0, vibrato_variance=0, attack=1, decay=1)
        mixer.add_note(pitch_id, note=played_pitch, octave=octave, duration=duration, amplitude=amp)

    pitch_iter_offset = len(primary_pitchsets[primary_pitchset_choice]) # calculate pitch_id offset

    amp = 0.8
    for pitch_id in range(len(secondary_pitchsets[secondary_pitchset_choice])):
        pitch_content = secondary_pitchsets[secondary_pitchset_choice][pitch_id]
        played_pitch = notes[(pitch_content[0] + pitch_offset) % 12]
        octave=pitch_content[1]
        mixer.create_track(pitch_id+pitch_iter_offset, SINE_WAVE, vibrato_frequency=0, vibrato_variance=0, attack=1, decay=1)
        mixer.add_note(pitch_id+pitch_iter_offset, note=played_pitch, octave=octave, duration=duration, amplitude=amp)

    mixer.write_wav(path + '/' + imageID + '.wav')
    log.info("Audio written...")

    return imageID + '.wav'