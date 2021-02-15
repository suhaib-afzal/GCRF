
"""Multi-channel time series streamer (LSL)."""
import sys
import getopt

import time
from random import random as rand

from pylsl import StreamInfo, StreamOutlet, local_clock   # import dependencies from LSL module


def main(argv):
    srate = 4096
    name = 'BioSemi'
    type = 'g.tec'
    n_channels = 32
    help_string = 'SendData.py -s <sampling_rate> -n <stream_name> -t <stream_type>'
    try:
        opts, args = getopt.getopt(argv, "hs:c:n:t:", longopts=["srate=", "channels=", "name=", "type"])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_string)|
            sys.exit()
        elif opt in ("-s", "--srate"):
            srate = float(arg)
        elif opt in ("-c", "--channels"):
            n_channels = int(arg)
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("-t", "--type"):
            type = arg

    # first create a new stream info (here we set the name to BioSemi,
    # the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
    # last value would be the serial number of the device or some other more or
    # less locally unique identifier for the stream as far as available (you
    # could also omit it but interrupted connections wouldn't auto-recover)
    info = StreamInfo(name, type, n_channels, srate, 'double64', 'myuid34234')

    # next make an outlet
    outlet = StreamOutlet(info)

    f = open("random_EEG_500_test.txt")  # open file with data
    lines = f.readlines()  # read lines from files into a list
    # print(lines[0])
    # print(len(lines)-30)
    for i in range(len(lines)):  # prepare lines to be sended
        l = lines[i]  # read ech line
        l = l.replace("\n", "")  # erase new line comand
        l = l.replace("]", "")  # erase open bracket
        l = l.replace("[", "")  # erase close bracket
        as_list = l.split(",")  # split line on comas
        lines[i] = (list(map(float, as_list)))  # push the item to list

    # Start the stream
    print("now sending data...")
    start_time = local_clock()
    sent_samples = 0
    t=time.perf_counter()
    while sent_samples <= 30000:  # establish number of sample to be sent
        elapsed_time = local_clock() - start_time    # set time
        required_samples = int(srate * elapsed_time) - sent_samples   # calculate samples needed for the time
        for sample_ix in range(required_samples):

            outlet.push_sample(lines[sent_samples+(required_samples-1)])  # sending samples
        sent_samples += required_samples
        # now send it and wait for a bit before trying again.
        #print(sent_samples)
        #time.sleep(0.0001)
    #print(sent_samples)
    #print(time.perf_counter()-t)


if __name__ == '__main__':
    main(sys.argv[1:])