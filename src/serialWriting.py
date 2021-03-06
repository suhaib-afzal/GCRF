import serial
import time
import struct

import copy
import numpy as np

from EEGSignalGenerator import EEGSignalGenerator

from fNIRSSignalGenerator import fNIRSSignalGenerator

from src.EEGSignalGenerator import plotSyntheticEEG

def main():
    iD_Run = 1
    print(f'Run ID:       \t\t {iD_Run:03d}')
    print("")

    samplingTime = 200   # time in secs
    print("Sampling time:\t\t", samplingTime, "secs")
    print("")

    print("EEG information")
    print("--------------------------")
    sgEEG = EEGSignalGenerator()
    sgEEG.samplingRate = 128
    sgEEG.nSamples = int(sgEEG.samplingRate * samplingTime)
    sgEEG.nChannels = 6
    print("Sampling rate:\t\t",    sgEEG.samplingRate, "Hz")
    print("Number of samples:\t",  sgEEG.nSamples)
    print("Number of channels:\t", sgEEG.nChannels)
    #sgEEG.execute()
    # print(sg.data)
    #plotSyntheticEEG(sgEEG.data)
    print("")

    input("Press Enter to continue...")
    print("")
    print("")

    print("fNIRS information")
    print("--------------------------")
    sgfNIRS = fNIRSSignalGenerator()
    sgfNIRS.samplingRate = 10
    sgfNIRS.nSamples = int(sgfNIRS.samplingRate * samplingTime)
    sgfNIRS.nChannels = 4
    print("Sampling rate:\t\t",    sgfNIRS.samplingRate, "Hz")
    print("Number of samples:\t",  sgfNIRS.nSamples)
    print("Number of channels:\t", sgfNIRS.nChannels)
    #sgfNIRS.execute()
    print("")
    print("")

    input("Press Enter to continue...")
    print("")
    print("")

    # Read the input file that contains the EEG synthetic data
    fileName = "synthetic{:03d}_EEG_{}Chs_{:.0f}Hz_{}secs.txt"
    print("File openning: ", fileName.format(iD_Run, sgEEG.nChannels, sgEEG.samplingRate, samplingTime))
    print("-------------")
    print("")
    fileIn = open(fileName.format(iD_Run, sgEEG.nChannels, sgEEG.samplingRate, samplingTime), "r")

    print("File header:")
    for i in range(6):
        print(fileIn.readline(), end='')
    print("")

    if sgEEG.nSamples < sgfNIRS.nSamples:
        sgEEG.nSamples = sgfNIRS.nSamples

    input("Press Enter to continue...")
    print("")
    print("")

    print("File data:")
    newData = np.zeros((sgEEG.nSamples, sgEEG.nChannels, 1),dtype=float)
    for i in range(sgEEG.nSamples):
        line = fileIn.readline()
        line_list = line.split(", ")
        newData[i, :, 0] = np.array(list( map(float, line_list) ))

    fileIn.close()

    sgEEG.data = copy.deepcopy(newData)
    for i in range(sgEEG.nSamples):
        print(sgEEG.data[i, :, 0])
    print("")
    print("")


    # Read the input file that contains the fNIRs synthetic data
    fileName = "synthetic{:03d}_fNIRs_{}Chs_{:.0f}Hz_{}secs.txt"
    print("File openning: ", fileName.format(iD_Run, sgfNIRS.nChannels, sgfNIRS.samplingRate, samplingTime))
    print("-------------")
    print("")
    fileIn = open(fileName.format(iD_Run, sgfNIRS.nChannels, sgfNIRS.samplingRate, samplingTime), "r")

    print("File header:")
    for i in range(6):
        print(fileIn.readline(), end='')
    print("")

    if sgfNIRS.nSamples < sgEEG.nSamples:
        sgfNIRS.nSamples = sgEEG.nSamples

    input("Press Enter to continue...")
    print("")
    print("")

    print("File data:")
    newData = np.zeros((sgfNIRS.nSamples, sgfNIRS.nChannels, 2), dtype=float)
    for i in range(sgfNIRS.nSamples):
        for j in range(2):
            line = fileIn.readline()
            line_list = line.split(", ")
            newData[i, :, j] = np.array(list( map(float, line_list) ))

    fileIn.close()

    sgfNIRS.data = copy.deepcopy(newData)
    for i in range(sgfNIRS.nSamples):
        for j in range(2):
            print(sgfNIRS.data[i, :, j])
    print("")
    print("")


    #initialization and open the port

    #possible timeout values:
    #    1. None: wait forever, block call
    #    2. 0: non-blocking mode, return immediately
    #    3. x, x is bigger than 0, float allowed, timeout block call

    ser = serial.Serial()
    #ser.port = "/dev/ttyUSB0"
    #ser.port = "/dev/ttyUSB7"
    #ser.port = "/dev/ttyS2"
    ser.port = "COM1"
    #ser.port = "/Device/USBPDO-3"      #In windows go to Device Manager and look for USB controllers
    ser.baudrate = 9600                 #How fast your COM port operates.
    ser.bytesize = serial.EIGHTBITS     #Number of bits per bytes
    ser.parity = serial.PARITY_NONE     #These are used for error correction but are not normally used. Set parity check: no parity
    ser.stopbits = serial.STOPBITS_ONE  #number of stop bits
    #ser.timeout = None                 #Used to prevent the serial port from hanging. block read
    ser.timeout = 1                     #Used to prevent the serial port from hanging. non-block read
    #ser.timeout = 2                    #Used to prevent the serial port from hanging. timeout block read
    ser.xonxoff = False                 #disable software flow control
    ser.rtscts = False                  #disable hardware (RTS/CTS) flow control
    ser.dsrdtr = False                  #disable hardware (DSR/DTR) flow control
    ser.writeTimeout = 2                #timeout for write

    try:
        ser.open()
    except Exception as e:
        print("error open serial port: " + str(e))
        exit()

    if ser.isOpen():

        try:
            ser.flushInput()   #flush input buffer, discarding all its contents
            ser.flushOutput()  #flush output buffer, aborting current output
                               # and discard all that is in buffer

            # Send EEG data
            print("Send EEG data:")
            input("Press Enter to continue...")
            numOfWrongSending = 0
            for i in range(sgEEG.nSamples):
                for j in range(sgEEG.nChannels):
                    my_float = sgEEG.data[i, j, 0]
                    my_data = struct.pack('f', my_float)
                    #print(my_data, "Bytes of my data")

                    n = ser.write(my_data)
                    #print(n, "Bytes successfully written")
                    if n < 4:
                        print(4 - n, "Bytes no successfully written")
                        numOfWrongSending = numOfWrongSending + 1

                    time.sleep(0.0000005)  #give the serial port sometime to receive the data

            print(f'Number of EEG data sent with loss of information: {numOfWrongSending}')
            input("Press Enter to continue...")
            print("")
            print("")

            numOfData = 0
            while True:
                #response = ser.readline()
                #print("read data: " + str(response))
                response = ser.read()
                print(f'read data: {response}')
                print(repr(response))
                #print(response.encode('hex'))

                numOfData = numOfData + 1

                if numOfData >= sgEEG.nSamples * sgEEG.nChannels:
                    break

            print(f'Number of EEG data: {numOfData}')
            print("")
            print("")

            # Send fNIRs data
            print("Send fNIRs data:")
            input("Press Enter to continue...")
            numOfWrongSending = 0
            for i in range(sgfNIRS.nSamples):
                for k in range(2):
                    for j in range(sgfNIRS.nChannels):
                        my_float = sgfNIRS.data[i, j, k]
                        my_data = struct.pack('f', my_float)
                        #print(my_data, "Bytes of my data")

                        n = ser.write(my_data)
                        #print(n, "Bytes successfully written")
                        if n < 4:
                            print(4-n, "Bytes no successfully written")
                            numOfWrongSending = numOfWrongSending + 1

                        time.sleep(0.0000005)  # give the serial port sometime to receive the data
            print(f'Number of fNIRs data sent with loss of information: {numOfWrongSending}')
            input("Press Enter to continue...")
            print("")
            print("")

            numOfData = 0
            while True:
                #response = ser.readline()
                #print("read data: " + str(response))
                response = ser.read()
                print("read data: ", response)
                print(repr(response))
                #print(response.encode('hex'))

                numOfData = numOfData + 1

                if numOfData >= sgfNIRS.nSamples * sgfNIRS.nChannels:
                    break

            print(f'Number of fNIRs data: {numOfData}')
            print("")
            print("")

            ser.close()
        except Exception as e1:
            print("error communicating...: " + str(e1))

    else:
        print("cannot open serial port ")

# end main()


if __name__ == '__main__':
    main()
