import os
import sys

rootPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)

from DVS128_Gait_Day.CRM import CRM_SNN, Config


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


logPath = Config.configs().recordPath
if not os.path.exists(logPath):
    os.makedirs(logPath)
sys.stdout = Logger(logPath + os.sep + "log_DVS128_Gait_Day_CRM_SNN.txt")


def main():
    for i in range(1, 11):
        CRM_SNN.main(i=i, dt=15, T=30, rate_c=0.9)


if __name__ == "__main__":
    main()
