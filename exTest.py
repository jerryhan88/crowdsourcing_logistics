from hiTree import HiTree
import os.path as opath
import pickle
from exactMM import run as exactMM_run

ifpath = 'nt05-np12-nb2-tv3-td4.pkl'

#
def test():

    with open(ifpath, 'rb') as fp:
        inputs = pickle.load(fp)

    prefix = ifpath[:-len('.pkl')]
    exLogF = opath.join('z_files', '%s-%s.log' % (prefix, 'ex'))
    exResF = opath.join('z_files', '%s-%s.csv' % (prefix, 'ex'))
    #
    probSetting = {'problem': inputs}
    grbSetting = {'LogFile': exLogF,
                  'Threads': 8,}
    etcSetting = {'exLogF': exLogF,
                  'exResF': exResF
                  }

    exactMM_run(probSetting, grbSetting, etcSetting)



if __name__ == '__main__':
    test()
    # calTime_test()
