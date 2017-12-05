import os.path as opath
import pickle

from hiBNP.hiTree import HiTree

prob_dir = opath.join('z_files', '_problems')
fn = 'nt05-np12-nb2-tv3-td4.pkl'
prefix = fn[:-len('.pkl')]
ifpath = opath.join(prob_dir, fn)
#
def test():

    with open(ifpath, 'rb') as fp:
        inputs = pickle.load(fp)


    ghLogF, orLogF, cgLogF, bnpLogF = [opath.join('z_files', '%s-%s' % (prefix, fn)) for fn in
                                       ['Hgh.log', 'Hor.log', 'Hcg.log', 'Hbnp.log']]
    ghResF, orResF, cgResF, bnpResF = [opath.join('z_files', '%s-%s' % (prefix, fn)) for fn in
                                       ['Hgh.csv', 'Hor.csv', 'Hcg.csv', 'Hbnp.csv']]
    bptFile = opath.join('z_files', '%s-%s' % (prefix, 'Hbpt.csv'))
    emsgFile, epklFile = 'E.txt', 'E.pkl'
    #
    probSetting = {'problem': inputs,
                   'inclusiveC': [], 'exclusiveC': []}
    grbSetting = {'LogFile': bnpLogF,
                  'Threads': 8}
    etcSetting = {'ghLogF': ghLogF, 'orLogF': orLogF, 'cgLogF': cgLogF, 'bnpLogF': bnpLogF,
                  #
                  'ghResF': ghResF, 'orResF': orResF, 'cgResF': cgResF, 'bnpResF': bnpResF,
                  #
                  'bptFile': bptFile,
                  #
                  'EpklFile': epklFile, 'EmsgFile': emsgFile,
                  }
    bnbTree = HiTree(probSetting, grbSetting, etcSetting)
    bnbTree.startBnP()
    # objV, gap, eliCpuTime, eliWallTime = run(inputs, log_fpath=, pfCst=)


if __name__ == '__main__':
    test()
    # calTime_test()
