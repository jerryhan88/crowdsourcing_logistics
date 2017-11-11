from bnpTree import BnPTree


ifpath = 'nt05-np12-nb2-tv3-td5.pkl'
#
def test():
    import pickle
    with open(ifpath, 'rb') as fp:
        inputs = pickle.load(fp)

    ghLogF, orLogF, cgLogF, bnpLogF = 'gh.log', 'or.log', 'cg.log', 'bnp.log'
    ghResF, orResF, cgResF, bnpResF = 'gh.csv', 'or.csv', 'cg.csv', 'bnp.csv'
    bptFile = 'bpt.csv'
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
    bnbTree = BnPTree(probSetting, grbSetting, etcSetting)
    bnbTree.startBnP()
    # objV, gap, eliCpuTime, eliWallTime = run(inputs, log_fpath=, pfCst=)


if __name__ == '__main__':
    test()
    # calTime_test()
