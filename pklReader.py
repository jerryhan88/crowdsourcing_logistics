import pickle


def read_pkl(ifpath):
    with open(ifpath, 'rb') as fp:
        problem = pickle.load(fp)
    return problem