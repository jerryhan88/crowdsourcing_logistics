from functools import reduce
import os.path as opath
import os

DATA_HOME = reduce(lambda prefix, name: opath.join(prefix, name),
                   [opath.join(opath.dirname(opath.realpath(__file__))),
                    '..', 'crowdsourcing_data'])
dpath = {}
dpath['raw'] = opath.join(DATA_HOME, 'raw')
dpath['geo'] = opath.join(DATA_HOME, 'geo')
# --------------------------------------------------------------
dpath['home'] = opath.join(DATA_HOME, 'crowdsourcingLogistics')
dpath['experiment'] = opath.join(dpath['home'], 'experiment')
dpath['flow'] = opath.join(dpath['home'], 'flow')


for dn in ['home', 'experiment', 'flow']:
    try:
        if not opath.exists(dpath[dn]):
            os.makedirs(dpath[dn])
    except OSError:
        pass