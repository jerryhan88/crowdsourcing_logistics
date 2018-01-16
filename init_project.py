import os.path as opath
import os
dpath = {}
taxi_data_home = opath.join(opath.join(opath.dirname(opath.realpath(__file__)), '..'), 'crowdsourcing_data')
dpath['raw'] = opath.join(taxi_data_home, 'raw')
dpath['geo'] = opath.join(taxi_data_home, 'geo')
# --------------------------------------------------------------
dpath['home'] = opath.join(taxi_data_home, 'crowdsourcingLogistics')
dpath['experiment'] = opath.join(dpath['home'], 'experiment')
dpath['flow'] = opath.join(dpath['home'], 'flow')

# dpath['dayInstance'] = opath.join(dpath['home'], 'dayInstance')


for dn in ['home', 'experiment', 'flow']:
    try:
        if not opath.exists(dpath[dn]):
            os.makedirs(dpath[dn])
    except OSError:
        pass