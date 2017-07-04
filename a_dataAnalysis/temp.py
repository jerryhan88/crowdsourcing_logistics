import __init__
from init_project import *
#
import pandas as pd

day_flow = {}


for ft, tt in [(6, 11), (12, 17), (18, 23)]:
    df = pd.read_csv(opath.join(dpath['dayInstance'], 'dayInstance-D%02d-%02d-%02d.csv' % (1, ft, tt)))
    df = df.groupby(['fromDistrict', 'toDistrict']).count().reset_index()
    for fromDistrict, toDistrict, num, _, _ in df.values:
        k = (fromDistrict, toDistrict)
        if not day_flow.has_key(k):
            day_flow[k] = 0
        day_flow[k] += num

count = 0
for (fromDistrict, toDistrict), num in day_flow.iteritems():
    print "['from %s', 'to %s', %d]," % (fromDistrict, toDistrict, num)
    count += num

print count


# df = pd.read_csv(opath.join(dpath['dayInstance'], 'dayInstance-D%02d-%02d-%02d.csv' % (3, 6, 11)))
#
# df = df.groupby(['fromDistrict', 'toDistrict']).count().reset_index()
#
# for fromDistrict, toDistrict, num, _, _ in df.values:
#     print "['from %s', 'to %s', %d]," % (fromDistrict, toDistrict, num)

