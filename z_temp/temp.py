import csv

ifpath = 'nt08-np20-nb4-tv3-td7-colGenMM.txt'
ofpath = 'nt08-np20-nb4-tv3-td7-colGenMM.csv'



prefixs = [ 'nt08-np20-nb4-tv3-td7-colGenMM',
            'nt08-np42-nb3-tv4-td9-colGenMM',
            'nt10-np20-nb4-tv3-td7-colGenMM',
            'nt14-np12-nb3-tv7-td6-colGenMM',
            'nt20-np12-nb5-tv6-td6-colGenMM'
            ]

for pf in prefixs:
    print(pf)
    ifpath = '%s.txt' % pf
    ofpath = '%s.csv' % pf

    with open(ifpath, 'r') as f:
        for _ in range(3):
            f.readline()
        # print(f.readline())
        col = f.readline()
        Nodes, CurrentNode, ObjectiveBounds, Work = col.split('|')
        Expl, Unexpl = Nodes.split()
        Obj, Depth, IntInf = CurrentNode.split()
        Incumbent, BestBd, Gap = ObjectiveBounds.split()
        ItNode, Time = Work.split()
        #
        f.readline()
        with open(ofpath, 'wt') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            header = [Expl, Unexpl,
                      Obj, Depth, IntInf,
                      Incumbent, BestBd, Gap,
                      ItNode, Time]
            writer.writerow(header)
            while True:
                row = f.readline().split()
                if not row:
                    break
                try:
                    if row[0] == 'H':
                        new_row = row[1:3]
                        new_row += [None, None, None]
                        new_row += row[3:5]
                        new_row += [row[5][:-1]]
                        new_row += [row[6]]
                        new_row += [row[7][:-1]]
                    elif row[2] == 'cutoff' or row[2] == 'infeasible':
                        new_row = row[:4]
                        new_row += [None]
                        new_row += row[4:6]
                        new_row += [row[6][:-1]]
                        new_row += [row[7]]
                        new_row += [row[8][:-1]]
                    else:
                        new_row = row[:7]
                        new_row += [row[7][:-1]]
                        new_row += [row[8]]
                        new_row += [row[9][:-1]]
                    writer.writerow(new_row)
                except:
                    continue
