from init_project import *
#
from datetime import datetime
#
from _utils.recording import record_log


def run(problem, log_fpath=None):
    travel_time, tasks, paths, detour_th, volume_th, num_bundles = convert_input4greedyHeuristic(*problem)
    #
    def insert_task(b, t1, path_insertion_estimation=None):
        b.tasks[t1.tid] = t1
        path_ws = 0
        if path_insertion_estimation == None:
            seq = ['p0%d' % t1.tid, 'd%d' % t1.tid]
            for p in paths:
                b.path_pd_seq[p] = seq[:]
                detour = travel_time[p.ori, t1.pp] + \
                         travel_time[t1.pp, t1.dp] + \
                         travel_time[t1.dp, p.dest]
                if detour > detour_th:
                    b.path_detour[p] = 1e400
                else:
                    b.path_detour[p] = detour
                    path_ws += p.w
        else:
            for p, (additional_detour, i0, j0) in path_insertion_estimation.items():
                if b.path_detour[p] + additional_detour < detour_th:
                    path_ws += p.w
                    b.path_detour[p] += additional_detour
                    b.path_pd_seq[p].insert(i0, 'p0%d' % t1.tid)
                    b.path_pd_seq[p].insert(j0 + 1, 'd%d' % t1.tid)
                else:
                    b.path_detour[p] = 1e400
        b.bundle_attr = sum(t.r for t in b.tasks.values()) * path_ws
    #
    def estimate_bundle_attr(b, t1):
        if volume_th < sum(t0.v for t0 in b.tasks.values()) + t1.v:
            return 0, None
        else:
            task_path_insertion_estimation = {}
            for p in paths:
                pd_seq = b.path_pd_seq[p]
                least_addditional_detour, i0, j0 = 1e400, None, None
                for i in range(len(pd_seq)):
                    if i == len(pd_seq) - 1:
                        j = i
                        p0 = get_point(b, pd_seq[i])
                        p1 = p.dest
                        additional_detour = travel_time[p0, t1.pp] + \
                                            travel_time[t1.pp, t1.dp] + \
                                            travel_time[t1.dp, p1]
                        additional_detour -= travel_time[p0, p1]
                        if additional_detour < least_addditional_detour:
                            least_addditional_detour, i0, j0 = additional_detour, i, j
                    else:
                        for j in range(i, len(pd_seq)):
                            if i == j:
                                #
                                # task t1's pick-up is just before t1's delivery
                                #
                                if i == 0:
                                    p0 = p.ori
                                    p1 = get_point(b, pd_seq[i])
                                else:
                                    p0 = get_point(b, pd_seq[i])
                                    p1 = get_point(b, pd_seq[i + 1])
                                additional_detour = travel_time[p0, t1.pp] + \
                                                    travel_time[t1.pp, t1.dp] + \
                                                    travel_time[t1.dp, p1]
                                additional_detour -= travel_time[p0, p1]
                            else:
                                #
                                # task t1's pick-up is not just before t1's delivery
                                #
                                if i == 0:
                                    #
                                    # Insert a new pick-up task before the first pick-up task
                                    #
                                    p0 = p.ori
                                    p1 = get_point(b, pd_seq[i])
                                else:
                                    p0 = get_point(b, pd_seq[i])
                                    p1 = get_point(b, pd_seq[i + 1])
                                #
                                if j == len(pd_seq) - 1:
                                    #
                                    # Insert a new delivery task after the last delivery task
                                    #
                                    p2 = get_point(b, pd_seq[j])
                                    p3 = p.dest
                                else:
                                    p2 = get_point(b, pd_seq[j])
                                    p3 = get_point(b, pd_seq[j + 1])
                                #
                                additional_detour = travel_time[p0, t1.pp] + \
                                                    travel_time[t1.pp, p1] + \
                                                    travel_time[p2, t1.dp] + \
                                                    travel_time[t1.dp, p3]
                                additional_detour -= travel_time[p0, p1]
                                additional_detour -= travel_time[p2, p3]
                            if additional_detour < least_addditional_detour:
                                least_addditional_detour, i0, j0 = additional_detour, i, j
                assert least_addditional_detour != 1e400, (b, p, pd_seq)
                task_path_insertion_estimation[p] = (least_addditional_detour, i0, j0)
            #
            path_ws = 0
            for p, (additional_detour, _, _) in task_path_insertion_estimation.items():
                if b.path_detour[p] + additional_detour < detour_th:
                    path_ws += p.w
            bundle_attr = (sum(t.r for t in b.tasks.values()) + t1.r) * path_ws
            return bundle_attr, task_path_insertion_estimation
    #
    def get_point(b, pd_name):
        t = b.tasks[int(pd_name[len('p0'):])]
        if pd_name.startswith('p0'):
            return t.pp
        else:
            assert pd_name.startswith('d')
            return t.dp
    #
    startTime = datetime.now()
    #
    # initialize variable to accumulate task attractiveness score
    #
    for t in tasks:
        task_att = 0
        for p in paths:
            detour_time = travel_time[p.ori, t.pp] + \
                       travel_time[t.pp, t.dp] + \
                       travel_time[t.dp, p.dest]
            #
            # compute detour percentage (add small detour to avoid division by zero)
            #
            detour_dist = detour_time + 0.000001
            detour_percent = detour_dist / travel_time[p.ori, p.dest]
            #
            # compute t_p_att and add to task_att
            #
            task_att += (t.r * p.w) / detour_percent
        t.set_attr(task_att)
    tasks.sort(key=lambda t: t.attr, reverse=True)
    #
    # start bundling
    #
    bundles = [bundle(bid, paths) for bid in range(num_bundles)]
    #
    candi_bundles = bundles
    while candi_bundles:
        new_candi_bundles = []
        for i, b in enumerate(candi_bundles):
            if not b.tasks:
                for i, t0 in enumerate(tasks):
                    path_ws = 0
                    seq = ['p0%d' % t0.tid, 'd%d' % t0.tid]
                    for p in paths:
                        b.path_pd_seq[p] = seq[:]
                        detour = travel_time[p.ori, t0.pp] + \
                                 travel_time[t0.pp, t0.dp] + \
                                 travel_time[t0.dp, p.dest]
                        if detour < detour_th:
                            path_ws += p.w
                    if path_ws != 0:
                        best_task = tasks.pop(i)
                        insert_task(b, best_task)
                        assert b.bundle_attr != 0
                        new_candi_bundles += [b]
                        break
            else:
                max_attr_bun, best_task, best_task_path_insertion = 0, None, None
                for t in tasks:
                    bundle_attr, task_path_insertion_estimation = estimate_bundle_attr(b, t)
                    if bundle_attr == 0:
                        continue
                    if max_attr_bun < bundle_attr:
                        max_attr_bun, best_task, best_task_path_insertion = bundle_attr, t, task_path_insertion_estimation
                if max_attr_bun != 0:
                    tasks.pop(tasks.index(best_task))
                    insert_task(b, best_task, best_task_path_insertion)
                    new_candi_bundles += [b]
        candi_bundles = new_candi_bundles
    unassigned_tasks = tasks
    #
    endTime = datetime.now()
    eliTime = (endTime - startTime).seconds
    #
    logContents = '\n\n'
    logContents += 'Summary\n'
    logContents += '\t Sta.Time: %s\n' % str(startTime)
    logContents += '\t End.Time: %s\n' % str(endTime)
    logContents += '\t Eli.Time: %d\n' % eliTime
    logContents += '\t ObjV: %.3f\n' % sum(b.bundle_attr for b in bundles)
    logContents += '\t chosen B.: %s\n' % str(bundles)
    record_log(log_fpath, logContents)
    return sum(b.bundle_attr for b in bundles), eliTime, [[t.tid for t in b.tasks.values()] for b in bundles]


class task(object):
    def __init__(self, tid, pp, dp, v, r):
        self.tid = tid
        self.pp, self.dp = pp, dp
        self.v, self.r = v, r

    def set_attr(self, attr):
        self.attr = attr

    def __repr__(self):
        return 't%d(%s->%s;%.03f)' % (self.tid, self.pp, self.dp, self.r)

class path(object):
    def __init__(self, ori, dest, w=None):
        self.ori, self.dest, self.w = ori, dest, w

    def __repr__(self):
        if self.w != None:
            return '%d->%d;%.03f' % (self.ori, self.dest, self.w)
        else:
            return '%d->%d' % (self.ori, self.dest)


class bundle(object):
    def __init__(self, bid, paths):
        self.bid = bid
        #
        self.tasks = {}
        self.path_pd_seq, self.path_detour = {}, {}
        for p in paths:
            self.path_pd_seq[p] = []
            self.path_detour[p] = 0
        self.bundle_attr = 0

    def __repr__(self):
        return 'b%d(ts:%s)' % (self.bid, ','.join(['t%d' % t.tid for t in self.tasks.values()]))

def convert_input4greedyHeuristic(travel_time,
                                  flows, paths,
                                  tasks, rewards, volumes,
                                  num_bundles, volume_th, detour_th):
    #
    # Convert inputs for the greedy heuristic
    #
    tasks = [task(i, pp, dd, volumes[i], rewards[i]) for i, (pp, dd) in enumerate(tasks)]
    total_flows = sum(flows[i][j] for i in range(len(flows)) for j in range(len(flows)))
    paths = [path(ori, dest, flows[ori][dest] / float(total_flows)) for ori, dest in paths]
    #
    return travel_time, tasks, paths, detour_th, volume_th, num_bundles



if __name__ == '__main__':
    from problems import *
    print(run(convert_input4greedyHeuristic(*ex2())))


