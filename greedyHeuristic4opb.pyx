from problems import *

def run(problem, k):
    travel_time, tasks, paths, detour_th, volume_th, num_bundles = convert_input4greedyHeuristic(*problem)
    #
    def insert_task(b, t1, estimation=None):
        b.tasks[t1.tid] = t1
        if estimation is None:
            b.path_pd_seq = ['p0%d' % t1.tid, 'd%d' % t1.tid]
            detour = travel_time[b.p.ori, t1.pp] + \
                     travel_time[t1.pp, t1.dp] + \
                     travel_time[t1.dp, b.p.dest]
            detour -= travel_time[b.p.ori, b.p.dest]
            if detour > detour_th:
                b.path_detour = 1e400
            else:
                b.path_detour = detour
        else:
            additional_detour, i0, j0 = estimation
            if b.path_detour + additional_detour <= detour_th:
                b.path_detour += additional_detour
                b.path_pd_seq.insert(i0, 'p0%d' % t1.tid)
                b.path_pd_seq.insert(j0 + 1, 'd%d' % t1.tid)
            else:
                b.path_detour = 1e400
    #
    def estimate_detour(b, t1):
        if volume_th < sum(t0.v for t0 in b.tasks.values()) + t1.v:
            return None
        else:
            pd_seq = b.path_pd_seq
            least_detour, i0, j0 = 1e400, None, None
            for i in range(len(pd_seq)):
                if i == len(pd_seq) - 1:
                    j = i
                    p0 = get_point(b, pd_seq[i])
                    p1 = b.p.dest
                    detour = travel_time[p0, t1.pp] + \
                                        travel_time[t1.pp, t1.dp] + \
                                        travel_time[t1.dp, p1]
                    detour -= travel_time[p0, p1]
                    if detour < least_detour:
                        least_detour, i0, j0 = detour, i, j
                else:
                    for j in range(i, len(pd_seq)):
                        if i == j:
                            #
                            # task t1's pick-up is just before t1's delivery
                            #
                            if i == 0:
                                p0 = b.p.ori
                                p1 = get_point(b, pd_seq[i])
                            else:
                                p0 = get_point(b, pd_seq[i])
                                p1 = get_point(b, pd_seq[i + 1])
                            detour = travel_time[p0, t1.pp] + \
                                                travel_time[t1.pp, t1.dp] + \
                                                travel_time[t1.dp, p1]
                            detour -= travel_time[p0, p1]
                        else:
                            #
                            # task t1's pick-up is not just before t1's delivery
                            #
                            if i == 0:
                                #
                                # Insert a new pick-up task before the first pick-up task
                                #
                                p0 = b.p.ori
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
                                p3 = b.p.dest
                            else:
                                p2 = get_point(b, pd_seq[j])
                                p3 = get_point(b, pd_seq[j + 1])
                            #
                            detour = travel_time[p0, t1.pp] + \
                                                travel_time[t1.pp, p1] + \
                                                travel_time[p2, t1.dp] + \
                                                travel_time[t1.dp, p3]
                            detour -= travel_time[p0, p1]
                            detour -= travel_time[p2, p3]
                        if detour < least_detour:
                            least_detour, i0, j0 = detour, i, j
            assert least_detour != 1e400, (b, pd_seq)
            #
            return least_detour, i0, j0
    #
    def get_point(b, pd_name):
        t = b.tasks[int(pd_name[len('p0'):])]
        if pd_name.startswith('p0'):
            return t.pp
        else:
            assert pd_name.startswith('d')
            return t.dp
    #
    # start bundling
    #
    bundles = [bundle4onePath(bid, paths[k]) for bid in range(num_bundles)]
    #
    candi_bundles = bundles
    while candi_bundles:
        new_candi_bundles = []
        for i, b in enumerate(candi_bundles):
            if not b.tasks:
                for i, t0 in enumerate(tasks):
                    seq = ['p0%d' % t0.tid, 'd%d' % t0.tid]
                    b.path_pd_seq = seq[:]
                    detour = travel_time[b.p.ori, t0.pp] + \
                             travel_time[t0.pp, t0.dp] + \
                             travel_time[t0.dp, b.p.dest]
                    detour -= travel_time[b.p.ori, b.p.dest]
                    if detour <= detour_th:
                        best_task = tasks.pop(i)
                        insert_task(b, best_task)
                        new_candi_bundles += [b]
                        break
            else:
                least_detour = 1e400
                best_task, i0, j0 = None, None , None
                for t in tasks:
                    task_insertion_estimation = estimate_detour(b, t)
                    if task_insertion_estimation is None:
                        continue
                    detour_estimation, i1, j1 = task_insertion_estimation
                    if detour_estimation < least_detour:
                        least_detour = detour_estimation
                        best_task, i0, j0 = t, i1, j1
                if b.path_detour + least_detour <= detour_th:
                    tasks.pop(tasks.index(best_task))
                    insert_task(b, best_task, (least_detour, i0, j0))
                    new_candi_bundles += [b]
        candi_bundles = new_candi_bundles
    unassigned_tasks = [t.tid for t in tasks]
    #
    return sum(b.bundle_attr for b in bundles), [[t.tid for t in b.tasks.values()] for b in bundles], unassigned_tasks


if __name__ == '__main__':
    from problems import *
    print(run(ex2(), 1))
