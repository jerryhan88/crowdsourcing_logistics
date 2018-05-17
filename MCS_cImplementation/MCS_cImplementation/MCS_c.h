//
//  MCS_c.h
//  MCS_cImplementation
//
//  Created by Chung-Kyun HAN on 23/04/2018.
//  Copyright © 2018 Chung-Kyun HAN. All rights reserved.
//

#ifndef MCS_c_h
#define MCS_c_h

typedef struct {
    int tid;
    int pp, dp;
    int reward, volume;
} Task;

typedef struct {
    char *problemName;
    int numBundles, thVolume, thDetour;
    int numLocations;
    
    int numTasks;
    Task *tasks;
    
    float **travel_time;
    float **flow;
    
    char *logF, *resF, *itrF;
    
} Problem;

typedef struct {
    int nB, nT;
    int *r_i, *v_i;
    float _lambda;
    
    int nK;
    double *w_k;
    
    double **t_ij;
    double _delta;
    
} CP;


Problem read_pkl_file(char *fpath);

#endif /* MCS_c_h */
