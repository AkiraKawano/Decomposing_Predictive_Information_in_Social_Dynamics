import numpy as np
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
import numpy.ma as ma

def TDMI3(x,y,z,maxtau,T,lag=1,ka=0):
    '''
    I(z;x,y)
    '''
    a1 = delay_embedding(np.array(x),maxtau,s=T)[:-lag].T[::-1].T
    maxt = a1.shape[1]
    D = x.shape[1]
#     print(maxt,D)
#     print(y[:10])
    a2 = delay_embedding(np.array(y),maxtau,s=T)[:-lag].T[::-1].T
#     print(a2[0])
    a = np.concatenate((a1,a2),axis=1)
#     print(z[:10])
    b = delay_embedding(np.array(z),maxtau,s=T)[lag:].T[::-1].T[:,0:D]
#     print(b[0])
    c = np.concatenate((b,a),axis=1)
    cov = ma.cov(ma.masked_invalid(c.T))
    val,rvec,lvec = lin.eig(ma.getdata(cov), left=True, right=True)
    val,rvec,lvec = val.real,rvec.real,lvec.real
    val[np.where(val<=0.)]=np.min(val[np.where(val>0.)])
    cov_c = rvec @ np.diag(val) @ lvec.T
#     print(cov_c.shape)
#     plt.plot(cov_c[0])
#     plt.show()
    cov_a = cov_c[D:,D:]
    cov_b = cov_c[:D,:D]
    hb = entropy_gaussian(cov_b,b,avoid_negative=avoid_negative,reg=reg)
    ha = entropy_gaussian(cov_a,a,avoid_negative=avoid_negative,reg=reg)
    hab = entropy_gaussian(cov_c,c,avoid_negative=avoid_negative,reg=reg)
    
    cov_a = cov_c[D:maxt+1,D:maxt+1]
    cov_c = cov_c[:maxt+1,:maxt+1]
#     print(cov_c.shape)
#     plt.plot(cov_c[0])
#     plt.show()
    cov_b = cov_c[:D,:D]

    hb2 = entropy_gaussian(cov_b,b,avoid_negative=avoid_negative,reg=reg)
    ha2 = entropy_gaussian(cov_a,a,avoid_negative=avoid_negative,reg=reg)
    hab2 = entropy_gaussian(cov_c,c,avoid_negative=avoid_negative,reg=reg)
    
    a = np.concatenate((a2,a1),axis=1)
    c = np.concatenate((b,a),axis=1)
    cov = ma.cov(ma.masked_invalid(c.T))
    val,rvec,lvec = lin.eig(ma.getdata(cov), left=True, right=True)
    val,rvec,lvec = val.real,rvec.real,lvec.real
    val[np.where(val<=0.)]=np.min(val[np.where(val>0.)])
    cov_c = rvec @ np.diag(val) @ lvec.T
    
    cov_a = cov_c[D:maxt+1,D:maxt+1]
    cov_c = cov_c[:maxt+1,:maxt+1]
    cov_b = cov_c[:D,:D]

    hb3 = entropy_gaussian(cov_b,b,avoid_negative=avoid_negative,reg=reg)
    ha3 = entropy_gaussian(cov_a,a,avoid_negative=avoid_negative,reg=reg)
    hab3 = entropy_gaussian(cov_c,c,avoid_negative=avoid_negative,reg=reg)
    return ha2+hb2-hab2, ha3+hb3-hab3, ha+hb-hab

def reduce_dim(a,dim):
    '''
    a.shape = (n_data,dim)
    '''
    a = ma.masked_invalid(a)
    w,v = np.linalg.eig(ma.cov(a.T))
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:,idx]
    return ma.dot(a, v)[:,:dim]

'''
def delay_embedding(a,K,dim=0,s=1):
    if dim==0:
        b = [np.reshape(a[i:i+K:s],(a[i:i+K:s].shape[1]*a[i:i+K:s].shape[0])) for i in range(a.shape[0]-K)]
    else:
        b = [np.reshape(a[i:i+K:s],(a[i:i+K:s].shape[1]*a[i:i+K:s].shape[0])) for i in range(a.shape[0]-K)]
        phspace, S, V = np.linalg.svd(b,full_matrices=0)
        b = np.dot(phspace, np.diag(S))[:,:dim]
    return np.array(b)
'''

def delay_embedding(a,K,dim=0,s=1):
    K = K+1
    if dim==0:
        b = [np.reshape(a[i:i+K:s],(a[i:i+K:s].shape[1]*a[i:i+K:s].shape[0])) for i in range(a.shape[0]-(K-1))]
    else:
        b = [np.reshape(a[i:i+K:s],(a[i:i+K:s].shape[1]*a[i:i+K:s].shape[0])) for i in range(a.shape[0]-(K-1))]
        b = reduce_dim(b,dim)
    return np.array(b)

def replace_nan(proj_vel_1):
    data = proj_vel_1[~np.isnan(proj_vel_1)]
    random.seed(0)
    shuffled_1 = random.sample(list(data),len(data))
    if len(data)*2 <= len(proj_vel_1):
        shuffled_1 = np.concatenate((shuffled_1,random.sample(list(data),len(data))))
    projvel1_replaced_nan = np.zeros(len(proj_vel_1))
    projvel1_replaced_nan = proj_vel_1
    projvel1_replaced_nan[np.isnan(projvel1_replaced_nan)] = shuffled_1[:sum(np.isnan(projvel1_replaced_nan))]
    return projvel1_replaced_nan

def replace_nan_vector(data):
    replaced = np.zeros(data.shape)
    for i in range(data.shape[1]):
        replaced.T[i] = replace_nan(data.T[i])
    return replaced


def moving_average(a,l,t=1):
    '''
    a : time series 
    l : window sise 
    t : moving step (step to move the window)
    '''
    b = np.empty(int(len(a)/t))
    b[:] = np.nan
    try:
        b[math.floor(l/2/t):] = np.array([np.nanmean(a[i*t:i*t+l]) for i in range(math.ceil((len(a)-l)/t))])
    except:
        try:
            b[math.floor(l/2/t):-math.floor(l/2/t)] = np.array([np.nanmean(a[i*t:i*t+l]) for i in range(math.floor((len(a)-l)/t))])
        except:
            b[math.floor(l/2/t):-math.floor(l/2/t)] = np.array([np.nanmean(a[i*t:i*t+l]) for i in range(math.ceil((len(a)-l)/t))])
    return b
def moving_sd(a,l,t=1):
    '''
    a : time series 
    l : window sise 
    t : moving step (step to move the window)
    '''
    b = np.empty(int(len(a)/t))
    b[:] = np.nan
    try:
        b[math.floor(l/2/t):] = np.array([np.sqrt(np.nanmean(a[i*t:i*t+l]**2) - np.nanmean(a[i*t:i*t+l])**2) for i in range(math.ceil((len(a)-l)/t))])
    except:
        try:
            b[math.floor(l/2/t):-math.floor(l/2/t)] = np.array([np.sqrt(np.nanmean(a[i*t:i*t+l]**2) - np.nanmean(a[i*t:i*t+l])**2) for i in range(math.floor((len(a)-l)/t))])
        except:
            b[math.floor(l/2/t):-math.floor(l/2/t)] = np.array([np.sqrt(np.nanmean(a[i*t:i*t+l]**2) - np.nanmean(a[i*t:i*t+l])**2) for i in range(math.ceil((len(a)-l)/t))])
    return b

# def moving_average(a,l,t=1):
#     '''
#     a : time series 
#     l : window sise 
#     t : moving step (step to move the window)
#     '''
#     b = np.empty(int(len(a)/t))
#     b[:] = np.nan
#     b[math.floor(l/2/t):-math.floor(l/2/t)] = np.array([np.nanmean(a[i*t:i*t+l]) for i in range(math.ceil((len(a)-l)/t))])
#     return b
# def moving_sd(a,l,t=1):
#     '''
#     a : time series 
#     l : window sise 
#     t : moving step (step to move the window)
#     '''
#     b = np.empty(int(len(a)/t))
#     b[:] = np.nan
#     b[math.floor(l/2/t):-math.floor(l/2/t)] = np.array([np.sqrt(np.nanmean(a[i*t:i*t+l]**2) - np.nanmean(a[i*t:i*t+l])**2) for i in range(math.ceil((len(a)-l)/t))])
#     return b


def autocorrelations(data, shifts, avg=True):
    """Returns the autocorrelation of the *k*th lag in a time series data.

    Parameters
    ----------
    data : one dimentional numpy array
    k : the *k*th lag in the time series data (indexing starts at 0)
    """
    y_avg = np.nanmean(data)
    
    if avg==False:
        y_avg = 0.
    
    denominators = [(a-y_avg)**2 for a in data]
    mean_of_denominator = np.nanmean(denominators)

    mean_of_covariances = []
    
    for k in shifts:
        covariances = [(data[i]-y_avg)*(data[i+k]-y_avg) for i in range(len(data[k:]))]
        mean_of_covariances.append(np.nanmean(covariances))
    mean_of_covariances = np.array(mean_of_covariances)
    return mean_of_covariances / mean_of_denominator

def crosscorrelations(data1, data2, shifts):
    """Returns the crosscorrelations (eq.14.1 in Kantz 2003)

    Parameters
    ----------
    data1 : one dimentional numpy array
    data2 : one dimentional numpy array
    shifts : one dimentional numpy array of lags in the time series data
    """

    avg1 = np.nanmean(data1)
    avg2 = np.nanmean(data2)
    
    var1 = np.nanmean([(a-avg1)**2 for a in data1])
    var2 = np.nanmean([(a-avg2)**2 for a in data2])
    
    mean_of_covariances = []
    
    for k in shifts:
        k = int(k)
        if k>=0:
            covariances = [(data1[i+k]-avg1)*(data2[i]-avg2) for i in range(len(data2[k:]))]
            mean_of_covariances.append(np.nanmean(covariances))
        else:
            covariances = [(data1[i+k]-avg1)*(data2[i]-avg2) for i in range(len(data2[:k]))]
            mean_of_covariances.append(np.nanmean(covariances))            
    mean_of_covariances = np.array(mean_of_covariances)
    return mean_of_covariances / np.sqrt(var1*var2)

def get_velocity(idx, ep, tau=10*60*100, count=0, norm=False):
    file_list = [["FishTank20200127_143538", 618000,804000,0,1],
    ["FishTank20200129_140656", 480000,660000,0,1],
    ["FishTank20200130_153857", 15900,178000,0,1], #2 the canonical experiment
    ["FishTank20200130_181614", 318000,432000,0,1],
    ["FishTank20200213_154940", 468000,564000,0,1],
    ["FishTank20200214_153519", 198000,252000,0,1],
    ["FishTank20200217_160052", 480000,552000,0,1],
    ["FishTank20200218_153008", 180000,360000,0,1],
    ["FishTank20200327_154737", 18000,78000,0,1],
    ["FishTank20200330_161100", 222000,282000,0,1],
    ["FishTank20200331_162136", 126000,360000,0,1],
    ["FishTank20200520_152810", 684000,744000,0,1],
    ["FishTank20200521_154541", 18000,66000,0,1],
    ["FishTank20200525_161602", 18000,180000,0,1], #13 nice linear slope
    ["FishTank20200526_160100", 228000,276000,0,1],
    ["FishTank20200824_151740", 420000,540000,0,1],
    ["FishTank20200828_155504", 492000,570000,0,1],
    ["FishTank20200902_160124", 36000,156000,0,1],
    ["FishTank20200903_160946", 168000,228000,0,1]]
    # set the path of the h5 file containing the trajectories
    filename = file_list[idx][0]
    print(file_list[idx][0])
    datapath = '/bucket/StephensU/akira/data/trajectories_A/'+filename+'.h5'
    # load the raw tracking results (3D) and the associated image coordinates
    with h5py.File(datapath, 'r') as hf:
        #trajectories_3D = hf['tracks_3D_smooth'][:]
        trajectories_3D = hf['tracks_3D_raw'][:]
        trajectories_imCoords = hf['tracks_imCoords_raw'][:]
        winner = hf['winnerIdx'][()]
    loser = int(not winner)
    if idx == 4:
        winner = int(not winner)
        loser = int(not loser)
    fight_start = file_list[idx][1]
    fight_end = file_list[idx][2]
    #f0, fE = 0, -1 # start/end of frame
    if ep == 'fight':
        f0, fE = fight_start, fight_end # start/end of frame
    elif ep == 'before':
        f0, fE = fight_start-tau, fight_start
    else:
        f0, fE = fight_end, fight_end+tau

    fishIdx = winner # fish 1 (Winner)
    bpIdx = 1 # for the head. bpIdx=1 for the pec, bpIdx=2 for the tail
    pec_1 = np.array([trajectories_3D[f0:fE, fishIdx, bpIdx, coordIdx] for coordIdx in range(3)]).T


    fishIdx = loser # fish 2 (Loser)
    bpIdx = 1 # for the head. bpIdx=1 for the pec, bpIdx=2 for the tail
    pec_2 = np.array([trajectories_3D[f0:fE, fishIdx, bpIdx, coordIdx] for coordIdx in range(3)]).T

    dt = 1/100. # sampling interval (100 fps)
    vel_1 = np.array(pec_1[1:]-pec_1[:-1])/dt # forward difference approximation of pec velocity 
    vel_2 = np.array(pec_2[1:]-pec_2[:-1])/dt

    T = 120
    '''
    for i in range(len(vel_1)):
        x = np.linalg.norm(vel_1[i])
        y = np.linalg.norm(vel_2[i])
        if x > T and y > T:
            pec_1[i] = np.nan
            pec_2[i] = np.nan
            pec_1[i+1] = np.nan
            pec_2[i+1] = np.nan
    '''
    for i in range(len(vel_1)):
        x = np.linalg.norm(vel_1[i])
        if x > T:
            pec_1[i] = np.nan
            pec_1[i+1] = np.nan
    for i in range(len(vel_2)):
        x = np.linalg.norm(vel_2[i])
        if x > T:
            pec_2[i] = np.nan
            pec_2[i+1] = np.nan
            
    if norm == True:
        vel_1 = np.array(pec_1[2:]-pec_1[:-2])/(2.*dt) # (2nd order) central difference approximation of pec velocity 
        vel_2 = np.array(pec_2[2:]-pec_2[:-2])/(2.*dt)
        vec_nans = [np.nan, np.nan, np.nan]
        vel_1 = np.insert(vel_1, 0, vec_nans, axis=0)
        vel_2 = np.insert(vel_2, 0, vec_nans, axis=0)
        speed_1 = np.array([np.linalg.norm(a) for a in vel_1])
        speed_2 = np.array([np.linalg.norm(a) for a in vel_2])
        vel_1 = vel_1/np.nanmean(speed_1)
        vel_2 = vel_2/np.nanmean(speed_2)

        rel_pos = np.array([pec_2[i] - pec_1[i] for i in range(len(pec_1))])
        dist = np.array([np.linalg.norm(a) for a in rel_pos])
        proj_vel_1 = np.array([np.dot(vel_1[i],rel_pos[i].T)/dist[i] for i in range(len(vel_1))])
        proj_vel_2 = np.array([np.dot(vel_2[i],-rel_pos[i].T)/dist[i] for i in range(len(vel_2))])
    else:
        vel_1 = np.array(pec_1[2:]-pec_1[:-2])/(2.*dt) # (2nd order) central difference approximation of pec velocity 
        vel_2 = np.array(pec_2[2:]-pec_2[:-2])/(2.*dt)
        vec_nans = [np.nan, np.nan, np.nan]
        vel_1 = np.insert(vel_1, 0, vec_nans, axis=0)
        vel_2 = np.insert(vel_2, 0, vec_nans, axis=0)

        rel_pos = np.array([pec_2[i] - pec_1[i] for i in range(len(pec_1))])
        dist = np.array([np.linalg.norm(a) for a in rel_pos])
        proj_vel_1 = np.array([np.dot(vel_1[i],rel_pos[i].T)/dist[i] for i in range(len(vel_1))])
        proj_vel_2 = np.array([np.dot(vel_2[i],-rel_pos[i].T)/dist[i] for i in range(len(vel_2))])
        
    return proj_vel_1, proj_vel_2

def get_velocity_combined(order, ep, tau=10*60*100, norm=False):
    #order = order[6:11] # pair3
    proj_vel_1_all, proj_vel_2_all = np.array([]), np.array([])

    for i in order:
        try:
            proj_vel_1, proj_vel_2 = get_velocity(i,ep,tau,norm=norm)
            proj_vel_1_all, proj_vel_2_all = np.concatenate([proj_vel_1_all, proj_vel_1]), np.concatenate([proj_vel_2_all, proj_vel_2])
        except:
            print("error_at_"+str(i))
    return proj_vel_1_all, proj_vel_2_all


def plot_hist(title,x,y,ymax,xlim=50,label='$V_{i\\to j}$ (cm/s)'):  
    plt.rcParams.update({'font.size': 16})
    nbins = 150
    #ymax = 0.18
    plt.title(title)
    sns.histplot(data=x,stat='probability',bins=nbins, binrange=(-xlim,xlim),color='r',alpha=0.5,element='step',label='$V_{1\\to2}$')
    sns.histplot(data=y,stat='probability',bins=nbins, binrange=(-xlim,xlim),alpha=0.5,element='step',label='$V_{2\\to1}$')
    plt.xlabel(label)
    plt.vlines(0,ymin=0,ymax=ymax,linestyles='--',colors='k',lw=0.5)
    #plt.ylim([0,ymax])
    plt.legend()
    plt.show()
    
def plot_2d_hist(x,y,nbins=60,lim=15,Cmax=0):  
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=[8*0.8,6.3*0.8])
    #plt.title(title)
    plt.hist2d(replace_nan(x),replace_nan(y),bins=nbins,range=[[-lim,lim],[-lim,lim]],normed = False,cmap=plt.cm.jet)#'Greys')
    plt.xlabel('$V_{1\\to 2}$')
    plt.ylabel('$V_{2\\to 1}$')
    #plt.vlines(0,ymin=0,ymax=ymax,linestyles='--',colors='k')
    #plt.ylim([0,ymax])
    plt.colorbar(label='Counts')
    plt.show()
    if Cmax!=0:
        plt.figure(figsize=[8*0.8,6.3*0.8])
        plt.hist2d(replace_nan(x),replace_nan(y),bins=nbins,range=[[-lim,lim],[-lim,lim]],normed=False,cmap=plt.cm.jet,cmax=Cmax)#'Greys')
        plt.xlabel('$V_{1\\to 2}$')
        plt.ylabel('$V_{2\\to 1}$')
        #plt.vlines(0,ymin=0,ymax=ymax,linestyles='--',colors='k')
        #plt.ylim([0,ymax])
        plt.colorbar(label='Counts')
        plt.show()
        
def plot_ccorr(x,y,maxtau,npoints,compare=True,legend=False,y0=-0.55,y1=0.2):
    shifts = np.linspace(-maxtau,maxtau,npoints,dtype=int)
    ccorr = crosscorrelations(x,y,shifts)
    shuffled1 = random.sample(list(x),len(x))
    shuffled2 = random.sample(list(y),len(y))
    ccorr_shuf = crosscorrelations(shuffled1,shuffled2,shifts)
    fig = plt.figure(figsize=(5*1.2,4*1.2))
    plt.plot(shifts/100,ccorr,'.-',label='Observed')
    plt.plot(shifts/100,ccorr_shuf,'.r',label='Shuffle')
    plt.vlines(0,ymin=y0,ymax=y1,linestyle='-',lw=0.5)
    plt.hlines(0,xmin=-maxtau/100,xmax=maxtau/100,linestyle='-',lw=0.5)
    plt.xlim([-maxtau/100,maxtau/100])
    plt.ylim([y0,y1])
    plt.ylabel('$C(\\tau)$')
    plt.xlabel('time $\\tau$ [s]')
    if legend==True:
        plt.legend()
    #plt.title(str(format((i*l/100/60), '.1f'))+' mim (fight start: ' + str(int(fight_start/100/60))+' mim, fight end: ' + str(int(fight_end/100/60))+' min)',fontsize=18)
    plt.show()
    if compare==True:
        fig = plt.figure(figsize=(5*0.5,4*0.5))
        plt.plot(shifts[np.where(shifts>=0)]/100,np.flip(ccorr[np.where(shifts<=0)]),'-r', label='$C_{12}$')
        plt.plot(shifts[np.where(shifts>=0)]/100,ccorr[np.where(shifts>=0)],'-b', label='$C_{12}$')
        #plt.vlines(0,ymin=y0,ymax=y1,linestyle='-',lw=0.5)
        plt.hlines(0,xmin=0,xmax=maxtau/100,linestyle='--',lw=0.5)
        plt.xlim([0,maxtau/100])
        plt.ylim([y0,y1])
        plt.ylabel('$C_{ij}(\\tau)$')
        plt.xlabel('time $\\tau$ [s]')
        plt.legend(loc='lower right')
        #plt.title(str(format((i*l/100/60), '.1f'))+' mim (fight start: ' + str(int(fight_start/100/60))+' mim, fight end: ' + str(int(fight_end/100/60))+' min)',fontsize=18)
        plt.show()
        
def jointProb(labels_1,labels_2):
    numStates_1 = max(labels_1)+1#len((np.unique(ma.compressed(labels_1))))
    numStates_2 = max(labels_2)+1#len((np.unique(ma.compressed(labels_2))))
    mtx1=np.zeros([numStates_1,numStates_2],dtype=int)
    for i in np.unique(labels_1):
        times = np.where(labels_1==i)[0]
        cross_map = labels_2[times]
        for mapped in cross_map:
            mtx1[i][mapped]+=1
    numObs1 = len(labels_1)
    #print(len(labels_1)-np.sum(labels_1.mask),numObs1)
    return mtx1/(numObs1)

def get_info(labels_1,labels_2):
    joprob = jointProb(labels_1,labels_2)
    prob_1 = np.array([np.sum(joprob[i]) for i in range(joprob.shape[0])])
    prob_2 = np.array([np.sum(joprob.T[i]) for i in range(joprob.shape[1])])
    info = np.zeros(len(labels_1),dtype=float)
    for i in range(len(labels_1)):
        info[i] = (np.log(joprob[labels_1[i],labels_2[i]]/(prob_1[labels_1[i]]*prob_2[labels_2[i]])))
    return np.nanmean(info)




