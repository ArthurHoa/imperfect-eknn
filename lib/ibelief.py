"""
Adapted from https://github.com/jusdesoja/iBelief_python (27/11/2021).
"""

import numpy as np
import math

def DST(massIn, criterion, TypeSSF=0):
    """
    Combination rules for multiple masses.
    Parameters
    ----------
    massIn: ndarray
        Masses to be combined, represented by a 2D matrix
    criterion: integer
        Combination rule to be applied
        The criterion values represented respectively the following rules:
            criterion=1 Smets criterion (conjunctive combination rule)
            criterion=2 Dempster-Shafer criterion (normalized)
            criterion=3 Yager criterion
            criterion=4 Disjunctive combination criterion
            criterion=5 Dubois criterion (normalized and disjunctive combination)
            criterion=6 Dubois and Prade criterion (mixt combination), only for Bayesian masses whose focal elements are singletons
            criterion=7 Florea criterion
            criterion=8 PCR6
            criterion=9 Cautious Denoeux Min for functions non-dogmatics
            criterion=10 Cautious Denoeux Max for separable masses
            criterion=11 Hard Denoeux for functions sub-normal
            criterion=12 Mean of the bbas
            criterion=13 LNS rule, for separable masses
            criterion=131 LNSa rule, for separable masses
    TypeSSF: integer
        If TypeSSF = 0, it is not a SSF (the general case).
        If TypeSSF = 1, it is a SSF with a singleton as a focal element.
        If TypeSSF = 2, it is a SSF with any subset of \Theta as a focal element.
    Return
    ----------
    Mass: ndarray
        a final mass vector combining all masses
    """
    n, m = massIn.shape
    if criterion in (4,5,6,7):
        b_mat = np.apply_along_axis(mtob, axis = 0, arr = massIn)
        b = np.apply_along_axis(np.prod, axis = 1, arr = b_mat )
    if criterion in (1,2,3,6,7, 14):

        q_mat = np.apply_along_axis(mtoq, axis = 0, arr = massIn) #apply on column. (2 in R)
        q = np.apply_along_axis(np.prod, axis = 1,arr = q_mat) # apply on row (1 in R)
    if criterion == 1:
        #Smets criterion
        Mass = qtom(q)
        Mass[0] = 1.0 - np.sum(Mass[1:])
    elif criterion == 2:
        #Dempster-Shafer criterion (normalized)
        Mass = qtom(q)
        Mass = Mass/(1-Mass[0])
        Mass[0] = 0
    elif criterion == 3:
        #Yager criterion
        Mass = qtom(q)
        Mass[-1] = Mass[-1]+Mass[0]
        Mass[0]=0
    elif criterion == 4:
        #disjunctive combination
        Mass = btom(b)
    elif criterion == 5:
        # Dubois criterion (normalized and disjunctive combination)
        Mass = btom(b)
        Mass = Mass/(1-Mass[0])
        Mass[0] = 0
    #elif criterion == 6:
    #elif criterion == 7:
    #elif criterion == 8:
    elif criterion == 9:
        #Cautious Denoeux min for fonctions non-dogmatic
        wtot = np.apply_along_axis(mtow, axis = 0, arr = massIn)
        w = np.apply_along_axis(np.ndarray.min , axis = 1, arr = wtot )
        Mass = wtom(w)
    elif criterion == 10:
        #Cautious Denoeux max only for separable fonctions
        wtot = np.apply_along_axis(mtow, axis = 0, arr = massIn)
        w = np.apply_along_axis(np.ndarray.max, axis = 1, arr = wtot)
        Mass = wtom(w)
    #elif criterion == 11:
    elif criterion == 12:
        # mean of the masses
        Mass = np.apply_along_axis(np.mean, axis = 1, arr = massIn)
    elif criterion == 13:
        if TypeSSF==0:
            Mass = LNS(massIn, mygamma = 1)
        elif TypeSSF == 1:
            Mass = LNS_SSF(massIn, mygamma =1, singleton =True)
        elif TypeSSF == 2:
            Mass = LNS_SSF(massIn, mygamma = 1, singleton= False)
    elif criterion == 14:
        Mass = qtom(np.apply_along_axis(np.mean, axis=1, arr=q_mat))
    return Mass[np.newaxis].transpose()


def LNS(massIn, mygamma,ifnormalize = False, ifdiscount = True, approximate=False, eta = 0):
    """
    LNS rule.  Can also be used for conjunctive rule, cautious rule and DS rule.
    Only for seperable masses
    Parameters
    ----------
    massIn: ndarray
        Masses to be combined, represented by a 2D matrix
    mygamma: integer
        Parameter of the family of conjunctive and disjunctive rules using triangular norms by Denoeux.
        mygamma = 1, with ifnormalize = FALSE, smets conjunctive rule
        mygamma = 1, with ifnormalize = TRUE, Dempster rule
        mygamma = 0, cautious rule
        mygamma between 0 and 1, the generalized case of the rules using triangular by Denoeux
        mygamma = 1, ifnormalize = FALSE, ifdiscount = TRUE, LNS rule
    approximate: bool
        If TRUE, LNSa rule is conducted, which is the approximation method for LNS rule
    eta: float
        A parameter in LNS rule, controling the specificity of the decision
    Return
    ----------
    Mass: ndarray
        a final mass vector combining all masses
    """
    nf,n = massIn.shape
    ThetaSize = np.log2(nf)
    w_mat = np.apply_along_axis(mtow,axis = 0,arr = massIn)
    if approximate:
        # This case has not been tested
        num_eff = np.apply_along_axis(lambda x: np.sum(np.abs(x-1)>1e-6) ,1,arr = w_mat)
        id_eff = np.where(num_eff > 0)
        num_group_eff = num_eff[id_eff]
        beta_vec = np.ones(len(id_eff))
        if (eta != 0):
            myc = np.array([np.sum([int(d) for d in bin(xx)[2:][::-1]]) for xx in range(nf)])
            beta_vec = (ThetaSize / myc[id_eff]) **eta
        alpha_vec = beta_vec * num_group_eff / np.sum(beta_vec * num_group_eff)
        w_eff = 1-alpha_vec
        w_vec = np.ones(nf)
        w_vec[id_eff] = w_eff
    else:
        if mygamma == 1:
            w_vec = np.apply_along_axis(np.prod,axis=1, arr = w_mat)
        elif mygamma == 0:
            w_vec = np.apply_along_axis(np.ndarray.min, axis = 1, arr = w_mat)
        #else:
        if ifdiscount:
            num_eff= np.apply_along_axis(lambda x: np.sum(np.abs(x-1)>1e-6), axis=1, arr=w_mat)
            id_eff = np.where(num_eff>0)
            w_eff = w_vec[id_eff]
            num_group_eff = num_eff[id_eff]
            beta_vec = np.ones(len(id_eff))
            if eta != 0:
                myc = np.array([np.sum([int(d) for d in bin(xx)[2:][::-1]]) for xx in range(nf)])
                beta_vec = (ThetaSize / myc[id_eff]) ** eta
            alpha_vec = beta_vec * num_group_eff / np.sum(beta_vec * num_group_eff)
            w_eff = 1 - alpha_vec + alpha_vec * w_eff
            w_vec[id_eff] = w_eff
    out = wtom(w_vec)
    if ifnormalize and mygamma:
        out[0] = 0
        out = out/np.sum(out)
    return out


def LNS_SSF(massIn, mygamma, ifnormalize = False, ifdiscount =True, approximate=False, eta = 0, singleton = False):
    """
    LNS rule for single support mass functions
    
    Parameters
    ----------
    massIn: ndarray
        Masses to be combined, represented by a 2D matrix
    mygamma: integer
        Parameter of the family of conjunctive and disjunctive rules using triangular norms by Denoeux.
        mygamma = 1, with ifnormalize = FALSE, smets conjunctive rule
        mygamma = 1, with ifnormalize = TRUE, Dempster rule
        mygamma = 0, cautious rule
        mygamma between 0 and 1, the generalized case of the rules using triangular by Denoeux
        mygamma = 1, ifnormalize = FALSE, ifdiscount = TRUE, LNS rule
    approximate: bool
        If TRUE, LNSa rule is conducted, which is the approximation method for LNS rule
    Return
    ----------
    Mass: ndarray
        a final mass vector combining all masses
    """
    m,n = massIn.shape
    if singleton:
        ThetaSize = m
        nf = 2 ** m
        w_mat = massIn[0:-1,::]
        w_mat = 1 - w_mat
        eta = 0
    else:
        nf = m
        ThetaSize = np.log2(nf)
        w_mat = massIn[0: -1, ::]
        w_mat = 1-w_mat
    if approximate:
        num_eff = np.apply_along_axis(lambda x: np.sum(np.abs(x - 1) > 1e-6), axis = 1, arr = w_mat)
        id_eff = np.argwhere(num_eff > 0)
        num_group_eff = num_eff[id_eff]
        if(eta != 0):
            beta_vec = np.ones(len(id_eff))
            myc = np.array([np.sum([int(d) for d in bin(xx)[2:][::-1]]) for xx in range(nf)])
            beta_vec = (ThetaSize/myc[id_eff]) ** eta
            alpha_vec = beta_vec * num_group_eff /np.sum(beta_vec * num_group_eff)
        else:
            alpha_vec = num_group_eff / sum(num_group_eff)
        w_eff = 1 - alpha_vec
        if(singleton):
            w_vec = np.ones(ThetaSize)
        else:
            w_vec = np.ones(nf-1)
        w_vec[id_eff] = w_eff
    else:
        if mygamma == 1:
            w_vec = np.apply_along_axis(np.prod, 1, arr = w_mat)
        elif mygamma == 0:
            w_vec = np.apply_along_axis(np.ndarray.min, 1,arr = w_mat)
        #else: TODO to complete
        if ifdiscount:
            num_eff = np.apply_along_axis(lambda x: np.sum(np.abs(x-1)>1e-6), axis = 1, arr =w_mat)
            id_eff = np.argwhere(num_eff > 0)
            w_eff = w_vec[id_eff]
            num_group_eff = num_eff[id_eff]
            if eta != 0:
                beta_vec = np.ones(len(id_eff))
                myc = np.array([np.sum([int(d) for d in bin(xx)[2:][::-1]]) for xx in range(nf)])
                beta_vec = (ThetaSize/myc[id_eff]) ** eta
                alpha_vec = beta_vec * num_group_eff / np.sum(beta_vec * num_group_eff)
            else:
                alpha_vec = num_group_eff / np.sum(num_group_eff)
            w_eff = 1 - alpha_vec + alpha_vec * w_eff
            w_vec[id_eff] = w_eff
    w_vec_complete = np.ones(nf)
    if singleton:
        w_vec_complete[2*np.arange(ThetaSize)] = w_vec
    else:
        w_vec_complete[0:-1] = w_vec
    if np.ndarray.min(w_vec_complete)>0:
        out = wtom(w_vec_complete)
    else:
        id = np.argwhere(w_vec_complete==0)
        out = np.zeros(nf)
        out[id] = 1
    if(ifnormalize & mygamma == 1):
        out[0] = 0
        out = out/np.sum(out)
    return out.T


def mtobetp(InputVec):
    """Computing pignistic propability BetP on the signal points from the m vector (InputVec) out = BetP
    vector beware: not optimize, so can be slow for more than 10 atoms
    Parameter
    ---------
    InputVec: a vector representing a mass function
    Return
    ---------
    out: a vector representing the correspondant pignistic propability
    """
    # the length of the power set, f
    mf = InputVec.size
    # the number of the signal point clusters
    natoms = round(math.log(mf, 2))
    if math.pow(2, natoms) == mf:
        if InputVec[0] == 1:
            # bba of the empty set is 1
            raise ValueError("warning: all bba is given to the empty set, check the frame\n")
            out = np.ones(natoms)/natoms
        else:
            betp = np.zeros(natoms)
            for i in range(1, mf):
                # x , the focal sets InputVec the dec2bin form
                x = np.array(list(map(int, np.binary_repr(i, natoms)))[
                             ::-1])  # reverse the binary expression
                # m_i is assigned to all the signal points equally

                betp = betp + np.multiply(InputVec[i]/sum(x), x)
            out = np.divide(betp, (1.0 - InputVec[0]))
        return out
    else:
        raise ValueError(
            "Error: the length of the InputVec vector should be power set of 2, given %d \n" % mf)


def mtoq(InputVec):
    """
    Computing Fast Mobius Transfer (FMT) from mass function m to commonality function q
    Parameters
    ----------
    InputVec : vector m representing a mass function
    Return:
    out: a vector representing a commonality function
    """
    InputVec = InputVec.copy()
    mf = InputVec.size
    natoms = round(math.log2(mf))
    if 2 ** natoms == mf:
        for i in range(natoms):
            i124 = int(math.pow(2, i))
            i842 = int(math.pow(2, natoms - i))
            i421 = int(math.pow(2, natoms - i - 1))
            InputVec = InputVec.reshape(i124, i842, order='F')
            # for j in range(1, i421 + 1): #to be replaced by i842
            for j in range(i421):  # not a good way for add operation coz loop matrix for i842 times
                InputVec[:, j * 2] = InputVec[:, j * 2] + InputVec[:, j * 2+1]
        out = InputVec.reshape(1, mf, order='F')[0]
        return out
    else:
        raise ValueError(
            "ACCIDENT in mtoq: length of input vector not OK: should be a power of 2, given %d\n" % mf)


def mtob(InputVec):
    """
    Comput InputVec from m to b function.  belief function + m(emptset)
    Parameter
    ---------
    InputVec: vector m representing a mass function
    Return
    ---------
    out: a vector representing a belief function
    """
    InputVec = InputVec.copy()
    mf = InputVec.size
    natoms = round(math.log(mf, 2))
    if math.pow(2, natoms) == mf:
        for i in range(natoms):
            i124 = int(math.pow(2, i))
            i842 = int(math.pow(2, natoms - i))
            i421 = int(math.pow(2, natoms - i - 1))
            InputVec = InputVec.reshape(i124, i842, order='F')
            # for j in range(1, i421 + 1): #to be replaced by i842
            for j in range(i421):  # not a good way for add operation coz loop matrix for i842 times
                InputVec[:, j * 2 + 1] = InputVec[:,
                    j * 2 + 1] + InputVec[:, j * 2]
        out = InputVec.reshape(1, mf, order='F')[0]
        return out
    else:
        raise ValueError(
            "ACCIDENT in mtoq: length of input vector not OK: should be a power of 2, given %d\n" % mf)

# def btopl(InputVec):
#    """Compute pl from b InputVec
#    InputVec : vector f*1
#    out = pl

#    """
#    mf = InputVec.size
#    natoms = round(math.log(mf,2))
#    if math.pow(2, natoms) == mf:
#        InputVec = InputVec[-1] -


def mtonm(InputVec):
    """
    Transform bbm into normalized bbm
    Parameter
    ---------
    InputVec: vector m representing a mass function
    Return
    ---------
    out: vector representing a normalized mass function
    """
    if InputVec[0] < 1:
        out = InputVec/(1-InputVec[0])
        out[0] = 0
    return out


def mtobel(InputVec):
    return mtob(mtonm(InputVec))


def qtom(InputVec):
    """
    Compute FMT from q to m.
    Parameter
    ----------
    InputVec: commonality function q
    Return
    --------
    output: mass function m
    """
    InputVec = InputVec.copy()
    lm = InputVec.size
    natoms = round(math.log(lm, 2))
    if math.pow(2, natoms) == lm:
        for i in range(natoms):
            i124 = int(math.pow(2, i))
            i842 = int(math.pow(2, natoms - i))
            i421 = int(math.pow(2, natoms - i - 1))
            InputVec = InputVec.reshape(i124, i842, order='F')
            # for j in range(1, i421 + 1): #to be replaced by i842
            for j in range(i421):  # not a good way for add operation coz loop matrix for i842 times
                InputVec[:, j * 2] = InputVec[:, j * 2] - InputVec[:, j * 2+1]
        out = InputVec.reshape(1, lm, order='F')[0]
        return out
    else:
        raise ValueError("ACCIDENT in qtom: length of input vector not OK: should be a power of 2\n")


def btom(InputVec):
    """
    Compute FMT from b to m.
    Parameter
    ---------
    InputVec: commonality function q
    Return
    --------
    output: mass function m
    """
    mass_t = InputVec.copy()
    mf = mass_t.size
    natoms = round(math.log2(mf))
    if 2 ** natoms == mf:
        for i in range(natoms):
            i124 = int(2 ** i)
            i842 = int(2 ** (natoms - i))
            i421 = int(2 ** (natoms - i - 1))
            mass_t = mass_t.reshape(i124, i842, order='F')
            # for j in range(i421):    #not a good way for add operation coz loop matrix for i842 times
            #    InputVec[:, j * 2 ] = InputVec[:, j * 2 - 1] - InputVec[:, j * 2 - 2]
            # print(i421)
            mass_t[:, np.array(range(1, i421 + 1)) * 2 - 1] = mass_t[:, np.array(
                range(1, i421 + 1)) * 2 - 1] - mass_t[:, np.array(range(i421)) * 2]
        out = mass_t.reshape(1, mf, order='F')
        return out
    else:
        raise ValueError(
            "ACCIDENT in btom: length of input vector not OK: should be a power of 2, given %d\n" % mf)


def pltob(InputVec):
    """
    Compute from plausibility pl to belief b.
    Parameter
    ----------
    InputVec: plausibility function pl
    Return
    --------
    output: belief function m
    """
    mf = InputVec.size
    natoms = round(math.log2(mf))
    if 2 ** natoms == mf:
        InputVec = 1 - InputVec[::-1]
        out = InputVec
        return out
    else:
        raise ValueError(
            "ACCIDENT in pltob: length of input vector not OK: should be a power of 2, given %d\n" % mf)


def mtopl(InputVec):
    """
    Compute from mass function m to plausibility pl.
    Parameter
    ----------
    InputVec: mass function m
    Return
    --------
    output: plausibility function pl
    """
    InputVec = mtob(InputVec)
    out = btopl(InputVec)
    return out


def pltom(InputVec):
    """
    Compute from  plausibility pl to mass function m.
    Parameter
    ----------
    InputVec: plausibility function pl
    Return
    --------
    output: mass function m
    """

    out = btom(pltob(InputVec))
    return out


def qtow(InputVec):
    """
    Compute FMT from commonality q to weight w, Use algorithm qtom on log q
    Parameter
    ----------
    input: commonality function q
    Return
    ---------
    output: weight function w
    """
    InputVec = InputVec.astype(float)
    lm = InputVec.size
    natoms = round(math.log(lm, 2))
    if math.pow(2, natoms) == lm:
        if InputVec[-1] > 0:  # non dogmatic
            out = np.exp(-qtom(np.log(InputVec)))
            out[-1] = 1
        else:
            # print("""Accident in qtow: algorithm works only if q(lm) > 0\n
            # add an epsilon to m(lm)\n
            # Nogarantee it is really OK\n
            # """)
            """
            mini = 1
            for i in range(lm):
                if (InputVec[i] >0):
                    mini = min(mini, InputVec[i])
            mini = mini / 10000000.0
            for i in range(lm):
                InputVec[i] = max(InputVec[i],mini)
            """
            for i in range(lm):
                if InputVec[i] == 0:
                    InputVec[i] = 1e-9
            out = np.exp(-qtom(np.log(InputVec)))
            out[-1] = 1
    else:
        raise ValueError(
            "ACCIDENT in qtom: length of input vector not OK: should be a power of 2, given %d\n" % lm)
    return out

def decisionDST(mass, criterion, r=0.5, return_prob=False):
    """Different rules for decision making in the framework of belief functions
    
    Parameters
    -----------
    mass: mass function to decide with
    
    criterion: integer
        different decision rules to apply.
            criterion=1 maximum of the plausibility
            criterion=2 maximum of the credibility
            criterion=3 maximum of the credibility with rejection
	        criterion=4 maximum of the pignistic probability
	        criterion=5 Appriou criterion (decision onto \eqn{2^\Theta})
    """
    mass = mass.copy()
    if (mass.size in mass.shape):   #if mass is a 1*N or N*1 array
        mass = mass.reshape(mass.size,1)
    nbEF, nbvec_test = mass.shape   #number of focal elements = number of mass rows 
    nbClasses = round(math.log(nbEF,2))
    class_fusion = []


    for k in range(nbvec_test):
        massTemp = mass[:,k]

        #Select only singletons
        natoms = round(math.log(massTemp.size, 2))
        singletons_indexes = np.zeros(natoms)
        for i in range(natoms):
            if i == 1:
                singletons_indexes[i] = 2
            else:
                singletons_indexes[i] = 2**i

        singletons_indexes = singletons_indexes.astype(int)

        if criterion == 1:
            pl = np.array(mtopl(massTemp))
            indice = np.argmax(pl[singletons_indexes])
            class_fusion.append(indice)
        elif criterion == 2:
            bel = np.array(mtobel(massTemp))
            indice = np.argmax(bel[singletons_indexes])
            class_fusion.append(indice)
        elif criterion == 4:
            pign = np.array(mtobetp(massTemp.T))
            if return_prob:
                indice = pign
            else:
                indice = np.random.choice(np.flatnonzero(pign == pign.max()))
            class_fusion.append(indice)

    return np.array(class_fusion)

def btopl(InputVec):
    """
    Compute from belief b to plausibility pl
    Parameter
    ---------
    InputVec: belief function b
    Return
    ------
    out: plausibility function pl
    """

    lm = InputVec.size
    natoms = round(math.log2(lm))
    if 2 ** natoms == lm:
        InputVec = InputVec[-1] - InputVec[::-1]
        out = InputVec
        return out
    else:
        raise ValueError("ACCIDENT in btopl: length of input vector not OK: should be a power of 2, given %d\n" % lm)
############################################
# functions below are not tested
############################################
def wtoq(InputVec):
    """
    Compute FMT from weight w to commonality q
    
    Parameter
    ----------
    input: weight function w
    
    Return
    ---------
    output: commonality function q
    """
    lm = InputVec.size
    natoms = round(math.log2(lm))
    if 2 ** natoms == lm:
        if np.ndarray.min(InputVec) > 0:
            out = np.prod(InputVec)/np.exp(mtoq(np.log(InputVec)))
            return out
        else:
            raise ValueError('ACCIDENT in wtoq: one of the weights are non positive\n')
    else:
        raise ValueError('Accident in wtoq: length of input vector illegal: should be a power of 2')

def mtow(InputVec):
    """Compute FMT from m to w.
    
    Parameter
    ---------
    InputVec: mass function m
    
    Return
    ------
    output: weight function w
    """
    out = qtow(mtoq(InputVec))
    return out

def wtom(InputVec):
    """
    Compute FMT from weight w to mass function m
    
    Parameter
    ----------
    input: weight function w
    
    Return
    ---------
    output: mass function m
    """
    out = qtom(wtoq(InputVec))
    return out

def Dcalculus(lm):
    """Compute the Jaccard matrix for the disernment framework of the given mass function
	
	Parameter
	---------
	lm: a vector representing a mass function
	
	Return
	------
	out: the Jaccard matrix for the given mass function
	
	"""
    natoms = round(math.log2(lm))
    ind = [{}]*lm
    if (math.pow(2, natoms) == lm):
        ind[0] = {0} #In fact, the first element should be a None value (for empty set).
        #But in the following calculate, we'll deal with 0/0 which shoud be 1 bet in fact not calculable. So we "cheat" here to make empty = {0}
        ind[1] = {1}
        step = 2
        while (step < lm):
            ind[step] = {step}
            step = step+1
            indatom = step
            for step2 in range(1,indatom - 1):
                #print(type(ind[step2]))
                ind[step] = (ind[step2] | ind[indatom-1])
                #ind[step].sort()
                step = step+1
    out = np.zeros((lm,lm))

    for i in range(lm):
        for j in range(lm):
            out[i][j] = float(len(ind[i] & ind[j]))/float(len(ind[i] | ind[j]))
    return out

def JousselmeDistance(mass1, mass2, D = "None"):
    """
    Calclate Jousselme distance between two mass functions mass1 and mass2
	This function is able to calcuate the Jaccard matrix if not given. 
	Attention: calculation of Jaccard matrix is a heavy task.
	
	Parameters
	----------
	mass1: a vector representing first mass function
	
	mass2: a vector representing other mass function
	
	D: a square matrix representing Jaccard matrix
	
	Return
	------
	out: float representing the distance
    """
    m1 = np.array(mass1).reshape((1,mass1.size))
    m2 = np.array(mass2)
    if m1.size != m2.size:
        raise ValueError("mass vector should have the same size, given %d and %d" % (m1.size, m2.size))
    else:
        if type(D)== str:
            D = Dcalculus(m1.size)
        m_diff = m1 - m2
        #print(D)
        #print(m_diff.shape,D.shape)
        #print(np.dot(m_diff,D))
        #----JousselmeDistance modified for testing, don't forget to correct back------#

        out = math.sqrt(np.dot(np.dot(m_diff,D),m_diff.T)/2.0)
        #out = np.dot(np.dot(m_diff,D),m_diff.T)
        return out

def _calculateDistanceMat(singleton_dist_mat):
    """
    Calculate weighted distance on multiple singletons of preference.
    The simmilarity between two singletons is considered as a common part even though they are exclusive in the definition level.
    To simplify the calculation the dissimilarity between unions, we assume that the common part exist only between pairs. (i.e. no common part is shared among three or more elements)
    """
    n_singleton = singleton_dist_mat.shape[0]

    singleton_sim_mat = (1-np.eye(n_singleton)) - singleton_dist_mat # To avoid erreur in singleton self similarity, on singleton's self similarity is 0
    singleton_sim_mat = 2 * singleton_sim_mat/(1+singleton_sim_mat)
    print(singleton_sim_mat)
    n_element = 2 ** n_singleton
    dist_mat = np.zeros((n_element,n_element))
    for i in range(1,n_element + 1):
        for j in range(i+1,n_element):

            A_vec = np.array([int(d) for d in bin(i)[2:][::-1]])
            B_vec = np.array([int(d) for d in bin(j)[2:][::-1]])
            A_vec = np.pad(A_vec,(0,n_singleton-A_vec.size),'constant',constant_values=(0))

            B_vec = np.pad(B_vec,(0,n_singleton-B_vec.size),'constant',constant_values=(0))

            common_vec = np.logical_and(A_vec,B_vec)
            diff_vec = np.logical_xor(A_vec,B_vec)
            all_vec = np.logical_or(A_vec,B_vec)

            p_common = np.count_nonzero(common_vec==1) \
                - np.dot(common_vec, np.dot(singleton_sim_mat, common_vec.T)) / 2 \
                + np.dot(diff_vec, np.dot(singleton_sim_mat, diff_vec.T)) / 2 \
                - np.dot(np.logical_and(A_vec,diff_vec),np.dot(singleton_sim_mat,np.logical_and(A_vec,diff_vec)))/2\
                - np.dot(np.logical_and(B_vec,diff_vec),np.dot(singleton_sim_mat,np.logical_and(B_vec,diff_vec)))/2
            p_all =  np.count_nonzero(all_vec == 1)\
                - np.dot(all_vec, np.dot(singleton_sim_mat,all_vec.T)) / 2

            dist_mat[i][j] = p_common / p_all
    dist_mat = 1 - (dist_mat + dist_mat.T + np.eye(n_element))
    return dist_mat

def weighted3SingletonDistance(mass1, mass2, singleton_dist_mat):
    """
    Weighted distance on multiple singletons of preferences.
    Parameter
    ---------
    mass1: mass function of size 8
    mass2: mass function of size 8
    singleton_dist_mat: matrix of distances between different singletons
    Return
    ---------
    dist: distance of mass1 and mass2
    """
    n_singleton = round(math.log2(mass1.size))
    if singleton_dist_mat.shape[0] != n_singleton:
        raise ValueError("mass and singleton distance matrix are not compatible!")
    else:
        m1 = np.array(mass1).reshape((1,mass1.size))
        m2 = np.array(mass2).reshape((1,mass2.size))
        dist_mat = _calculateDistanceMat(singleton_dist_mat)
        m_diff = m1 - m2
        return math.sqrt(np.dot(np.dot(m_diff,dist_mat),m_diff.T)/2.0)