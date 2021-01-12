#!/usr/bin/env python

import numpy as np
import random
import shutil, sys, os, subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
from scipy.stats import norm
import chaospy as cp
from UQpy.SampleMethods import LHS
from UQpy.Distributions import Uniform
import itertools as it

plt.switch_backend('Agg')

# Constants
kB = 8.617330350e-5
TZ = 21000
nseed = 7536655
np.random.seed(nseed)
qq = open("dictionary.txt","w")

"""
Optimization Model Class
"""
class OptimizationModel:

    temperatureFieldPrefix = 'tem.'
    strainFieldPrefix = 'Exy.'
    stressFieldPrefix = 'tau.'

    def __init__(self, nInitialObs=100, samplingstyle=0, showsampling=True, nTotalObs=120, nSurrEvals=2000, 
                    dtol=0.01,
                    distanceMetric=0, 
                    detectOutliers=False, contamination=0.05, anomoly_algorithm="Local Outlier Factor",
                    kernelIndex=0, kernel_nu=2.5, optimizerRestarts=50, normalize_y=True, 
                    manifoldDistanceMethod=None, isRankConstant=True, rank=None,
                    continuumOutPath='sct_q.out', nContinuumFrames=101, continuumVersion='2.0.0',
                    prefactors=[1,1],
                    decompositionRank=-1, verboseDecomposition=True, rankTolerance=1e-6):

        self.nInitialObs = nInitialObs
        self.nTotalObs = nTotalObs
        self.nSurrEvals = nSurrEvals
        self.dtol = dtol
        self.distanceMetric = distanceMetric
        self.manifoldDistanceMethod = manifoldDistanceMethod
        self.isRankConstant = isRankConstant   
        self.rank = rank
        self.continuumOutPath = continuumOutPath       
        self.continuumVersion = continuumVersion
        self.nContinuumFrames = nContinuumFrames
        self.kernelIndex = kernelIndex
        self.kernel_nu = kernel_nu
        self.optimizerRestarts = optimizerRestarts
        self.normalize_y = normalize_y
        self.detectOutliers = detectOutliers
        self.contamination = contamination
        self.anomoly_algorithm = anomoly_algorithm
        self.decompositionRank = decompositionRank
        self.verboseDecomposition = verboseDecomposition
        self.rankTolerance = rankTolerance
        self.samplingstyle = samplingstyle
        self.showsampling = showsampling
        self.priorkernel = None

        if self.distanceMetric == 0:
            #self.distanceMetric_hc = r"$\Sigma_{\gamma=0}^{0.5}|\tau_\gamma^{MD}-\tau_\gamma^{CM}|$"
            self.distanceMetric_hc = r"$\Sigma | \tau^* (\gamma) - \tau^n (\gamma) |$"

        if self.distanceMetric == 1:
            self.distanceMetric_hc = r"$|\int \tau_{*} d\epsilon_* - \int \tau_n d\epsilon_n|$"
        ## Load continuum model
        #cmd = 'module unload continuum_model'
        #os.system(cmd)
        #cmd = 'module load continuum_model/' + continuumVersion
        #os.system(cmd)

    def trainModel(self, X, Y):

        if self.kernelIndex == 0:
            self.kernel = 1.0*Matern(length_scale=np.ones(np.size(X, 1)), 
                                nu=self.kernel_nu)

        if self.kernelIndex == 1:
            self.kernel = 1.0*RBF(length_scale=np.ones(np.size(X, 1))) + WhiteKernel()

        if self.kernelIndex == 2:
            self.kernel = 1.0*RBF(length_scale=np.ones(np.size(X, 1)),length_scale_bounds=[(0.1,10),(0.1,10),(0.1,10),(0.1,10), \
                      (0.1,10),(0.1,10),(0.1,10),(0.1,10)]) 

        if self.kernelIndex == 3:
            self.kernel = 1.0*RBF(length_scale=np.ones(np.size(X, 1))) + C()

        if self.priorkernel:
            self.kernel = self.priorkernel

        print("\nPrior\n{}\n".format(self.kernel))
        gp = GP(kernel=self.kernel, 
                n_restarts_optimizer=self.optimizerRestarts, 
                normalize_y=self.normalize_y)

        # Toggle outlier removal
        if self.detectOutliers:

            if self.anomoly_algorithm == 'Local Outlier Factor':
                algorithm = LocalOutlierFactor(contamination=self.contamination)

            elif self.anamoly_algorithm == 'One-Class SVM':
                algorithm = svm.OneClassSVM(nu=self.contamination)

            elif self.anamoly_algorithm == 'Isolation Forest':
                algorithm = IsolationForest(behavior='new',
                                            contamination=self.contamination)

            elif self.anamoly_algorithm == 'Robust Covariance':
                algorithm = EllipticEnvelope(contamination=self.contamination)

            else:
                print("Unknown outlier detection algorithm encountered.")
                print("Using complete dataset to train surrogate model.")
                algorithm = None

            if algorithm == None:
                gp.fit(X, Y)

            else:
                print("Removing outliers prior to GPR training.")
                y_out = algorithm.fit_predict(Y[:, np.newaxis])
                gp.fit(X[y_out==1], Y[y_out==1])

        else:
            gp.fit(X, Y)
        
        #qq.write("%s\n" % gp.kernel_)
        print("\nPosterior\n{}\n".format(gp.kernel_))
        self.priorkernel = gp.kernel_        
        return gp

""" 
Parameter Class
"""
class Parameter:
    
    name = None
    value = None
    vmin = -np.inf
    vmax = np.inf
    units_so = None
    units_hc = None
    text_so = None
    text_hc = None
    is_constant = True
    
    def __init__(self, name, value, vmin=-np.inf, vmax=np.inf):

        self.name = name
        self.value = value
        self.vmin = vmin
        self.vmax = vmax

        if self.vmin != -np.inf or self.vmax != np.inf:
            self.is_constant = False

        if self.name == 'beta':
            self.units_so = '1/eV'
            self.units_hc = r'$eV^{-1}$'
            self.text_so = 'beta'
            self.text_hc = r'$\beta$'

        if self.name == 'u0':
            self.units_so = 'eV'
            self.units_hc = 'eV'
            self.text_so = 'u0'
            self.text_hc = r'$u_0$'
        
        if self.name == 'chi_len':
            self.units_so = 'Angstroms'
            self.units_hc = r'$\AA$'
            self.text_so = 'chi_len'
            self.text_hc = r'$\ell_\chi$'

        if self.name == 'c0':
            self.units_so = '--'
            self.units_hc = '--'
            self.text_so = 'c0'
            self.text_hc = r'$c_0$'
        
        if self.name == 'ep':
            self.units_so = '--'
            self.units_hc = '--'
            self.text_so = 'ep'
            self.text_hc = r'$\varepsilon_0$'
        
        if self.name == 's_y':
            self.units_so = 'GPa'
            self.units_hc = 'GPa'
            self.text_so = 's_y'
            self.text_hc = r'$s_y$'

            try:
                assert(self.vmin >= 0)
            except:
                print("Warning! {} cannot be negative. Minimimum set to 0 K.".format(self.name))
                self.vmin = 0 
    
            if self.name == 'chi_inf':
                self.text_so = 'chi_inf'
                self.text_hc = r'$\chi_\infty$'

            if vmin == -np.inf:
                self.vmin = 0
            elif vmin < 0:
                print("Warning! {} should not be negative. Please review value.".format(self.name))
                self.vmin = vmin
            else:
                self.vmin = vmin
   
            if self.vmin == -np.inf:
                self.vmin = 1
            elif self.vmin < 0:
                print("Warning! Volume cannot be less than or equal to 0.")
                print("Volume set to 1 Angstrom.")
                self.vmin = 1

    def set_min(self, value):
        self.is_constant = False
        self.vmin = value
        if self.value < self.vmin:
            self.value = self.vmin

    def set_max(self, value):
        self.is_constant = False
        self.vmax = value
        if self.value > self.vmax:
            self.value = self.vmax


"""
Parameters Class
"""
class Parameters():
    parameters = []
    ndims = None
    variableNames = []

    def __init__(self, parameters):
        self.parameters = parameters

        self.ndims = 0
        for parameter in self.parameters:
            if not parameter.is_constant: 
                self.ndims += 1
                self.variableNames = np.append(self.variableNames, parameter.name)
                        
    def list_parameters(self):
        for parameter in self.parameters:
            if parameter.is_constant:  
                print("{} = {} {} and is constant.".format(
                    parameter.name, parameter.value, parameter.units_so))
            else:
                print("{} = {} {} and varies from {} to {}.".format(
                    parameter.name, parameter.value, parameter.units_so,
                    parameter.vmin, parameter.vmax))

    def getValue(self, name):
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter.value


    def getVariables(self):
        output = []
        for param in self.parameters:
            if not param.is_constant:
                output = np.append(param, output)


    def setValue(self, name, value):
        for parameter in self.parameters:
            if parameter.name == name:
                parameter.value = value


"""
Reference Class
"""
class Reference():

    def __init__(self, sourceDir='/home-3/kkontol1@jhu.edu/CuZr_Ref/',
                    stressFileName='stress_rand2_matlab.txt', stressCol=6, stressDelimiter='\t', stressScaleFactor=10000, maxStrain=0.5,
                    energyFieldPrefix='pe.MD.', strainFieldPrefix='Exy.MD.', nFields=101, fromContinuum=False):

        if fromContinuum:
        
            # Calculate T_inf
            T_ = np.fromfile('reference/tem.100', dtype=np.float32)
            ny = int(T_[0]) + 1
            nx = int(len(T_)/ny)
            T_ = T_.reshape(nx, ny)
            T_ = T_[1:, 1:]
            self.T_inf = np.amax(T_)

            self.strain = np.linspace(0, maxStrain, nFields)
            self.stress = np.zeros(nFields)

            for i in np.arange(0, nFields):
                self.stress[i] = computeContinuumDeviatoricStress(\
                    'reference', i)
         
            self.MD_sourceDir = sourceDir
            self.MD_energyFieldPrefix = energyFieldPrefix 
            self.toughness = np.trapz(self.stress, self.strain)
            self.sourceDir = 'reference/'
            self.temperatureFieldPrefix = 'tem.'
            self.strainFieldPrefix = 'Exy.'
            self.stressFieldPrefix = 'tau.'

        else:
            self.sourceDir = sourceDir
            self.energyFieldPrefix = energyFieldPrefix
            self.strainFieldPrefix = strainFieldPrefix
        

        self.stressDelimiter = stressDelimiter
        self.stressFileName = stressFileName
        self.stressCol = stressCol
        self.maxStrain = maxStrain
        self.stressScaleFactor = stressScaleFactor
        self.stressDelimiter = stressDelimiter
        self.nFields = nFields
        self.fromContinuum = fromContinuum

    def getStressStrainData(self):
        
        stressData = np.loadtxt(self.sourceDir + self.stressFileName, delimiter = self.stressDelimiter)
        self.stress = -1*stressData[:, self.stressCol]/self.stressScaleFactor
        self.strain = np.linspace(0, self.maxStrain, len(self.stress))
        del stressData
        return self.strain, self.stress

    def getFieldValues(self, fileName, step=0, skiprows=1):
        if self.fromContinuum:
            return np.loadtxt(self.MD_sourceDir + fileName, skiprows = skiprows)
        else:
            return np.loadtxt(self.sourceDir + fileName, skiprows = skiprows)


"""
Observation Class
"""
class Observation:
    
    def __init__(self, parameters, reference, optimizationModel): 

        self.parameters = parameters
        self.reference = reference
        self.optimizationModel = optimizationModel

    def computeDistance(self):
        
        if os.path.isdir(self.optimizationModel.continuumOutPath): 
            shutil.rmtree(self.optimizationModel.continuumOutPath)
    
        if os.path.exists(self.optimizationModel.continuumOutPath):
            command = ['rm', '-r', self.optimizationModel.continuumOutPath]
            subprocess.run(command)
   
        if self.reference.fromContinuum:

            T_ = np.fromfile('reference/tem.100', dtype=np.float32)
            ny = int(T_[0]) + 1
            nx = int(len(T_)/ny)
            T_ = T_.reshape(nx, ny)
            T_ = T_[1:, 1:]
            T_inf = np.amax(T_)
            
            PE_0 = self.reference.MD_sourceDir + self.reference.MD_energyFieldPrefix + '0'
            PE_f = np.amax(self.reference.getFieldValues('pe.MD.100'))
            T_inf = self.parameters.getValue('beta')*(PE_f-self.parameters.getValue('u0'))*TZ
            
        else: 
            PE_0 = self.reference.sourceDir + self.reference.energyFieldPrefix + '0'

            # Calculate final steady-state effective temperature
            PE_f = np.amax(self.reference.getFieldValues('pe.MD.100'))
            T_inf = self.parameters.getValue('beta')*(PE_f-self.parameters.getValue('u0'))*TZ
    
            # Neglect b,u0 values that result in a negative (or another threshold) value of the final MD eff. temp. field
            PE_MD_final = self.reference.getFieldValues('pe.MD.100')
            T_MD_final = self.parameters.getValue('beta')*(PE_MD_final-self.parameters.getValue('u0'))*TZ
           
            #  --  Neglect b,u0 values that result in a negative (or another threshold) value of the initial eff. temp. field 
            # PE_MD_initial = self.reference.getFieldValues('pe.MD.0')
            # T_initial = self.parameters.getValue('beta')*(PE_MD_initial-self.parameters.getValue('u0'))*TZ # used to initialize CM
            
            # Condition
            if np.any(T_MD_final<20)==True: 

                self.distance = np.nan
                return self.distance

        
        self.parameters.setValue('chi_inf', T_inf)
        
        command = ['shear_energy_Adam3', 'qs', PE_0,
                    '{}'.format(self.parameters.getValue('beta')), 
                    '{}'.format(self.parameters.getValue('u0')),
                    'chi_inf', '{}'.format(T_inf),
                    'chi_len', '{}'.format(self.parameters.getValue('chi_len')),
                    'c0', '{}'.format(self.parameters.getValue('c0')),
                    'ep', '{}'.format(self.parameters.getValue('ep')),
                    's_y', '{}'.format(self.parameters.getValue('s_y'))]

        try:    
            subprocess.run(command, timeout=360)

            om = self.optimizationModel
            finalCMStrainField = om.continuumOutPath + '/' + om.strainFieldPrefix + '{}'.format(om.nContinuumFrames-1)
    
            if os.path.isfile(finalCMStrainField):
                if om.distanceMetric == 0:
                    # Sum of Difference in Stress Magnitudes 
                    strain_cm = np.linspace(0, self.reference.maxStrain, om.nContinuumFrames)
                    stress_cm = np.zeros(strain_cm.shape)
                    distance_i = np.zeros(strain_cm.shape)
    
                    for i in np.arange(0, om.nContinuumFrames):

                        stress_cm[i] = computeContinuumDeviatoricStress(\
                            om.continuumOutPath, i)

                        ix = np.argmin(np.abs(self.reference.strain-strain_cm[i]))
                        distance_i[i] = np.abs(self.reference.stress[ix]-stress_cm[i])
                    
                    self.distance = np.sum(distance_i)                   
                    print("\nDistance from magnitude of stress is {:.4f}\n".format(self.distance))

                elif om.distanceMetric == 1:
                    # Difference in toughness
                    strain = np.linspace(0, self.reference.maxStrain, om.nContinuumFrames)
                    stress = np.zeros(strain.shape)
                    
                    for i in np.arange(0, om.nContinuumFrames):
                        stress[i] = computeContinuumDeviatoricStress(\
                            om.continuumOutPath, i)
                
                    toughness = np.trapz(stress, strain)
                    self.distance = np.abs(self.reference.toughness-toughness)
                    print("\nDistance from magnitude of difference in toughness is {:.4f}.\n".format(self.distance))
 
                    plt.figure()
                    titletext = r'$|\Delta G(*,n)|$' + ' = {:4f} '.format(self.distance) + r'eV$\cdot\AA^{-3}$'
                    plt.title(titletext) 
                    #plt.title('Toughness G = {:.4f}'.format(np.abs(toughness-self.reference.toughness)))
                    plt.plot(self.reference.strain, self.reference.stress, label='Ref')
                    plt.plot(strain, stress, label='Obs')
                    plt.legend()
                    plt.savefig('toughness.png')

                elif om.distanceMetric == 2:
                    # Average Grassmann distance between strain fields
                    distance_ = np.zeros(om.nContinuumFrames)

                    # Compare strain fields
                    for i in np.arange(0, om.nContinuumFrames): 
                        # Observation, O
                        fileName = om.continuumOutPath + '/' + om.strainFieldPrefix + '{}'.format(i)
                        strain_ = np.fromfile(fileName, dtype=np.float32)
                        ny = int(strain_[0]) + 1
                        nx = int(len(strain_)/ny)
                        strain_ = strain_.reshape(nx, ny)
                        strain_ = strain_[1:, 1:]

                        # Load reference strain field
                        if self.reference.fromContinuum:
                            fileName = self.reference.sourceDir + \
                                        self.reference.strainFieldPrefix + '{}'.format(i)
                            strainR_ = np.fromfile(fileName, dtype=np.float32)
                            ny = int(strainR_[0]) + 1
                            nx = int(len(strainR_)/ny)
                            strainR_ = strainR_.reshape(nx, ny)
                            strainR_ = strainR_[1:, 1:]
                            
                        distance_[i], self.r0, self.rR = compareMatrices(strainR_, strain_, \
                                stol=om.rankTolerance, constantrank=om.decompositionRank, \
                                verbose=om.verboseDecomposition)
    
                    #self.distance = np.sum(distance_) 
                    self.distance = np.mean(distance_) 

                elif om.distanceMetric == 3:
                    """
                    Distance is the average distance between observation and reference temperature 
                    fields for all frames.
                    """
                    distance_ = np.zeros(om.nContinuumFrames)
                    
                    # Compare strain fields
                    for i in np.arange(0, om.nContinuumFrames): 
                        # Observation, O
                        fileName = om.continuumOutPath + '/' + om.temperatureFieldPrefix + '{}'.format(i)
                        B_ = np.fromfile(fileName, dtype=np.float32)
                        ny = int(B_[0]) + 1
                        nx = int(len(B_)/ny)
                        B_ = B_.reshape(nx, ny)
                        B_ = B_[1:, 1:]

                        # Load reference strain field
                        if self.reference.fromContinuum:
                            fileName = self.reference.sourceDir + \
                                        self.reference.temperatureFieldPrefix + '{}'.format(i)
                            A_ = np.fromfile(fileName, dtype=np.float32)
                            ny = int(A_[0]) + 1
                            nx = int(len(A_)/ny)
                            A_ = A_.reshape(nx, ny)
                            A_ = A_[1:, 1:]
                        else:
                            fileName = self.reference.sourceDir + \
                                        self.reference.energyFieldPrefix + \
                                        '{}'.format(i)
                            A_ = np.loadtxt(fileName, skiprows=1)
                            A_ = self.parameters.getValue('beta')*(A_-self.parameters.getValue('u0'))*21000
                                                        

                        distance_[i], self.r0, self.rR = compareMatrices(A_, B_, \
                                stol=om.rankTolerance, constantrank=om.decompositionRank, \
                                verbose=om.verboseDecomposition)

                    self.distance = np.mean(distance_)
                
                elif om.distanceMetric == 4:
                    """
                    Distance metric combines strain fields at 9 global strain
                        values into a single 3*ny x 3*ny matrix, then uses 
                        computes the Grassmannian distance/
                    """
                    print('Comparing strain fields at specific strain values.')

                    # Compare frames N frames in single go
                    rank_ = [2, 4, 5, 5, 10, 10, 14, 15, 16]                    
                    strain = [0.05, 0.06, 0.08, 0.09, 0.10, 0.11, 0.12, 0.35, 0.5]
                    gamOBS = np.linspace(0, 0.5, om.nContinuumFrames)
    
                    A = []
                    B = []
                
                    for val in strain:
    
                        if self.reference.fromContinuum:

                            gamREF = np.linspace(0, self.reference.maxStrain, self.reference.nFields)                
                            idRef = np.argmin(np.abs(gamREF-val))
                            Aname = self.reference.sourceDir + \
                                    self.reference.strainFieldPrefix + \
                                    '{}'.format(idRef)

                            A_ = np.fromfile(Aname, dtype=np.float32)
                            nyA = int(A_[0]) + 1
                            nxA = int(len(A_)/nyA)
                            A_ = A_.reshape(nxA, nyA)
                            A_ = A_[1:, 1:]
                            A.append(A_)                        

                        else:

                            gamREF = np.linspace(0, self.reference.maxStrain, self.reference.nFields)
                            idREF = np.argmin(np.abs(gamREF-val))

                            Aname = self.reference.sourceDir + '/' + \
                                    self.reference.strainFieldPrefix + \
                                    '{}'.format(idREF)

                            A_ = np.loadtxt(Aname, skiprows=1)
                            A.append(A_)                        

                            
                        # Get indices where strain nearest ref, obs strain
                        idOBS = np.argmin(np.abs(gamOBS-val))
                        
                        Bname = om.continuumOutPath + '/' + \
                                om.strainFieldPrefix + \
                                '{}'.format(idOBS)

                        B_ = np.fromfile(Bname, dtype=np.float32)
                        nyB = int(B_[0]) + 1
                        nxB = int(len(B_)/nyB)
                        B_ = B_.reshape(nxB, nyB)
                        B_ = 2.*B_[1:, 1:]
                        B.append(B_)

                    distance_temp = np.zeros(len(A))
                    r0_temp = np.zeros(len(A))
                    rR_temp = np.zeros(len(A))
                    for k in range(len(A)):
                        distance_temp[k], r0_temp[k], rR_temp[k] = compareMatrices(A[k], B[k], rank=rank_[k], \
                            stol=om.rankTolerance, constantrank=om.decompositionRank, \
                            verbose=om.verboseDecomposition)
                    distance1 = np.mean(distance_temp)   # Grassmann distance of strain fields
                    self.r0 = r0_temp
                    self.rR = rR_temp
                    

                    # Also, Sum of Difference in Stress Magnitudes 
                    strain_cm = np.linspace(0, self.reference.maxStrain, om.nContinuumFrames)
                    stress_cm = np.zeros(strain_cm.shape)
                    distance_i = np.zeros(strain_cm.shape)
            
                    fileName = self.reference.sourceDir + '/' + self.reference.stressFileName
                    tau_MD = np.loadtxt(fileName, delimiter="\t")
                    tau_MD = -1*tau_MD[:, self.reference.stressCol]/10000.
                    strain_MD = np.linspace(0, self.reference.maxStrain, len(tau_MD))

                    
                    for i in range(0, om.nContinuumFrames):

                        stress_cm[i] = self.parameters.getValue('s_y')*computeContinuumDeviatoricStress(\
                            om.continuumOutPath, i)
                        
                        ix = np.argmin(np.abs(strain_MD-strain_cm[i]))
                        distance_i[i] = np.abs(tau_MD[ix]-stress_cm[i])
                        
                    distance_i[15:28] *= 5
                    distance2 = np.sum(distance_i)/np.count_nonzero(distance_i)                   
                    
                    # print the distances in a .txt file
                    save_dist = np.array([distance1,distance2]).reshape((1,-1))
                    np.savetxt(qq,save_dist)                    
 
                    # Combined average Grassmann distance at 9 strain field snapshots and 
                    # average difference of stress magnitudes at all 100 frames                     
                    self.distance = (1*distance1 + 3*distance2)/4 
                    print('Distance is {}.'.format(self.distance))

                else: 
                    # NULL Case... will fill in later
                    print("\nNo distance computed.\n")
                    self.distance = 23
            else:
                # NAN distance
                print('\nDistance of NAN.\n')
                self.distance = np.nan

        except subprocess.TimeoutExpired:
            self.distance = np.nan

        return self.distance 
 
    def getContinuumField(self, fieldName):
        field = np.fromfile(self.optimizationModel.continuumOutPath + '/' + fieldName, dtype=np.float32)
        ny = int(field[0]) + 1
        nx = int(len(field)/ny)
        field = field.reshape(nx, ny)
        field = field[1:, 1:]
        return field

    def changeContinuumModel(self):
        pass



"""
K-nn function
"""
def knn(X, a, k):
    dist = [np.linalg.norm(X[a,:] - X[i,:]) for i in range(X.shape[0])]
    ind = np.argsort(dist)
    dist.sort()
    knn = ind[:k+1]
    other = np.zeros((X.shape[0]-k))
    other = ind[k+1:]
    other = other.astype(int)
    d_max = np.mean(dist)
    return knn, other, d_max


"""
Optimization Function
"""
def ego(optimizationModel, reference, initParams, checkPrevious=True, 
            prevFileName='ego_data.npz', outFileName='ego_data.npz', normalize=True):

    om = optimizationModel

    if checkPrevious and os.path.isfile(prevFileName):
        priorData = np.load(prevFileName)

        X = priorData['X']
        Y = priorData['Y']
        Y_best = priorData['Y_best']
        #parameters = priorData['parameters']
        #param_names = parameters.variableNames

    else:

        bounds = []
        for param in initParams.parameters:
            if not param.is_constant:
                bounds.append([param.vmin, param.vmax])

        Xe = np.array(np.meshgrid(*bounds)).T.reshape(-1, initParams.ndims)

        if om.samplingstyle == 0:
            # Uniform grid
            N = om.nInitialObs
            d = initParams.ndims
            n = int(np.ceil(N**(1/d)))

            A = np.linspace(0, 1, n)
            AA = np.zeros((n, d))

            i = 0
            for param in initParams.parameters:
                if not param.is_constant:
                    AA[:, i] = A*np.abs(param.vmin-param.vmax)+param.vmin
                    i += 1
            *aa, = np.meshgrid(*AA.T)

            X = np.zeros((aa[0].size, d))
            for i in range(0, d):
                X[:,i] = aa[i].flatten()

            X = np.append(X, Xe, axis=0)
            om.nInitialObs = X[:,0].size

            Y=[]

        elif om.samplingstyle == 1:
            # Sobol sequence
            print('Sampling from a Sobol sequence.') 
            A = []
            for param in initParams.parameters:
                if not param.is_constant:
                    A.append(cp.Uniform(param.vmin, param.vmax))
            distribution = cp.J(*A)
            X = distribution.sample(om.nInitialObs, \
                    rule='S')
            X = X.T
            #X = np.append(X, Xe, axis=0)
            print('Dimensions of X are:',X.shape)
            Y = []
            del A, distribution
            
        elif om.samplingstyle == 2:
            # Halton sequence
            print('Sampling from a Halton sequence.') 
            A = []
            for param in initParams.parameters:
                if not param.is_constant:
                    A.append(cp.Uniform(param.vmin, param.vmax))
            distribution = cp.J(*A)
            X = distribution.sample(om.nInitialObs, \
 
                  rule='H')
            X = X.T
            #X = np.append(X, Xe, axis=0)
            Y = []
            del A, distribution

        elif om.samplingstyle == 3:
            # Latin Hypercube Sampling
            print('Latin Hypercube Sampling.')
            A = []
            seed = 541122
            for param in initParams.parameters:
                if not param.is_constant:
                    A.append(Uniform(loc=param.vmin, scale=param.vmax-param.vmin))
            x1 = LHS(dist_object=A, nsamples=om.nInitialObs, random_state=np.random.RandomState(seed), verbose=False)
            X = x1.samples
            print('Dimensions of X matrix are:',X.shape)
            Y = []
            del A, x1

        else:
            # Random sampling
            X = np.zeros((optimizationModel.nInitialObs, initParams.ndims))
            Y = []

            i = 0
            param_names = []
            for param in initParams.parameters:
                if not param.is_constant:
                    X[:, i] = np.random.uniform(param.vmin, param.vmax, \
                        optimizationModel.nInitialObs)
                    param_names = np.append(param_names, param.name)
                    i += 1
            #X = np.append(X, Xe, axis=0)

        if om.showsampling:
            fig = plt.figure()
            if initParams.ndims == 1:
                plt.plot(X)
            elif initParams.ndims == 2:
                plt.scatter(*X.T)
            elif initParams.ndims == 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(*X.T)
            else:
                print("Too many dimensions to visualize.")
            plt.savefig('initial_samples.png')
        
        Y_best = np.inf

    hold = 0
    book = [X]  # book-keeping the X's for each GP
    distances = [1e10]
    n_nearest_neighbors = 200
    
    while len(Y) < optimizationModel.nTotalObs and Y_best > optimizationModel.dtol:
        
        iObs = len(Y)
        print("\nObservation {} of {}\n".format(iObs + 1, optimizationModel.nTotalObs))

        if iObs < optimizationModel.nInitialObs:

            # Update parameter value
            ip = 0
            for param in initParams.parameters:
                if not param.is_constant:
                    param.value = X[iObs, ip]
                    ip += 1
            
            obs = Observation(initParams, reference, optimizationModel)
            dist = obs.computeDistance()
            Y = np.append(Y, dist)

        else:

            # Remove NANs
            Xs = X[~np.isnan(Y),:]
            Ys = Y[~np.isnan(Y)]

            # Generate surrogate points 
            # Sobol sequence
            A = []
            for param in initParams.parameters:
                if not param.is_constant:
                    A.append(cp.Uniform(param.vmin, param.vmax))
            distribution = cp.J(*A)
            x3 = distribution.sample(om.nSurrEvals, \
                    rule='S')
            X_surr = x3.T
            

            # Scaled data indicated by underscore "_"
            scaler = StandardScaler()
            if Xs.ndim == 1: 
                Xs_ = scaler.fit_transform(Xs.reshape(-1,1))
                X_ = scaler.transform(X.reshape(-1,1))
                X_surr_ = scaler.transform(X_surr.reshape(-1,1))

            else: 
                
                if hold == 0:
                    Xs_ = scaler.fit_transform(Xs)
                    X_ = scaler.transform(X)
                    X_surr_ = scaler.transform(X_surr)

                    # Train surrogate model using scaled or unscaled inputs
                    kernel = 1.0*RBF(length_scale=np.ones(np.size(Xs, 1))) + C()
                    gp = [GP(kernel=kernel, n_restarts_optimizer=10, random_state=0).fit(Xs_, Ys)]
                    Y_surr, s2 = gp[0].predict(X_surr_, return_std=True)

            if hold > 0:
                print('Hereeeeeeeeee')
                print(Xs.shape, Xs.shape[0]-1, n_nearest_neighbors)
                x_knn, x_other, d_max = knn(Xs, Xs.shape[0]-1, n_nearest_neighbors)
                X_knn_ = scaler.fit_transform(Xs[x_knn])
                kernel = 1.0*RBF(length_scale=np.ones(np.size(Xs, 1))) + C()
                # save X's and max distance to use for surrogate points
                book.append(Xs[x_knn])
                distances.append(d_max)
                # crete new GP regressor with scaled data
                gp.append(GP(kernel=kernel, n_restarts_optimizer=10, random_state=0).fit(X_knn_, Ys[x_knn]))               
                
                X_ = scaler.transform(X)
                X_surr_ = scaler.transform(X_surr) 


            if hold > 0:
                Y_surr, s2 = [], []
                for j in range(om.nSurrEvals):
                    # check if X_surr[i] lies inside the convex null of gp1, if not check for gp0 ...etc
                    d = np.linalg.norm(Xs[Xs.shape[0]-1,:] - X_surr[j,:])
                    for k in range(1,hold+2):
                        if d < distances[hold+1-k]:
                            y, s = gp[hold+1-k].predict(X_surr_[j,:].reshape(1, -1), return_std=True)
                            Y_surr.append(y)
                            s2.append(s)
                            break


            Y_surr = np.array(Y_surr)

            # Compute the expected improvement EI
            a = np.nanmin(Y) - Y_surr
            b = np.sqrt(s2)

            a = a.flatten()
            b = b.flatten()    

            EI = a*norm.cdf(np.divide(a,b)) + b*norm.pdf(np.divide(a,b))

            if len(Y[np.isnan(Y)]) > 0:
                nn = []
                for irow, row in enumerate(X_surr_):
                    nn.append(np.argmin(np.linalg.norm(row-X_, axis=1)))
    
                # Set expected improvement values to 0 where neihboring
                # observation point is NAN
                for i_nn, v_nn in enumerate(nn):
                    if np.isnan(Y[int(v_nn)]):     
                        EI[i_nn] = 0

            J = np.argmax(EI)
            
            print('\nNext guess: {}'.format(X_surr[J,:]))

            # Update parameter values
            ip = 0
            for param in initParams.parameters:
                if not param.is_constant:
                    param.value = X_surr[J, ip] 
                    ip += 1

            # Generate new observation object, compute distance and 
            #   append to the data set inputs (X) and outputs (Y).
            obs = Observation(initParams, reference, optimizationModel)
            dist = obs.computeDistance()
            Y = np.append(Y, dist)
            X = np.concatenate((X, X_surr[J, :].reshape(1, initParams.ndims)))
            print(Ys)
            # Visualize convergence

            plotConvergence(X, Y, initParams, om)
            hold = hold + 1

        if dist < Y_best:

            if len(Y[~np.isnan(Y)]) <= 1:
                
                T0_CM_curr = obs.getContinuumField('tem.0')
                Tf_CM_curr = obs.getContinuumField('tem.100')
                strain_CM_curr = obs.getContinuumField('Exy.100')
                strain_CM_curr = 2.*strain_CM_curr
                i_best_curr = np.nanargmin(Y)

                T0_CM_prev = T0_CM_curr
                Tf_CM_prev = Tf_CM_curr
                strain_CM_prev = strain_CM_curr
                i_best_prev = i_best_curr

            else:

                priorData = np.load(prevFileName)
                T0_CM_prev = priorData['T0_CM_curr']
                Tf_CM_prev = priorData['Tf_CM_curr']
                strain_CM_prev = priorData['strain_CM_curr']
                i_best_prev = priorData['i_best_curr']

                T0_CM_curr = obs.getContinuumField('tem.0')
                Tf_CM_curr = obs.getContinuumField('tem.100')
                strain_CM_curr = obs.getContinuumField('Exy.100')
                strain_CM_curr = 2.*strain_CM_curr
                i_best_curr = np.nanargmin(Y)

            Y_best = dist

            np.savez(outFileName, X = X, Y = Y, Y_best = Y_best, \
                    i_best_curr = i_best_curr, i_best_prev = i_best_prev, \
                    T0_CM_curr = T0_CM_curr, Tf_CM_curr = Tf_CM_curr, strain_CM_curr = strain_CM_curr, \
                    T0_CM_prev = T0_CM_prev, Tf_CM_prev = Tf_CM_prev, strain_CM_prev = strain_CM_prev)

            # Plot temperature, strain field comparison between reference and current observation
            plotComparison(initParams, optimizationModel, reference, X, Y)

        else:

            if len(Y[~np.isnan(Y)]) == 0:

                np.savez(outFileName, X = X, Y = Y, Y_best = Y_best)

            else:

                """
                    If current distance does not offer improvement, load all prior data,
                    and save along with updated 'X' and 'Y'.
                """

                priorData = np.load(prevFileName)

                Y_best = priorData['Y_best']
                T0_CM_prev = priorData['T0_CM_prev']
                Tf_CM_prev = priorData['Tf_CM_prev']
                strain_CM_prev = priorData['strain_CM_prev']
                i_best_prev = priorData['i_best_prev']

                T0_CM_curr = priorData['T0_CM_curr']
                Tf_CM_curr = priorData['Tf_CM_curr']
                strain_CM_curr = priorData['strain_CM_curr']
                strain_CM_curr = priorData['strain_CM_curr']
                i_best_curr = priorData['i_best_curr']
            
                np.savez(outFileName, X = X, Y = Y, Y_best = Y_best, \
                        i_best_curr = i_best_curr, i_best_prev = i_best_prev, \
                        T0_CM_curr = T0_CM_curr, Tf_CM_curr = Tf_CM_curr, strain_CM_curr = strain_CM_curr, \
                        T0_CM_prev = T0_CM_prev, Tf_CM_prev = Tf_CM_prev, strain_CM_prev = strain_CM_prev)


                 
    return X, Y

def plotComparison(parameters, optimizationModel, reference, X, Y):
    """
    This function generates a figure with the following subplots:
    
        (1) Stress vs. strain for reference and current observation,
        (2) Initial effective temperature (Teff) field contour with colorbar,
        (3) Final observation Teff field contour with colorbar,
        (4) Final observation strain (Exy) field contour with colorbar,
        (5) Final reference Teff field contour with colorbar,
        (6) Final reference Exy field contour with colorbar.

    Note: All colorbar ranges are independent. Contours use grayscale 
            with black corresponding to largest values. Parameter values are shown
            with respective units.

    """

    fig = plt.figure(figsize=(11, 8.5))
    
    textStr = '\n'.join((
        r'$\beta$ = %.4f eV$^{-1}$' % (parameters.getValue('beta'), ),
        r'$u_0$ = %.4f eV' % (parameters.getValue('u0'), ),
        r'$T_\infty$ = %d K' % (parameters.getValue('chi_inf'), ),
        r'$l_\chi$ = %.3f $\AA$' % (parameters.getValue('chi_len'), ),
        r'$c_0$ = %.3f --' % (parameters.getValue('c0'), ),
        r'$\epsilon_0$ = %.3f --' % (parameters.getValue('ep'), ),
        r'$s_y$ = %.3f GPa' % (parameters.getValue('s_y'), )))


    ax = plt.subplot(334)
    ax.axis('off')
    ax.text(0, 0.5, textStr, verticalalignment='center')

    # Plot initial CM effective temperature
    fileName = optimizationModel.continuumOutPath + '/' + optimizationModel.temperatureFieldPrefix + '0'
    T0 = np.fromfile(fileName, dtype=np.float32)
    ny = int(T0[0]) + 1
    nx = int(len(T0)/ny)
    T0 = T0.reshape(nx, ny)
    T0 = T0[1:, 1:]

    plt.subplot(333)
    plt.title(r'$T_{0}$')
    plt.contourf(T0, cmap='viridis')
    plt.colorbar()
    plt.axis('off')

    # Plot Final CM effective temperature
    fileName = optimizationModel.continuumOutPath + '/' + optimizationModel.temperatureFieldPrefix + '{}'.format(int(optimizationModel.nContinuumFrames-1))
    Tf_CM = np.fromfile(fileName, dtype=np.float32)
    Tf_CM = Tf_CM.reshape(nx, ny)
    Tf_CM = Tf_CM[1:, 1:]

    plt.subplot(335)
    plt.title(r'${T_{f}}^{CM}$')
    plt.contourf(Tf_CM, cmap='viridis')
    plt.colorbar()
    plt.axis('off')

    # Plot Final CM strain field
    fileName = optimizationModel.continuumOutPath + '/' + optimizationModel.strainFieldPrefix + '{}'.format(int(optimizationModel.nContinuumFrames-1))
    Exy_CM = np.fromfile(fileName, dtype=np.float32)
    Exy_CM = Exy_CM.reshape(nx, ny)
    Exy_CM = 2*Exy_CM[1:, 1:]

    plt.subplot(336)
    plt.title(r'${\Gamma_{f}}^{CM}$')
    plt.contourf(Exy_CM, np.linspace(0, 1.6, 9), cmap='viridis')
    plt.colorbar()
    plt.axis('off')

    if reference.fromContinuum:

        # Plot Stress-Strain
        strain_Ref = np.linspace(0, reference.maxStrain, reference.nFields)
        strain_CM = np.linspace(0, reference.maxStrain, optimizationModel.nContinuumFrames)
        tau_Ref = np.zeros(strain_Ref.shape)
        tau_CM = np.zeros(strain_CM.shape)
    
        for it in np.arange(0, optimizationModel.nContinuumFrames):

            tau_CM[it] = parameters.getValue('s_y')*computeContinuumDeviatoricStress(\
                optimizationModel.continuumOutPath, it)

            tau_Ref[it] = parameters.getValue('s_y')*computeContinuumDeviatoricStress(\
                reference.sourceDir, it)
            

        plt.subplot(332)
        plt.title(r'$\tau(\gamma)$')
        plt.plot(strain_CM, tau_CM, 'k--', label='CM')
        plt.plot(strain_Ref, tau_Ref, 'k:', label='MD')
        plt.xlabel(r'$\gamma$ [--]')
        plt.ylabel(r'$\tau$ [GPa]')
        plt.legend()

        # Plot Final CM effective Temperature
        fileName = reference.sourceDir + '/' + reference.temperatureFieldPrefix + '{}'.format(int(reference.nFields-1))
        Tf_Ref = np.fromfile(fileName, dtype=np.float32)
        Tf_Ref = Tf_Ref.reshape(nx, ny)
        Tf_Ref = Tf_Ref[1:, 1:]

        plt.subplot(338)
        plt.title(r'${T_{f}}^{Ref}$')
        plt.contourf(Tf_Ref, cmap='viridis')
        plt.colorbar()
        plt.axis('off')

        # Plot Final CM strain field
        fileName = reference.sourceDir + '/' + reference.strainFieldPrefix + '{}'.format(int(reference.nFields-1))
        Exy_Ref = np.fromfile(fileName, dtype=np.float32)
        Exy_Ref = Exy_Ref.reshape(nx, ny)
        Exy_Ref = Exy_Ref[1:, 1:]

        plt.subplot(339)
        plt.title(r'${\Gamma_{f}}^{Ref}$')
        plt.contourf(Exy_Ref, cmap='viridis')
        plt.colorbar()
        plt.axis('off')

    else:
       
        # Plot Stress-Strain
        fileName = reference.sourceDir + '/' + reference.stressFileName
        tau_MD = np.loadtxt(fileName, delimiter="\t")
        tau_MD = -1*tau_MD[:, reference.stressCol]/10000.
        strain_MD = np.linspace(0, reference.maxStrain, len(tau_MD))

        strain_CM = np.linspace(0, reference.maxStrain, optimizationModel.nContinuumFrames)
        tau_CM = np.zeros(strain_CM.shape)
    

        for it in np.arange(0, optimizationModel.nContinuumFrames):

            tau_CM[it] = parameters.getValue('s_y')*computeContinuumDeviatoricStress(\
                optimizationModel.continuumOutPath, it)

        plt.subplot(332)
        plt.title(r'$\tau(\gamma)$')
        plt.plot(strain_CM, tau_CM, 'k--', label='CM')
        plt.plot(strain_MD, tau_MD, 'k:', label='MD')
        plt.xlabel(r'$\gamma$ [--]')
        plt.ylabel(r'$\tau$ [GPa]')
        plt.legend()
 
        # Plot Final MD reference effective Temperature Field
        fileName = reference.sourceDir + '/' + reference.energyFieldPrefix + '{}'.format(int(reference.nFields-1))
        Uf_MD = np.loadtxt(fileName, skiprows=1)
        Tf_Ref = parameters.getValue('beta')*(Uf_MD-parameters.getValue('u0'))*21000

        plt.subplot(338)
        plt.title(r'${T_{f}}^{Ref}$')
        plt.contourf(Tf_Ref, cmap='viridis')
        plt.colorbar()
        plt.axis('off')

        # Plot Final MD strain field
        fileName = reference.sourceDir + '/' + reference.strainFieldPrefix + '{}'.format(int(reference.nFields-1))
        Exy_Ref = np.loadtxt(fileName, skiprows=1)

        plt.subplot(339)
        plt.title(r'${\Gamma_{f}}^{Ref}$')
        plt.contourf(Exy_Ref, cmap='viridis')
        plt.colorbar()
        plt.axis('off')

    fig.tight_layout()
    figName = 'Response.{:02d}.png'.format(int(len(Y)))
    plt.savefig(figName)

def plotConvergence(X, Y, parameters, optimizationModel, nPanes=4, targetVals={}):

    nVars = parameters.ndims
    nFigs = int(np.ceil(nVars/nPanes))
    varNames = parameters.variableNames

    iVar = np.arange(0, nVars)
    figNums = np.floor(iVar/nPanes)
    iMin = np.nanargmin(Y)
    nObs = np.arange(1, len(Y)+1)
    
    # Running minimum of distance and corresponding index of nearest occurrence
    Yi = np.copy(Y)
    Yi[np.isnan(Yi)] = np.inf
    Y_rmin = np.minimum.accumulate(Yi)
    Y_list = list(Yi)
    i_rmin = [Y_list.index(item) for item in Y_rmin]

    plt.figure(figsize=(8.5, 3))
    plt.plot(nObs[:optimizationModel.nInitialObs], Y[:optimizationModel.nInitialObs], '.', label='Initial samples') 
    plt.plot(nObs[optimizationModel.nInitialObs:], Y[optimizationModel.nInitialObs:], '.', color='darkorange', label='New samples')
    plt.plot(nObs, Y_rmin, color='black', label='Current best')
    plt.plot(iMin+1, Y[iMin], color='red', marker='s', markerfacecolor='none', label='Optimal')
    plt.axvspan(1, optimizationModel.nInitialObs, color='black', alpha=0.04)
    Ynan = nObs[np.isnan(Y)]
    #for anan in Ynan:
    #    plt.axvline(anan, linestyle='--', color='red')
    plt.xlabel('n')
    plt.ylabel('d')
    plt.legend()
    plt.xlim(left=1)
    plt.tight_layout()
    plt.savefig(fname='Convergence-0.png')
    plt.close()

    for iFig in np.arange(0, nFigs):
        fig = plt.figure(figsize=(8.5, 11))
        ip = 0
        iSubplot = 0
        for param in parameters.parameters:
            if not param.is_constant:

                if np.floor(ip/nPanes) == iFig:
                    plt.subplot(nPanes, 1, iSubplot+1)
                    
                    if param.name in targetVals:
                        plt.axhline(y=targetVals[param.name], color='black', \
                            linestyle='-', linewidth=2)        
                    
                    plt.plot(nObs[:optimizationModel.nInitialObs], X[:optimizationModel.nInitialObs, ip], '.', \
                            label='Initial samples') 
                    plt.plot(nObs[optimizationModel.nInitialObs:], X[optimizationModel.nInitialObs:, ip], '.', \
                            color='darkorange', label='New samples')
                    
                    # Plot running best line, current best point
                    plt.plot(nObs, X[i_rmin, ip], color='black', \
                            linestyle='-', linewidth=2, label='Current best')
                    plt.plot(iMin+1, X[iMin, ip], color='red', marker='s', markerfacecolor='none', markersize=8, label='Optimal')

                    #plt.axhline(X[iMin, ip], linestyle='--', color='red')
                    plt.axvspan(1, optimizationModel.nInitialObs, color='black', alpha=0.04)
                    #plt.axvline(optimizationModel.nInitialObs, linestyle='--', color='black')
                    plt.ylabel(param.text_hc)
                    plt.xlabel('n')
                    plt.legend()
                    plt.xlim(left=1)
                    print("{} is in figure {}, subplot {}.".format(param.name, iFig, iSubplot))
                    iSubplot += 1
                ip += 1

        figName = 'Convergence-{}.png'.format(iFig+1)
        plt.tight_layout()
        plt.savefig(fname=figName)
        plt.close()

def compareMatrices(A, B, rank=32, stol=1e-4, method=1, \
                    scalefactor=1.0, rankreduction=True, \
                    constantrank=2, verbose=False):

    # Compute SVD of A, B:
    UA, SA, VA = np.linalg.svd(A*scalefactor, full_matrices=True)
    UB, SB, VB = np.linalg.svd(B*scalefactor, full_matrices=True)

    # Diagonalize eigenvalue arrays SA, SB:
    SA = np.diag(SA)
    SB = np.diag(SB)

    # Determine ranks of A, B:
    if constantrank == -1:
        # Rank computed to the desired eigenvalue tolerance
        rA = np.linalg.matrix_rank(SA, tol=stol)
        rB = np.linalg.matrix_rank(SB, tol=stol)
        r = np.amin([rA, rB])

    elif constantrank == 0:
        # Rank computed using smallest rank of inputs
        rA = np.linalg.matrix_rank(SA)
        rB = np.linalg.matrix_rank(SB)
        r = np.amin([rA, rB])

    elif constantrank == 1:
        # Keep the max rank of A,B
        rA = np.linalg.matrix_rank(SA, tol=stol)
        rB = np.linalg.matrix_rank(SB, tol=stol)
        r = np.amax([rA, rB])
    
    elif constantrank == 2:
        rA, rB, r = rank, rank, rank

    else:
        # Rank computed from constant rank
        # *** Need to add handling case for when requested rank exceeds input value
        if isinstance(constantrank, int):
            rA = int(constantrank)
            rB = int(constantrank)
            r = int(constantrank)
        else:
            errtxt1 = "\nWarning: Unable to deduce rank from non-integer input."
            errtxt2 = "\nUsing minimum rank instead."
            print(errtxt1 + errtxt2)
            
            rA = np.linalg.matrix_rank(SA)
            rB = np.linalg.matrix_rank(SB)
            r = np.amin([rA, rB])

    if r == 0:
        print("Warning: Matrix of rank 0 determined.")
        return 0, 0, 0

    # Reduce UA, UB to min or max rank:
    UA = UA[:, :r]
    UB = UB[:, :r]

    if method == 0:
        # Compute Grassmannian distance between (double infinite)  UA, UB 
        R = np.dot(UA.T, UB)
        UR, SR, VR = np.linalg.svd(R, full_matrices=False)
        SR = np.round_(SR, 6)
        SR[np.where(SR >1)] = 1.0
        theta = np.arccos(SR)
        dist = np.sqrt(abs(rA-rB)*np.pi**2./4. + \
                np.sum(theta**2))
        methodtxt = "Grassmannian"

    if method == 1:
        # Compute geodesic distance on the Grassmann between U1, UB
        R = np.dot(UA.T, UB)
        UR, SR, VR = np.linalg.svd(R, full_matrices=False)
        SR = np.round_(SR, 6)
        SR[np.where(SR >1)] = 1.0
        theta = np.arccos(SR)
        dist = np.sqrt(np.sum(theta**2))
        methodtxt = "Geodesic"

    # Print to screen metric and distance
    if verbose:
        text1 = "{} distance of {} determined.\n".format(methodtxt, dist)
        text2 = "Original ranks rA = {} and rB = {}\n".format(rA, rB)
        text3 = "Final rank r = {}.\n".format(r)
        print(text1 + text2 + text3)

    return dist, rA, rB

def getContinuumFieldValues(fileName, dtype=np.float32):
    field_bin = np.fromfile(fileName, dtype=dtype)
    ny = int(field_bin[0]) + 1
    nx = int(len(field_bin)/ny)
    field = field_bin.reshape(nx, ny)
    field = field[1:, 1:]

    return field      

def plotFields(X, Y, parameters, reference, optimizationmodel, colorMap='inferno', nLevels=6, nContours=9):

    pars = parameters
    ref = reference
    om = optimizationmodel

    # Get indices of minimima
    Yi = np.copy(Y)
    Yi[np.isnan(Yi)] = np.inf
    Yi_min = np.minimum.accumulate(Yi)
    Y_list = list(Yi_min)
    i_min = [Y_list.index(item) for item in Y_list]
    i_min = np.unique(i_min)
    
    for ii, vi  in enumerate(i_min[1:]):

        fig = plt.figure(figsize=(8.5, 11))
        variables = []
        ip = 0
        for param in parameters.parameters:
            if not param.is_constant:
                param.value = X[vi, ip]
                variables.append(param.text_hc)
                ip += 1
        
        obs = Observation(parameters, reference, optimizationmodel)
        dist = obs.computeDistance()
        
        # Plot stress-strain curves 
        ax = plt.subplot(4, 3, 3)
        strain_cm = np.linspace(0, reference.maxStrain, optimizationmodel.nContinuumFrames)
        stress_cm = np.zeros(strain_cm.shape)

        for ij, vj in enumerate(strain_cm):
            stress_cm[ij] = parameters.getValue('s_y')*computeContinuumDeviatoricStress(\
                optimizationmodel.continuumOutPath, ij)

        plt.plot(reference.strain, reference.stress, \
                    color='red', label='*', linewidth=2)

        label = '{}'.format(vi+1)
        plt.plot(strain_cm, stress_cm, \
                    color='black', label=label, \
                    linewidth=2, linestyle='--')

        if ii > 0:
            vj = i_min[ii-1]
            label = '{}'.format(vj + 1)
            plt.plot(strain_cm, stress_prev, \
                        color='grey', label=label, \
                        linewidth=2, linestyle=':')

        plt.legend()
        plt.xlabel(r'$\gamma$')
        plt.ylabel(r'$\tau$')
        
        # Plot MD initial potential energy field
        pe_md_0 = reference.getFieldValues('pe.MD.0')
        levels = np.linspace(np.amin(pe_md_0), np.amax(pe_md_0), nLevels)

        ax = plt.subplot(4, 3, 4)
        plt.title(r'${\mathbf{U}_i}^{*}$')
        plt.contourf(pe_md_0, levels=levels, cmap=colorMap)
        plt.colorbar()
        ax.axis('equal')
        ax.axis('off')

        # Plot MD final temperature field
        pe_md_f = reference.getFieldValues('pe.MD.100')
        pe_Te_f = parameters.getValue('beta')*(pe_md_f-parameters.getValue('u0'))*TZ 
        levels = np.linspace(np.amin(pe_Te_f), np.amax(pe_Te_f), nLevels)

        ax = plt.subplot(4, 3, 5)
        plt.title(r'${\mathbf{T}_f}^{*}$')
        plt.contourf(pe_Te_f, levels=levels, cmap=colorMap)
        plt.colorbar()
        ax.axis('equal')
        ax.axis('off')
 
        # Plot MD final strain field
        pe_strain_f = reference.getFieldValues('Exy.MD.100')
        levels = np.linspace(np.amin(pe_strain_f), np.amax(pe_strain_f), nLevels)    
 
        ax = plt.subplot(4, 3, 6)
        plt.title(r'${\mathbf{\Gamma}_f}^{*}$')
        plt.contourf(pe_strain_f, levels=levels, cmap=colorMap)
        plt.colorbar()
        ax.axis('equal')
        ax.axis('off')

        # Plot CM current initial effective temperature field
        fileName = optimizationmodel.continuumOutPath + '/' + \
                    optimizationmodel.temperatureFieldPrefix + '0'
        Te_cm_0 = getContinuumFieldValues(fileName)
        levels = np.linspace(np.amin(Te_cm_0), np.amax(Te_cm_0), nLevels)

        ax = plt.subplot(4, 3, 7)
        plt.title(r'${\mathbf{T}_i}^{' + '{}'.format(vi+1) + '}$')
        plt.contourf(Te_cm_0, levels=levels, cmap=colorMap)
        plt.colorbar()
        ax.axis('equal')
        ax.axis('off')

        # Plot CM current final effective temperature field
        fileName = optimizationmodel.continuumOutPath + '/' + \
                    optimizationmodel.temperatureFieldPrefix + \
                    str(optimizationmodel.nContinuumFrames-1)
        Te_cm_f = getContinuumFieldValues(fileName)
        levels = np.linspace(np.amin(Te_cm_f), np.amax(Te_cm_f), nLevels)

        ax = plt.subplot(4, 3, 8)
        plt.title(r'${\mathbf{T}_f}^{' + '{}'.format(vi+1) + '}$')
        plt.contourf(Te_cm_f, levels=levels, cmap=colorMap)
        plt.colorbar()
        ax.axis('equal')
        ax.axis('off')

        # Plot current CM final strain field
        fileName = optimizationmodel.continuumOutPath + '/' + \
                    optimizationmodel.strainFieldPrefix + \
                    str(optimizationmodel.nContinuumFrames-1)
        strain_cm_f = getContinuumFieldValues(fileName)
        strain_cm_f = 2.*strain_cm_f
        levels = np.linspace(np.amin(strain_cm_f), np.amax(strain_cm_f), nLevels)

        ax = plt.subplot(4, 3, 9)
        plt.title(r'${\mathbf{\Gamma}_f}^{' + '{}'.format(vi+1) + '}$')
        plt.contourf(strain_cm_f, levels=levels, cmap=colorMap)
        plt.colorbar()
        ax.axis('equal')
        ax.axis('off')


        if ii == 0:

            # Add text label
            ax = plt.subplot(4, 3, 1)
            ax.axis('off')
            titleText = 'Observation: n = {}\n\n'.format(vi+1) + \
                        r'Input variables: $X_n$ = {' + \
                        ', '.join(variables) + '}\n' + \
                        r'Ouput distance: $Y_n$ = ' + optimizationmodel.distanceMetric_hc + \
                        '\n\nBest observations: \n\n' + r'$X_{' + '{}'.format(vi+1) + '}$' + \
                        ' = {' + ', '.join(['{:.4f}'.format(item) for item in X[vi, :]]) + \
                        '}\n' + r'$Y_{' + '{}'.format(vi+1) + '}$' + ' = {:.4f}\n\n'.format(Y[vi])
            plt.text(0, 1, titleText, \
                        horizontalalignment='left', verticalalignment='top', \
                        wrap=True)


        else:
            # Add text label
            ax = plt.subplot(4, 3, 1)
            ax.axis('off')
            vj = i_min[ii-1]
            titleText = 'Observation: n = {}\n\n'.format(vi+1) + \
                        r'Input variables: $X_n$ = {' + \
                        ', '.join(variables) + '}\n' + \
                        r'Ouput distance: $Y_n$ = ' + optimizationmodel.distanceMetric_hc + \
                        '\n\nBest observations: \n\n' + r'$X_{' + '{}'.format(vi+1) + '}$' + \
                        ' = {' + ', '.join(['{:.4f}'.format(item) for item in X[vi, :]]) + \
                        '}\n' + r'$Y_{' + '{}'.format(vi+1) + '}$' + ' = {:.4f}\n\n'.format(Y[vi]) + \
                        '\n' + r'$X_{' + '{}'.format(vj+1) + '}$' + ' = {' + \
                        ', '.join(['{:.4f}'.format(item) for item in X[vj, :]]) + '}\n' + \
                        r'$Y_{' + '{}'.format(vj+1) + '}$' + ' = {:.4f}\n\n'.format(Y[vj]) 

            plt.text(0, 1, titleText, \
                        horizontalalignment='left', verticalalignment='top', \
                        wrap=True)

            # Plot prior CM initial effective temperature field
            ax = plt.subplot(4, 3, 10)
            levels = np.linspace(np.amin(Te_0_prev), np.amax(Te_0_prev), nLevels)
            plt.title(r'${\mathbf{T}_0}^{' + '{}'.format(vj+1) + '}$')
            plt.contourf(Te_0_prev, levels=levels, cmap=colorMap)
            plt.colorbar()
            ax.axis('equal')
            ax.axis('off')

            # Plot prior CM final effective temperature field
            ax = plt.subplot(4, 3, 11)
            levels = np.linspace(np.amin(Te_f_prev), np.amax(Te_f_prev), nLevels)
            plt.title(r'${\mathbf{T}_f}^{' + '{}'.format(vj+1) + '}$')
            plt.contourf(Te_f_prev, levels=levels, cmap=colorMap)
            plt.colorbar()
            ax.axis('equal')
            ax.axis('off')
    
            # Plot prior CM final strain field
            ax = plt.subplot(4, 3, 12)
            levels = np.linspace(np.amin(strain_prev), np.amax(strain_prev), nLevels)
            plt.title(r'${\mathbf{\Gamma}_f}^{' + '{}'.format(vj+1) + '}$')
            plt.contourf(strain_prev, levels=levels, cmap=colorMap)
            plt.colorbar()
            ax.axis('equal')
            ax.axis('off')        


        # Update previous field values
        stress_prev = stress_cm
        Te_0_prev = Te_cm_0
        Te_f_prev = Te_cm_f 
        strain_prev = strain_cm_f

        plt.tight_layout()
        plt.savefig('field--obs_{:04d}.png'.format(vi+1), dpi=200)
        plt.close()


def getContinuumFieldValues(filePath, fieldName):
    """
      Retrieve continuum field quantities from binary file.
    """
    f = np.fromfile(filePath + '/' + fieldName, dtype=np.float32)
    ny = int(f[0]) + 1
    nx = int(len(f)/ny)
    f = f.reshape(nx, ny)
    f = f[1:, 1:]
    return f

def computeContinuumDeviatoricStress(filePath, frameNum):
    """
    Compute the magnitude of deviatoric shear stress for continuum output.
    """
    dev_ = getContinuumFieldValues(filePath, 'dev.{}'.format(int(frameNum)))
    dev = np.mean(dev_)
 
    s_ = getContinuumFieldValues(filePath, 's.{}'.format(int(frameNum)))
    s = np.mean(s_)
    
    tau_ = getContinuumFieldValues(filePath, 'tau.{}'.format(int(frameNum)))
    tau = np.mean(tau_)

    q_ = getContinuumFieldValues(filePath, 'q.{}'.format(int(frameNum)))
    q = np.mean(q_)

    return tau

"""
Main Program
"""
def main():

    ref = Reference()
    peMD0 = ref.getFieldValues('pe.MD.0')
    
    # Initialize optimization model 
    om = OptimizationModel(nInitialObs=350, nTotalObs=15000, nSurrEvals=100000, \
                            samplingstyle=3,\
                            distanceMetric=4, decompositionRank=2, verboseDecomposition=True,\
                            rankTolerance=1e-4, kernelIndex=3, \
                            detectOutliers=False, contamination=0.05, anomoly_algorithm="Local Outlier Factor")
    
    #plt.figure()
    #plt.plot(ref.strain, ref.stress)
    #plt.savefig('testing.png')

    # Generate initial parameter list`
    beta = Parameter('beta', 8.00, vmin=2, vmax=15)
    u0 = Parameter('u0', -3.3626, vmin=-3.395, vmax=-3.360)
    chi_len = Parameter('chi_len', 25.59275383)
    ep = Parameter('ep', 44.52901253)
    c0 = Parameter('c0', 0.3, vmin=0.05, vmax=1)
    chi_inf = Parameter('chi_inf', 2730)
    s_y = Parameter('s_y', 0.95, vmin=0.8, vmax=1.1)

    parameters = Parameters([beta, u0, chi_inf, chi_len, c0, ep, s_y])

    # Initialize dictionary to contain parameter, value pairs
    initparams = {}

    for param in parameters.parameters: 
        initparams[param.name] = param.value

    #for param in initparams:
    #    print(param)
    #    print(initparams[param])

    #sys.exit()
 
    parameters.list_parameters()
    print('ndims = {}'.format(parameters.ndims))

    # Generate the reference
    if os.path.exists('reference'):
        command = ['rm', '-r', 'reference']
        subprocess.run(command)

    if os.path.exists('sct_q.out'):
        command = ['rm', '-r', 'sct_q.out']
        subprocess.run(command)

    #ref = Reference()
    PE_0 = ref.sourceDir + ref.energyFieldPrefix + '0'
    PE_f = np.amax(ref.getFieldValues('pe.MD.100'))
    T_inf = parameters.getValue('beta')*(PE_f-parameters.getValue('u0'))*TZ

    command = ['shear_energy_Adam3', 'qs', PE_0,
                '{}'.format(parameters.getValue('beta')), 
                '{}'.format(parameters.getValue('u0')),
                'chi_inf', '{}'.format(T_inf),
                'chi_len', '{}'.format(parameters.getValue('chi_len')),
                'c0', '{}'.format(parameters.getValue('c0')),
                'ep', '{}'.format(parameters.getValue('ep')),
                's_y', '{}'.format(parameters.getValue('s_y'))]

    subprocess.run(command, timeout=360)
    
    command = ['mv', 'sct_q.out/', 'reference/']
    subprocess.run(command)

    # Load reference state from continuum model, set value of T_inf
    #ref = Reference(fromContinuum=True)
    if ref.fromContinuum:
        T_ = np.fromfile('reference/tem.100', dtype=np.float32)
        ny = int(T_[0]) + 1
        nx = int(len(T_)/ny)
        T_ = T_.reshape(nx, ny)
        T_ = T_[1:, 1:]
        T_inf = np.amax(T_)
        parameters.setValue('chi_inf', T_inf)

 
    X, Y = ego(om, ref, parameters)
    qq.close()
    
    #data = np.load('ego_data.npz')
    #X = data['X']
    #Y = data['Y']


    #plotConvergence(X, Y, parameters, om, targetVals=initparams)
    #plotFields(X, Y, parameters, ref, om)



if __name__ == '__main__': main()
