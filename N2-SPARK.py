# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:01:51 2021

@author: Hashimoto
"""

import numpy as np
user_Te=10000
user_Tr=400
user_Tv=400
dimenrot=100
dimenvib=11
vmaxx=99
DegSpNc=2
RefNeE=11
RefNeG=3
vmax=np.array([54.18546786 ,56.66147636 ,65.29356123 ,71.19326124 ,76.83058877 ,69.34948648 ,75.39238761 ,119.2751124 ,132.9922559 ,47.90218771 ,97.83134921 ,107.766521 ,44.86292906 ,183.8861016 ,169.9227578 ,47.44489509 ,51.3558952 ,55.80682605 ,57.97953216 ,127.5447955 ,61.51915528 ,61.92131767 ,56.35407454])
ge=np.array([3 ,6 ,6 ,3 ,1 ,2 ,2 ,5 ,6 ,6 ,6 ,10 ,3 ,2 ,1 ,3 ,2 ,1 ,2 ,6 ,1 ,2 ,1])
Te=np.array([50204 ,59619 ,59805 ,66272 ,68153 ,69283 ,72097 ,76436 ,89505 ,89137 ,98260 ,88739 ,96951 ,101667 ,105216 ,103647 ,104222 ,104419 ,105878 ,106176 ,113438 ,114305 ,115926])
ElectronicLevels=23
J=np.array([i+1 for i in range(100)])
Bv=np.loadtxt("Bv.txt")
Gv=np.loadtxt("Gv.txt")
FJ=np.repeat(Bv[None,:],dimenrot,axis=0).transpose(1,2,0)*np.reshape(np.tile(J.T*(J+1).T,[1,vmaxx+1,ElectronicLevels]),(23,100,100))
QR=np.tile(2*(J+1).T,[1,vmaxx+1,ElectronicLevels]).reshape((23,100,100))*np.exp(-(1.4388/user_Tr)*FJ)
SumQR=(user_Tr/(1.4388*Bv))/DegSpNc
NJ=QR/np.repeat(SumQR[None,:],dimenrot,axis=0).transpose(1,2,0)
NJE=NJ[RefNeE-1,0:dimenvib,:]
NJG=NJ[RefNeG-1,0:dimenvib,:]
Qv=np.exp(-(1.4388/user_Tv)*Gv)*SumQR
vmax=np.repeat(vmax.T[None,:],vmaxx+1,axis=0).T
VibrationLevels=np.repeat(np.array([i for i in range(vmaxx+1)])[None,:],ElectronicLevels,axis=0)
vmax=(vmax>=VibrationLevels)
Qv=Qv*vmax
SumQv=np.sum(Qv,axis=1)
Nv=Qv/np.repeat(SumQv[None,:],vmaxx+1,axis=0).T
NvE=Nv[RefNeE-1,0:dimenvib]
NvG=Nv[RefNeG-1,0:dimenvib]
Qe=ge.T*np.exp(-(1.4388/user_Te)*Te.T)*SumQv
SumQe=sum(Qe)
Ne=Qe/np.array([SumQe for i in range(ElectronicLevels)]).T
NeE=Ne[RefNeE-1]
NeG=Ne[RefNeG-1]