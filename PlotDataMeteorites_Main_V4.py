#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:56:47 2024

@author: louis
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#%% SECTION I : Frequency per Mass 
#Data + Functions + References

#Following (Dalton et al. 1972) (B.G Cour Palais et al. 1969)
def Dalton_CourPalais_avg_total_flux_micro2gram(m):
    N_t = (10**(-14.597 - 1.213*math.log10(m)))*1e6*3600
    return N_t

def Dalton_CourPalais_avg_total_flux_pico2micro(m):
    N_t = (10**(-14.566 - 1.584*math.log10(m) - 0.063*(math.log10(m))**2))*1e6*3600
    return N_t
#####################################################################

#Following (Hartmann,1965)
def Hartman_tera2higher(m):
    d_crater = (37.8)*m**(1/3.06) #in cm / mass should be in gram - avg = 29.15
    a = 5e-4*1e9*8766 #cst for diam = 1km ; units: /km^2*h
    N_t = a * (d_crater*1e5)**(-2.4) 
    return N_t
####################################################################

#Following (Suggs et al. 2014)
def Suggs_gram2kg(m):
    avg_flux = 1.03*1e-7 #met/km^2*h
    if m<1:
        print("This valus is outside the range for this study")
    elif 500 < m <= 1000:
        N_t = avg_flux * 1/126
    elif 250 < m <= 500:
        N_t = avg_flux * (1+3)/126
    elif 150 < m <= 250:
        N_t = avg_flux * (1+3+8)/126
    elif 100 < m <= 150:
        N_t = avg_flux * (1+3+8+10)/126 
    elif 50 < m <= 100:
        N_t = avg_flux * (1+3+8+10+30)/126
    elif 25 < m <= 50:
        N_t = avg_flux * (1+3+8+10+30+54)/126 
    elif 10 < m <= 25:
        N_t = avg_flux * (1+3+8+10+30+54+16)/126 
    elif 1 <= m <=10:
        N_t = avg_flux * (1+3+8+10+30+54+16+4)/126 
    elif m>1000:
        print("This valus is outside the range for this study")
    return(N_t)        
####################################################################    

#Following (Liakos et al. 2024)
def Liakos_gram2kg(m):
    avg_flux = 2.18*1e-7 #met/km^2*h
    #avg_flux = 6.43*1e-6 #met/km^2*h
    if m<1:
        print("This valus is outside the range for this study")
    elif 1 <= m <=25:
        N_t = avg_flux * (8+5+6+10+21+7+6+8+10+31+32+23+31)/198 
    elif 25 < m <= 50:
        N_t = avg_flux * (8+5+6+10+21+7+6+8+10+31+32+23)/198 
    elif 50 < m <= 75:
        N_t = avg_flux * (8+5+6+10+21+7+6+8+10+31+32)/198
    elif 75 < m <= 100:
        N_t = avg_flux * (8+5+6+10+21+7+6+8+10+31)/198
    elif 100 < m <= 125:
        N_t = avg_flux * (8+5+6+10+21+7+6+8+10)/198
    elif 125 < m <= 150:
        N_t = avg_flux * (8+5+6+10+21+7+6+8)/198
    elif 150 < m <= 175:
        N_t = avg_flux * (8+5+6+10+21+7+6)/198 
    elif 175 < m <= 200:
        N_t = avg_flux * (8+5+6+10+21+7)/198
    elif 200 < m <= 400:
        N_t = avg_flux * (8+5+6+10+21)/198
    elif 400 < m <= 600:
        N_t = avg_flux * (8+5+6+10)/198
    elif 600 < m <= 1200:
        N_t = avg_flux * (8+5+6)/198
    elif 1200 < m <= 2000:
        N_t = avg_flux * (8+5)/198 
    elif 2000 < m <= 2800:
        N_t = avg_flux * (5)/198
    elif m>2800:
        print("This valus is outside the range for this study")
    return(N_t)        
#################################################################### 

#Following (Oberst, Suggs et al. 2012)
 
masses_marchi = []
impactor_diam_marchi = [3e-4 , 1e-3 , 1e-2 , 0.07 , 0.65 , 4.5 , 10 , 70] #km
freq_marchi = [1e-6/8766 , 6e-8/8766 , 1.5e-10/8766 , 1e-12/8766 , 1e-14/8766 , 1e-16/8766 , 1e-17/8766 , 5.5e-20/8766] #in /km^2*h\

for i in range(0,len(impactor_diam_marchi)):
    V = 4/3 * np.pi *((impactor_diam_marchi[i]*1e5)/2)**3
    rho_met = 3.587 #g/cm^3 determined from (Dalton, 1972)
    m = rho_met * V
    masses_marchi.append(m)

#Following (Ivanov, 2001) --> SIMPLE & GRAVITY REGIME LAWS
def Ivanov_CratDiam2Mass(D): #D in cm
    rho_met = 3.587  #g/cm^3
    rho_reg = 1.5 #g/cm^3
    Vi = 21.2e5 #cm/h
    g = 1.62e2 #cm/s^2
    C = Vi * np.sin(np.pi/4) #45deg angle (most probable)
    m = 4/3*rho_met*np.pi * (D/1.16 * (rho_met/rho_reg)**(1/3) * g**0.22/C**0.43)**(1/0.26) #in g
    return(m)

def Ivanov_Mass2CratDiam(m): #m in g
    rho_met = 3.587  #g/cm^3
    rho_reg = 1.5 #g/cm^3
    Vi = 21.2e5 #cm/s
    g = 1.62e2 #cm/s^2
    C = Vi * np.sin(np.pi/4) #45deg angle (most probable)
    D = 1.16 * (rho_met/rho_reg)**(1/3) * (2*(3*m/(4*rho_met*np.pi))**(1/3))**0.78 * C**0.43 * g**(-0.22) #in cm
    return(D)


# # Following (Housen & Holsapple 2011)
# def Housen_Mass2CratDiam(m): #mass in g --> SIMPLE & GRAVITY REGIME LAWS
#     m = m*1e-3 #in kg
#     Y = 1e4 #Pa -> Pa = [kg/s^2*m]
#     rho_met = 3.244e3 #kg/m^3
#     rho_reg = 1.5e3 #kg/m^3
#     Vi = 20e3 #m/s
#     H1 = 0.59
#     mu = 0.41
#     nu = 0.4
#     R = (m/rho_met)**(1/3) * H2 * (rho_met/rho_reg)**((1-3*nu)/3) * (Y/(rho_met*Vi**2))**(-mu/2) #in m
#     D = 2*R #in m
#     return (D)

# #Following (Housen & Holsapple 2011) --> SIMPLE & STRENGTH REGIME LAW
# def Housen_Mass2CratDiam(m): #m in g
#     m = m*1e-3 #in kg
#     Y = 1e4 #Pa -> Pa = [kg/s^2*meter]
#     rho_met = 3.244e3 #kg/m^3
#     rho_reg = 1.5e3 #kg/m^3
#     Vi = 20e3 #m/s
#     alpha = 0.51
#     a = (3*m/(4*np.pi*rho_met))**(1/3) #assuming sphericity of the meteorite
#     R = a*(Y/(rho_reg*Vi**2))**(alpha/(alpha-3)) #in m
#     D = 2*R #in m
#     return (D)

def Housen_Mass2CratDiam(m): #mass in kg --> SIMPLE & STRENGTH REGIME LAW
    m = m*1e-3 #in kg
    Y = 1e4 #Pa -> Pa = [kg/s^2*m]
    rho_met = 3.587e3 #kg/m^3
    rho_reg = 1.5e3 #kg/m^3
    Vi = 21.2e3 #m/s
    H2 = 0.81
    mu = 0.41
    nu = 0.4
    
    R = (m/rho_met)**(1/3) * H2 * (rho_met/rho_reg)**((1-3*nu)/3) * (Y/(rho_met*Vi**2))**(-mu/2) #in m
    D = 2*R #in m
    return (D)


#In Oberst 2012 -> Following Shmidt & Housen 1987
def Shmidt_part2_KE2Diam(KE): #KE in J
    g= 1.62 #m/s^2
    Vi = 21.2e3 #m/s
    R = 0.122*(KE**0.28)*(Vi**(-0.21))*g**(-0.17) #in m
    D = (2*R)*1e-3 #in km
    return(D)

#In Oberst 2012 -> Following Shmidt & Housen 1987
def Shmidt_part1_KE2Diam(KE): #KE in J
    g= 1.62 #m/s^2
    Vi = 21.2e3 #m/s
    R = 0.1*(KE**0.28)*(Vi**(-0.21))*g**(-0.17) #in m
    D = (2*R)*1e-3 #in km
    return(D)

###### Get the frequency of crater formation
#In Oberst 2012 -> Following Revelle, 2001
def Revelle_Rad2Freq(R): #only valid for R in [20-323] meters!!
    Freq = (10**(3.84-2.59*math.log10(R))) * 1/(3.8e7*8766) #in km^-2h^-1
    return(Freq)

#In Oberst 2012 -> Following Brown et al 2002
def Brown_Rad2Freq(R): #only valid for R in [24-215] meters!!
    Freq = (10**(4.73-3.21*math.log10(R))) * 1/(3.8e7*8766) #in km^-2h^-1
    return(Freq)

#In Oberst 2012 -> Following Halliday et al 1996
def Halliday1_Rad2Freq(R): #only valid for R in [1.4-3.5] meters!! (averaged)
    Freq = (10**(3.89-1.79*math.log10(R))) * 1/(3.8e7*8766) #in km^-2h^-1
    return(Freq)

def Halliday2_Rad2Freq(R): #only valid for R in [3.5-5.3] meters!! (averaged)
    Freq = (10**(4.99-3.57*math.log10(R))) * 1/(3.8e7*8766) #in km^-2h^-1
    return(Freq)

#In Oberst 2012 -> Following Oberst & Nakamura 1989
def Oberst_Rad2Freq(R): #only valid for R in [24-33] meters!! (averaged)
    Freq = (10**(4.88-3.54*math.log10(R))) * 1/(3.8e7*8766) #in km^-2h^-1
    return(Freq)

#######

#Following (William et al, 2014)
crater_diam_william = [1e-3 , 1.5e-3 , 1e-2 , 2e-2  , 1e-1, 2.6e-1] #in km
freq_william = [2e-4/8766, 8e-5/8766 , 1.1e-6/8766, 2.1e-7/8766  , 2e-9/8766 , 5e-11/8766]#in /km^2*h

masses_william =[]

for i in range(0,len(crater_diam_william)):
    masses_william.append(Ivanov_CratDiam2Mass(crater_diam_william[i]*1e5))


####################################################################

# Data from Grun et al. (1984) 
mass_grun = np.logspace(-6,2,9) #in g
flux_grun = [170, 16.56, 1.12, 6.8e-2, 3.5e-3, 1.7e-4, 7.9e-6, 3.6e-7, 1.7e-8] #in /km^2*h
    
######################################################################

m_pico2micro = np.logspace(-12,-6,100)# in g
flux_pico2micro = []
for i in range(0,len(m_pico2micro)):
    flux_pico2micro.append(Dalton_CourPalais_avg_total_flux_pico2micro(m_pico2micro[i])) 
    

m_micro2_100gram = np.logspace(-6,2,10) #in g 
flux_micro2gram = []
for i in range(0,len(m_micro2_100gram)):
    flux_micro2gram.append(Dalton_CourPalais_avg_total_flux_micro2gram(m_micro2_100gram[i])) 
    
m_gram2kg = np.logspace(1,3,10) #in g

flux_gram2kg_suggs = []
flux_gram2kg_liakos = []
for i in range(0,len(m_gram2kg)):
    flux_gram2kg_suggs.append(Suggs_gram2kg(m_gram2kg[i]))
    flux_gram2kg_liakos.append(Liakos_gram2kg(m_gram2kg[i]))


m_kg2quintillion = np.logspace(3,17,10) #in g quintillon is 10^18 normally
flux_kg2quintillion_hartman = []
for i in range(0,len(m_kg2quintillion)):
    flux_kg2quintillion_hartman.append(Hartman_tera2higher(m_kg2quintillion[i]))
    

#%% Linear and Quadratic fitting  -> to be used all the time   


x_micro2kg = np.concatenate((m_micro2_100gram,mass_grun,m_gram2kg,m_gram2kg))
x_kg2higher = np.concatenate((np.array(masses_william),m_kg2quintillion,np.array(masses_marchi)))
    

y_micro2kg = np.concatenate((np.array(flux_micro2gram),flux_grun,np.array(flux_gram2kg_suggs),np.array(flux_gram2kg_liakos)))
y_kg2higher = np.concatenate((freq_william,np.array(flux_kg2quintillion_hartman),freq_marchi))


coef_lin_micro2kg = np.polyfit(np.log10(x_micro2kg), np.log10(y_micro2kg), 1)
poly_lin_micro2kg = np.poly1d(coef_lin_micro2kg)


coef_lin_kg2higher = np.polyfit(np.log10(x_kg2higher), np.log10(y_kg2higher), 1)
poly_lin_kg2higher = np.poly1d(coef_lin_kg2higher)

mass4plot_micro2kg = np.logspace(-6,3,100)
mass4plot_kg2higher = np.logspace(3,20,100)

frequ4plot_micro2kg = 10**poly_lin_micro2kg(np.log10(mass4plot_micro2kg))
frequ4plot_kg2higher = 10**poly_lin_kg2higher(np.log10(mass4plot_kg2higher))


mass4plot = np.concatenate((mass4plot_micro2kg,mass4plot_kg2higher))
frequ4plot = np.concatenate((frequ4plot_micro2kg,frequ4plot_kg2higher))

#%%  Plots - graphs : Cumulative frequency per mass 
#plt.close(figure1)
figure1, ax = plt.subplots()
#plt.loglog(m_pico2micro,flux_pico2micro,'b',linewidth = 3.5)
linewidth_set = 4
plt.loglog(m_micro2_100gram,flux_micro2gram,'b',label = 'Dalton et al. 1972',linewidth = linewidth_set)
plt.loglog(mass_grun,flux_grun,'brown',label = 'Grun et al. 1984',linewidth = linewidth_set)
plt.loglog(m_gram2kg,flux_gram2kg_suggs,'r',label='Suggs et al. 2014',linewidth = linewidth_set)
plt.loglog(m_gram2kg,flux_gram2kg_liakos,'y', label='Liakos et al. 2024',linewidth = linewidth_set)
plt.loglog(masses_william,freq_william,'c',label='William et al. 2014',linewidth = linewidth_set)
plt.loglog(m_kg2quintillion,flux_kg2quintillion_hartman,'g',label = 'Hartmann 1965',linewidth = linewidth_set)
plt.loglog(masses_marchi,freq_marchi,'chartreuse',label = 'Marchi et al. 2009',linewidth = linewidth_set)
#plt.plot(x4plot,10**poly_quad(np.log10(x4plot)),'k',label = 'Quadratic fit',linestyle = 'dashed')
plt.plot(mass4plot_micro2kg,frequ4plot_micro2kg,'grey',label = 'Linear fit:'+str(round(coef_lin_micro2kg[0],4))+'x'+str(round(coef_lin_micro2kg[1],4)),linestyle = 'dashed',linewidth = linewidth_set)
plt.plot(mass4plot_kg2higher,frequ4plot_kg2higher,'k',label = 'Linear fit:'+str(round(coef_lin_kg2higher[0],4))+'x'+str(round(coef_lin_kg2higher[1],4)),linestyle = 'dashed',linewidth = linewidth_set)

#plt.text(1e-10, 1e-3, '-0.9115x-5.1337', fontsize = 6)

plt.xlabel('Impactor mass - [g]',size=16)
plt.ylabel('Cumul. Frequency - $[km^{-2}h^{-1}]$',size=16)
plt.legend(fontsize = 13)
ax.tick_params(axis='both', which='major', labelsize=15)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid( which='both')
plt.title('Cumulative Frequency of Meteorite Strike on the Moon',size=15)
# Save the figure to PDF format
plt.tight_layout()
#plt.xlim([1,1e6])
#plt.ylim([1e-11,1e-5])
pdf_path = '/Users/louis/Desktop/Thèse UCSD/Internships/LPI - Summer 2024/Codes/Matlab - Meteorites/Plots_meteorites/Validated Plots Bigger/Mass_vs_Frequency_closeup.pdf'
#plt.savefig(pdf_path)

############################################################################################################
############################################################################################################
#%% SECTION II : Frequency per Energy (MJ)

#Using x4plot that spans the whole mass range

Vi = [16,18,21.2,22,24] #km/s -> check refs in yellow book
#Vsho = 24 #km/s -> check refs in yellow book


def KineticEnergy(m,V):
    KE = 0.5 * m * V**2 #KE in J -> mass (m) in kg and velocity (Vi) in m/s
    return(KE)

KE_micro2kg = np.empty((len(Vi),len(mass4plot_micro2kg)))
KE_kg2sextillion = np.empty((len(Vi),len(mass4plot_kg2higher)))
for j in range(len(Vi)):
    V = Vi[j]
    for i in range(len(mass4plot_micro2kg)):
        KE = KineticEnergy(mass4plot_micro2kg[i]*1e-3,V*1e3) #KE in J
        KE_micro2kg[j,i] = KE*1e-6  #KE in MJ
        
for j in range(len(Vi)):
    V = Vi[j]
    for i in range(len(mass4plot_kg2higher)):
        KE = KineticEnergy(mass4plot_kg2higher[i]*1e-3,V*1e3) #KE in J 
        KE_kg2sextillion[j,i] = KE*1e-6  #KE in MJ


#plt.close(figure(2))
figure2, ax = plt.subplots()
# plt.loglog(KE_micro2kg[0,:],frequ4plot_micro2kg,'b',label = 'V = 16km/s')
# plt.loglog(KE_kg2sextillion[0,:],frequ4plot_kg2higher,'b')

# plt.plot(KE_micro2kg[1,:],frequ4plot_micro2kg,'r',label = 'V = 18km/s')
# plt.loglog(KE_kg2sextillion[1,:],frequ4plot_kg2higher,'r')

plt.plot(KE_micro2kg[2,:],frequ4plot_micro2kg,'g',label = '$ \overline{V}_{met} = 21.2$ km/s',linewidth = 3.5)
plt.loglog(KE_kg2sextillion[2,:],frequ4plot_kg2higher,'g',linewidth = 3.5)

# plt.plot(KE_micro2kg[3,:],frequ4plot_micro2kg,'y',label = 'V = 22km/s')
# plt.loglog(KE_kg2sextillion[3,:],frequ4plot_kg2higher,'y')

# plt.plot(KE_micro2kg[4,:],frequ4plot_micro2kg,'c',label = 'V = 24km/s')
# plt.loglog(KE_kg2sextillion[4,:],frequ4plot_kg2higher,'c')

plt.axvline(x = 4e-3, color = 'k',linestyle = 'dashed',linewidth = 2)
plt.text(5e-3, 5e-10, 'Rifle Bullet', rotation=90, verticalalignment='bottom', color='k',size=13)
plt.axvline(x = 55, color = 'k',linestyle = 'dashed',linewidth = 2)
plt.text(35, 1e-5, 'Firework', rotation=90, verticalalignment='bottom', color='k',size=13)
plt.axvline(x = 500, color = 'k' ,linestyle = 'dashed',linewidth = 2)
plt.text(600, 1e-5, 'Hand Grenade', rotation=90, verticalalignment='bottom', color='k',size=13)
plt.axvline(x = 4000, color = 'k' ,linestyle = 'dashed',linewidth = 2)
plt.text(5200, 1e-5, 'RPG Rocket', rotation=90, verticalalignment='bottom', color='k',size=13)

plt.axvline(x = 100e6, color = 'k' ,linestyle = 'dashed',linewidth = 2)
plt.text(110e6, 1e-22, 'Hiroshima', rotation=90, verticalalignment='bottom', color='k',size=13)

#plt.xlim(0,1e13)
plt.xlabel('Impactor KE - [MJ]',size=13)
plt.ylabel('Cumul. Frequency - $[km^{-2} h^{-1}$]',size=13)
plt.legend(fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=13)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid( which='both')
plt.title('Cumulative Frequency of Meteorite Strike on the Moon',size=13)
pdf_path = '/Users/louis/Desktop/Thèse UCSD/Internships/LPI - Summer 2024/Codes/Matlab - Meteorites/Plots_meteorites/Validated Plots Bigger/KE_vs_Frequency_closeup.pdf'
plt.tight_layout()
#plt.xlim([1e-3,1e4])
#plt.ylim([1e-10,1e-2])
#plt.savefig(pdf_path)


#%% SECTION II : Plot of Crater diam vs frequency 

#Using Oberst sclaing law 
part1_KE = KE_micro2kg[2,:]#in MJ
part2_KE = KE_kg2sextillion[2,:] #in MJ

part1_frequ = frequ4plot_micro2kg
part2_frequ = frequ4plot_kg2higher

Schmidt_crat_part1 = []
Schmidt_crat_part2 = []

for i in range(len(part1_KE)):
    Schmidt_crat_part1.append(Shmidt_part1_KE2Diam(part1_KE[i]*1e6))
    Schmidt_crat_part2.append(Shmidt_part2_KE2Diam(part2_KE[i]*1e6))

crat_diam_Revelle = np.logspace(-1.4,-0.19,10) #in km 
crat_diam_Brown = np.logspace(-1.32,-0.367,10) #in km
crat_diam_Halliday1 = np.logspace(-2.7,-2.15,10) #in km
crat_diam_Halliday2 = np.logspace(-2.15,-1.97,10) #in km

#crat_rad_Oberst = np.logspace()

freq_Revelle = []
freq_Brown = []
freq_Halliday1 = []
freq_Halliday2 = []
for i in range(10):
    freq_Revelle.append(Revelle_Rad2Freq((crat_diam_Revelle[i]/2)*1e3))
    freq_Brown.append(Brown_Rad2Freq((crat_diam_Brown[i]/2)*1e3))
    freq_Halliday1.append(Halliday1_Rad2Freq((crat_diam_Halliday1[i]/2)*1e3))
    freq_Halliday2.append(Halliday2_Rad2Freq((crat_diam_Halliday2[i]/2)*1e3))
    

#frequ4plot_micro2kg -> mass4plot: logspace(-6,3,100)
#frequ4plot_kg2higher
#frequ4plot_micro2kg = 10**poly_lin_micro2kg(np.log10(mass4plot_micro2kg))
#frequ4plot_kg2higher = 10**poly_lin_kg2higher(np.log10(mass4plot_kg2higher))

mass4plot_micro2g = np.logspace(-6,0,100)
frequ4plot_micro2g = 10**poly_lin_micro2kg(np.log10(mass4plot_micro2g))

# mass4plot_g2_500g = np.logspace(0,2.7,10)
# frequ4plot_g2_500g = 10**poly_lin_micro2kg(np.log10(mass4plot_g2_500g))

mass4plot_g2kg = np.logspace(0,3,10)
frequ4plot_g2kg = 10**poly_lin_micro2kg(np.log10(mass4plot_g2kg))

# mass4plot_500g2kg = np.logspace(2.7,3,10)
# frequ4plot_500g2kg = 10**poly_lin_micro2kg(np.log10(mass4plot_500g2kg))

mass4plot_kg2tera = np.logspace(3,16,100)
frequ4plot_kg2tera = 10**poly_lin_kg2higher(np.log10(mass4plot_kg2tera))

# mass4plot_mega2higher = np.logspace(6,20,100)
# freq4plot_mega2higher = 10**poly_lin_kg2higher(np.log10(mass4plot_mega2higher))


crat_diam_micro2g = []
for i in range(len(mass4plot_micro2g)):
    crat_diam_micro2g.append(Housen_Mass2CratDiam(mass4plot_micro2g[i])*1e-3) #SIMPLE & STRENGHT REGIME LAW -> D in km
    
crat_diam_g2kg = []
for i in range(len(mass4plot_g2kg )):
    crat_diam_g2kg.append(Housen_Mass2CratDiam(mass4plot_g2kg[i])*1e-3)    #SIMPLE & STRENGHT REGIME LAW -> D in km

crat_diam_kg2tera = []
for i in range(len(mass4plot_kg2tera)):
    crat_diam_kg2tera.append(Ivanov_Mass2CratDiam(mass4plot_kg2tera[i])*1e-5) #SIMPLE & GRAVITY REGIME LAW
    

#plt.close(figure(10))
figure10, ax = plt.subplots()

plt.loglog(crat_diam_Revelle,freq_Revelle,'k',label='Revelle, 2001',linewidth = 3.5)

plt.loglog(crat_diam_Brown,freq_Brown,'grey',label = 'Brown, 2002',linewidth = 3.5)

plt.loglog(crat_diam_Halliday1,freq_Halliday1,'orange',label='Halliday, 1996',linewidth = 3.5)
plt.loglog(crat_diam_Halliday2,freq_Halliday2,'orange',linewidth = 3.5)

plt.loglog(Schmidt_crat_part1,part1_frequ,'g',label='Schmidt, 1987',linewidth = 3.5)
#plt.loglog(Schmidt_crat_part2[0:60],part2_frequ[0:60],'g',linewidth = 3.5)
plt.loglog(Schmidt_crat_part2,part2_frequ,'g',linewidth = 3.5)


plt.loglog(crat_diam_micro2g,frequ4plot_micro2g,'b',label = 'Housen et al, 2011:$m_{met} \in [10^{-6}: 1]g$',linewidth = 3.5)
plt.loglog(crat_diam_g2kg,frequ4plot_g2kg ,'r',label = 'Housen et al, 2011:$m_{met} \in [1: 10^3]g$',linewidth = 3.5)
plt.loglog(crat_diam_kg2tera,frequ4plot_kg2tera ,'y',label = 'Ivanov, 2001:$m_{met} \in [10^3: 10^{12}]g$',linewidth = 3.5)

plt.axvline(x = 5e-3, color = 'k' ,linestyle = 'dashed',linewidth = 2)
plt.text(3e-3, 1e-16, 'Strength Regime', rotation=90, verticalalignment='bottom', color='k',size=13)
plt.text(6.5e-3, 1e-4, 'Gravity Regime', rotation=90, verticalalignment='bottom', color='k',size=13)

plt.axvline(x = 10, color = 'k' ,linestyle = 'dashed',linewidth = 2)
plt.text(6, 1e-16, 'Simple Crater', rotation=90, verticalalignment='bottom', color='k',size=13)
plt.text(11, 1e-4, 'Complex Crater', rotation=90, verticalalignment='bottom', color='k',size=13)


plt.xlabel('Crater Diameter - [km]',size=13)
plt.ylabel('Cumul. Frequency - $[km^{-2} h^{-1}$]',size=13)
plt.legend(loc='lower left',fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=13)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid( which='both')
plt.title('Cumulative Frequency of Meteorite Strike on the Moon',size=13)
plt.tight_layout()
pdf_path = '/Users/louis/Desktop/Thèse UCSD/Internships/LPI - Summer 2024/Codes/Matlab - Meteorites/Plots_meteorites/Validated Plots Bigger/CraterDiam_vs_Frequency.pdf'
#plt.savefig(pdf_path)

#%% SECTION III : EJECTA LAWS AND PLOTS

def Veje_GravityRegime_Housen(crater_radius,pos): #crater_radius in km
    #pos is in ]0 1] and reprensent how close from the impact center you want to compute the ejected velocity. a value of 1 would represent computing the ejected velocity at the radius of the crater
    crater_radius = crater_radius*1e3 #in m
    g = 1.62 #m/s^2
    C1 = 0.55
    H1 = 0.59
    mu = 0.41
    C2 = C1*((4*np.pi/3)**(1/3)*H1)**(-(2+mu)/(2*mu))
    Veje = (math.sqrt(g*crater_radius)*C2*(pos)**(-1/mu))*1e-3 #to get it in km/s
    return(Veje)

def Veje_StrengthRegime_Housen(crater_radius,pos): #crater_radius in km
    #crater_radius = crater_radius*1e3 #in m
    rho = 1.5e3 #kg/m^3
    Y = 1e4 #Pa 
    H2 = 0.81
    mu = 0.41
    C1 = 0.55
    C3 = C1*(((4*np.pi/3)**(1/3))*H2)**(-1/mu)
    Veje = ((Y/rho)**(1/2)*C3*(pos)**(-1/mu))*1e-3 #to get it in km/s
    return(Veje)

#pos = np.logspace(-2,0,11) #for all the computation of Veje
pos = [1e-2,10**(-1.823),10**(-1.5),1e-1,1]

crat_diam_1cm_5m = [1e-5,5e-5,1e-4,5e-4,1e-3,5e-3] #in km
crat_diam_6m_1km = [6e-3,1e-2,2e-2,5e-2,1e-1,2e-1,5e-1,1] #in km
Veje_crat_diam_1cm_5m = np.empty((len(pos),len(crat_diam_1cm_5m)))
Veje_crat_diam_6m_1km = np.empty((len(pos),len(crat_diam_6m_1km)))

for j in range(len(pos)):
    x_R = pos[j]
    for i in range(len(crat_diam_1cm_5m)):
        Veje_crat_diam_1cm_5m[j,i] = Veje_StrengthRegime_Housen(crat_diam_1cm_5m[i]/2,x_R)
    for i in range(len(crat_diam_6m_1km)):
        Veje_crat_diam_6m_1km[j,i] = Veje_GravityRegime_Housen(crat_diam_6m_1km[i]/2,x_R)
        
#%% SECTION III : Plots of Veje

#plt.close(figure(4))
figure4, ax = plt.subplots()

plt.loglog(pos[0:2],Veje_crat_diam_1cm_5m[0:2,0],linestyle='dashed',color='lightgrey')
plt.loglog(pos[1:len(pos)],Veje_crat_diam_1cm_5m[1:len(pos),4],label='$[1e^{-2} : 5]$m',linewidth = 3.5)


for i in range(len(crat_diam_6m_1km )):
    plt.loglog(pos[0:3],Veje_crat_diam_6m_1km[0:3,i],linestyle='dashed',color='lightgrey')
    plt.loglog(pos[2:len(pos)],Veje_crat_diam_6m_1km[2:len(pos),i],label=str(crat_diam_6m_1km[i])+'km',linewidth = 3.5)


  
plt.axhline(y = 2.4, color = 'k' ,linestyle = 'dashed',linewidth = 2)
plt.text(0.18,3, 'Moon Esc. Vel. $\sim 2.4km/s$', horizontalalignment='left', color='k',size=13)
plt.ylim([0,2e2])
plt.xlabel('x/R (Particle Position from Impact Center)',size=13)
plt.ylabel('Ejecta Velocity - [km/s]',size=13)
plt.legend(title='Crater Diameter',loc='lower left',fontsize=13)
ax.tick_params(axis='both', which='major', labelsize=13)
plt.grid( which='both')
plt.title('Maximum Ejecta Velocity Distribution by Particle Position',size=13)
plt.tight_layout()
pdf_path = '/Users/louis/Desktop/Thèse UCSD/Internships/LPI - Summer 2024/Codes/Matlab - Meteorites/Plots_meteorites/Validated Plots Bigger/EjectaVelocity_vs_xR.pdf'
#plt.savefig(pdf_path)


#%% Equation from range and max altitude

Ve = np.logspace(-4,0.36,100) #Ve changes from 10^-4 km/s to 10^4 km/s depending on the position of the ejecta but we will restrict to 2km/s 

def Range_gault_shoemaker(Ve):
    g = 1.62e-3 #km/s^2
    r_moon = 1737.4 #km
    Ve_bar = Ve**2/(r_moon*g)
    theta = np.pi/4  #set constant (Cintala + gault et shoemaker)
    Range = 2*r_moon*math.atan((Ve_bar*math.sin(theta)*math.cos(theta))/(1-Ve_bar*math.cos(theta)**2))
    return(Range)

def h_max(Ve):
    g = 1.62e-3 #km/s^2
    r_moon = 1737.4 #km
    Ve_bar = Ve**2/(r_moon*g)
    theta = np.pi/4 #set constant (Cintala + gault et shoemaker)
    h_max = r_moon * (Ve_bar - 1 + (1-Ve_bar *(2-Ve_bar)*math.cos(theta)**2)**(1/2))/(2-Ve_bar)
    return(h_max)

range_ve = np.empty(len(Ve))
h_max_ve = np.empty(len(Ve))

for i in range(len(Ve)):
        range_ve[i] = Range_gault_shoemaker(Ve[i])
        h_max_ve[i] = h_max(Ve[i])
        
#%% Plots of max range and max altitude 

#plt.close(figure(12))
figure12, ax = plt.subplots()
plt.loglog(Ve,range_ve,linewidth = 3.5)
plt.xlabel('Ejecta Velocity - $[km/s]$',size=13)
plt.ylabel('Particle Range - $[km]$',size=13)
ax.tick_params(axis='both', which='major', labelsize=13)
plt.grid( which='both')
plt.title('Particle Range distribution by Ejecta Velocity',size=13)  
plt.axvline(x = 2.4, color = 'k' ,linestyle = 'dashed',label='Escape Velocity ~$ 2.4km/s$',linewidth = 2)
plt.legend(fontsize=13)
plt.tight_layout()
pdf_path = '/Users/louis/Desktop/Thèse UCSD/Internships/LPI - Summer 2024/Codes/Matlab - Meteorites/Plots_meteorites/Validated Plots Bigger/ParticleRange_vs_EjectaVelocity.pdf'
#plt.savefig(pdf_path)

#plt.close(figure(13))
figure13, ax = plt.subplots()
plt.loglog(Ve,h_max_ve,linewidth = 3.5)
plt.xlabel('Ejecta Velocity - $[km/s]$',size=13)
plt.ylabel('Altitude max - $[km]$',size=13)
ax.tick_params(axis='both', which='major', labelsize=13)
plt.grid( which='both')
plt.title('Particle Maximum Altitude by Ejecta Velocity',size=13)  
plt.axvline(x = 2.4, color = 'k' ,linestyle = 'dashed',label='Escape Velocity ~$ 2.4km/s$',linewidth = 2)
plt.legend(fontsize=13)
plt.tight_layout()
pdf_path = '/Users/louis/Desktop/Thèse UCSD/Internships/LPI - Summer 2024/Codes/Matlab - Meteorites/Plots_meteorites/Validated Plots Bigger/ParticleAltitudeMax_vs_EjectaVelocity.pdf'
#plt.savefig(pdf_path)
  
#%% Mass ejected with speed> threshold -> check fig17 of Housen & Holsapple 2011


crater_radius_list = [2.5e-2/2 , 5e-2/2 , 1e-1/2 , 2.5e-1/2 , 5e-1/2 , 1/2] #km
crater_radius_list_1w = [1e-4/2 , 2e-4/2 , 3e-4/2 , 4e-4/2 , 5e-4/2 ,6e-4/2 , 7e-4/2 , 8e-4/2 , 9e-4/2, 1e-3/2,1.8e-3/2] #km
crater_radius_list_40y = [2e-3/2 , 3.4e-3/2 , 4e-3/2 , 5e-3/2 ,  6e-3/2 , 7e-3/2 , 8e-3/2, 9e-3/2, 1e-2/2, 1.5e-2/2 ]

g = 1.62e-3 #km/s^2 
rho_reg = 1.5e12 #kg/km^3

#From figure 17
x_axis = [1,10,100,300,1000,1500,1800]
y_axis = [1.5e-1,1e-2,5.5e-4,1.1e-4,2e-5,5e-6,1e-7]


V = np.empty((len(crater_radius_list),len(x_axis)))
Mass = np.empty((len(crater_radius_list),len(y_axis)))

V_1w = np.empty((len(crater_radius_list_1w),len(x_axis)))
Mass_1w = np.empty((len(crater_radius_list_1w),len(y_axis)))


V_40y = np.empty((len(crater_radius_list_40y),len(x_axis)))
Mass_40y = np.empty((len(crater_radius_list_40y),len(y_axis)))

# Threshold value
#threshold_velocity = 2.4



for i in range(len(crater_radius_list)):
    crater_radius = crater_radius_list[i]
    for j in range(len(x_axis)):
        V[i,j] = x_axis[j]*(g*crater_radius)**(1/2)
        Mass[i,j] = y_axis[j]*rho_reg*crater_radius**3 
 
     

#plt.close(figure(15))
figure15, ax = plt.subplots()

for i in reversed(range(len(crater_radius_list_1w))):
    crater_radius_1w = crater_radius_list_1w[i]
    for j in range(len(x_axis)):
        V_1w[i,j] = x_axis[j]*(g*crater_radius_1w)**(1/2)
        Mass_1w[i,j] = y_axis[j]*rho_reg*crater_radius_1w**3 
    
    #Plot 
    plt.loglog(V_1w[i,:],Mass_1w[i,:],label=str(round(crater_radius_list_1w[i]*2*1e3,2))+'m',linewidth = 3.5)
#plt.loglog(V_1w_after,Mass_1w_after,linestyle='dashed',color='grey')
plt.grid('on' , which='both')
plt.axvline(x = 2.4, color = 'k' ,linestyle = 'dashed',linewidth = 2)
plt.text(1.8, 5e-2, 'Escape Velocity', rotation=90, verticalalignment='bottom', color='k',size=13)
plt.xlim(0,2.8)
plt.legend(title='Crater Diameter',fontsize=13)
ax.tick_params(axis='both', which='major', labelsize=13)
plt.title('Ejected Mass by Ejecta Velocity',size=13) 
plt.xlabel('Ejecta Velocity - $[km/s]$',size=13)
plt.ylabel('Ejected Mass with $V_{ej}>V$ - $[kg]$',size=13)
plt.title('Ejected Mass by Ejecta Velocity',size=13)   
plt.tight_layout()
pdf_path = '/Users/louis/Desktop/Thèse UCSD/Internships/LPI - Summer 2024/Codes/Matlab - Meteorites/Plots_meteorites/Validated Plots Bigger/EjectedMass_vs_EjectaVelocity_Drange(0.1m_1m).pdf'
#plt.savefig(pdf_path)    

    
#plt.close(figure(16))
figure16, ax = plt.subplots()

for i in reversed(range(len(crater_radius_list_40y))):
    crater_radius_40y = crater_radius_list_40y[i]
    for j in range(len(x_axis)):
        V_40y[i,j] = x_axis[j]*(g*crater_radius_40y)**(1/2)
        Mass_40y[i,j] = y_axis[j]*rho_reg*crater_radius_40y**3 
        
    plt.loglog(V_40y[i,:],Mass_40y[i,:],label=str(round(crater_radius_list_40y[i]*2*1e3,2))+'m',linewidth = 3.5)
    
plt.grid('on' , which='both')
plt.legend(title='Crater Diameter',fontsize=13)
plt.title('Ejected Mass by Ejecta Velocity',size=13) 
plt.xlabel('Ejecta Velocity - $[km/s]$',size=13)
plt.ylabel('Ejected Mass with $V_{ej}>V$ - $[kg]$',size=13)
plt.title('Ejected Mass by Ejecta Velocity',size=13)  
ax.tick_params(axis='both', which='major', labelsize=13)
plt.axvline(x = 2.4, color = 'k' ,linestyle = 'dashed',linewidth = 2)
plt.text(1.95, 1e2, 'Escape Velocity', rotation=90, verticalalignment='bottom', color='k',size=13)
plt.xlim(0,2.8)
plt.tight_layout()
pdf_path = '/Users/louis/Desktop/Thèse UCSD/Internships/LPI - Summer 2024/Codes/Matlab - Meteorites/Plots_meteorites/Validated Plots Bigger/EjectedMass_vs_EjectaVelocity_Drange(2m_15m).pdf'
#plt.savefig(pdf_path)




#plt.close(figure(14))
figure14, ax = plt.subplots()
for i in reversed(range(len(crater_radius_list))):
    plt.loglog(V[i,:],Mass[i,:],label=str(crater_radius_list[i]*2)+'km',linewidth = 3.5)

plt.xlabel('Ejecta Velocity - $[km/s]$',size=13)
plt.ylabel('Ejected Mass with $V_{ej}>V$ - $[kg]$',size=13)
plt.legend(title='Crater Diameter',loc='lower left',fontsize=13)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.axvline(x = 2.4, color = 'k' ,linestyle = 'dashed',linewidth = 2)
plt.text(1.95, 1e-1, 'Escape Velocity', rotation=90, verticalalignment='bottom', color='k',size=13)
ax.tick_params(axis='both', which='major', labelsize=13)
plt.xlim(0,2.8)
plt.grid( which='both')
plt.title('Ejected Mass by Ejecta Velocity',size=13) 
plt.tight_layout()
pdf_path = '/Users/louis/Desktop/Thèse UCSD/Internships/LPI - Summer 2024/Codes/Matlab - Meteorites/Plots_meteorites/Validated Plots Bigger/EjectedMass_vs_EjectaVelocity_Drange(25m_1km).pdf'
#plt.savefig(pdf_path)

#%% Plot of radius projectile vs crater radius 

m_projectile = [1e-6,1,1e3,1e12,3.1e15] #in g
a = np.empty(len(m_projectile))
for i in range(len(m_projectile)):
    rho_met = 3.587 #g/cm^3 determined from (Dalton, 1972)
    a[i] = ((3*m_projectile[i]/(4*np.pi*rho_met))**(1/3))*1e-2 #in m

crat_radius = [5e-6/2,5e-4/2,5.5e-3/2,1.45/2,20/2]


#plt.close(figure(30))
figure30, ax = plt.subplots()
plt.axhline(y = 2.5e-3, color = 'k' ,linestyle = 'dashed')
plt.text(6e-5, 1.2e-3, 'Strength Regime', verticalalignment='bottom', color='k',size=13)
plt.text(1, 2.7e-3, 'Gravity Regime', verticalalignment='bottom', color='k',size=13)
ax.tick_params(axis='both', which='major', labelsize=13)
plt.loglog(a,crat_radius,linewidth = 3.5)
plt.xlabel('Impactor Radius - $[m]$',size=13)
plt.ylabel('Crater Radius - $[km]$',size=13)
plt.grid(which='both')
plt.title('Crater Radius Production by Impactor Radius',size=13)
plt.tight_layout()
pdf_path = '/Users/louis/Desktop/Thèse UCSD/Internships/LPI - Summer 2024/Codes/Matlab - Meteorites/Plots_meteorites/Validated Plots Bigger/ImpactorRadius_vs_CraterRadius.pdf'
#plt.savefig(pdf_path)


