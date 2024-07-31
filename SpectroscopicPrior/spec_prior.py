import pickle as pkl
import numpy as np
import os
from scipy.optimize import curve_fit
#import pyred as py
import matplotlib
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.stats import kde
from scipy.interpolate import interpolate
import corner

mpl.rcParams['axes.labelweight']='semibold'
mpl.rcParams['mathtext.fontset']='stix'
mpl.rcParams['font.weight'] = 'semibold'
mpl.rcParams['axes.titleweight']='semibold'

mpl.rcParams['lines.linewidth']   = 2
mpl.rcParams['axes.linewidth']    = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5

mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['axes.labelsize']  = 12
mpl.rcParams['axes.titlesize']  = 16

path = '../Data/'

f_names = np.array(['Columba_spec1.csv','Columba_spec2.csv',
                    'Columba_spec3.csv','Columba_spec4.csv',
                    'Columba_spec5.csv','Columba_spec6.csv',
                    'Columba_spec7.csv','Columba_spec8.csv'])



RR_j = np.empty(0)
SBR_T_j = np.empty(0)
SBR_I_j = np.empty(0)
SBR_rp_j = np.empty(0)

T1_j = np.empty(0)
T2_j = np.empty(0)
R1_j = np.empty(0)
R2_j = np.empty(0)

pp = PdfPages('spec_prior.pdf')

for i in range(f_names.size):
	T1,T2,R1,R2,SBR_T,SBR_I,SBR_rp = np.loadtxt(path+f_names[i],unpack=True,delimiter=',')

	RR_j = np.append(RR_j,(R2/R1)[-20000:])
	SBR_T_j = np.append(SBR_T_j,SBR_T[-20000:])
	SBR_I_j = np.append(SBR_I_j,SBR_I[-20000:])
	SBR_rp_j = np.append(SBR_rp_j,SBR_rp[-20000:])
	R1_j = np.append(R1_j,R1[-20000:])
	R2_j = np.append(R2_j,R2[-20000:])
	T1_j = np.append(T1_j,T1[-20000:])
	T2_j = np.append(T2_j,T2[-20000:])

	fig,ax = plt.subplots(1)
	
	ax.plot(R2/R1,SBR_T,'o',color='grey',alpha=0.1,rasterized=True)
	
	ax.axhline(1.0,ls='--',color='k')
	ax.axvline(1.0,ls='--',color='k')
	
	ax.set_xlabel('$R_2/R_1$')
	ax.set_ylabel('$J_2/J_1$')
	ax.set_title('Epoch '+str(i))

	plt.tight_layout()
	pp.savefig()
	
	fig,ax = plt.subplots(7)
	
	ax[0].plot(T1,rasterized=True)
	ax[1].plot(T2,rasterized=True)
	ax[2].plot(R1,rasterized=True)
	ax[3].plot(R2,rasterized=True)
	ax[4].plot(SBR_T,rasterized=True)
	ax[5].plot(SBR_I,rasterized=True)
	ax[6].plot(SBR_rp,rasterized=True)
	
	ax[0].set_ylabel('T1')
	ax[1].set_ylabel('T2')
	ax[2].set_ylabel('R1')
	ax[3].set_ylabel('R2')
	ax[4].set_ylabel('SBR_T')
	ax[5].set_ylabel('SBR_I')
	ax[6].set_ylabel('SBR_rp')
	
	ax[6].set_xlabel('N')
	
	plt.tight_layout()
	pp.savefig()
	
	fig,ax = plt.subplots(7,sharex=True,figsize=(6.4,7*(4.8/5.0)))
	
	ax[0].plot(T2/T1,rasterized=True)
	ax[1].plot(R2/R1,rasterized=True)
	ax[2].plot((R2**2*T2**4)/(R1**2*T1**4),rasterized=True)
	ax[3].plot(T2**4/T1**4,rasterized=True)
	ax[4].plot(SBR_T,rasterized=True)
	ax[5].plot(SBR_I,rasterized=True)
	ax[6].plot(SBR_rp,rasterized=True)
	
	ax[0].axhline(1,color='k')
	ax[1].axhline(1,color='k')
	ax[2].axhline(1,color='k')
	ax[3].axhline(1,color='k')
	ax[4].axhline(1,color='k')
	ax[5].axhline(1,color='k')
	ax[6].axhline(1,color='k')
	
	ax[0].set_ylabel('T2/T1')
	ax[1].set_ylabel('R2/R1')
	ax[2].set_ylabel('<F2/F1>')
	ax[3].set_ylabel('~SBR')
	ax[4].set_ylabel('SBR_T')
	ax[5].set_ylabel('SBR_I')
	ax[6].set_ylabel('SBR_rp')
	
	ax[6].set_xlabel('N')
	
	plt.tight_layout()
	pp.savefig()
	plt.close('all')

fig,ax = plt.subplots(1)
	
ax.plot(RR_j,SBR_T_j,'o',color='grey',alpha=0.1,rasterized=True)

ax.axhline(1.0,ls='--',color='k')
ax.axvline(1.0,ls='--',color='k')

ax.set_xlabel('$R_2/R_1$')
ax.set_ylabel('$J_2/J_1$')
ax.set_title('Joined (TESS)')

plt.tight_layout()
pp.savefig()

fig,ax = plt.subplots(1)
	
ax.plot(RR_j,SBR_rp_j,'o',color='grey',alpha=0.1,rasterized=True)

ax.axhline(1.0,ls='--',color='k')
ax.axvline(1.0,ls='--',color='k')

ax.set_xlabel('$R_2/R_1$')
ax.set_ylabel('$J_2/J_1$')
ax.set_title('Joined (rp)')

plt.tight_layout()
pp.savefig()

fig,ax = plt.subplots(1)
	
ax.plot(RR_j,SBR_I_j,'o',color='grey',alpha=0.1,rasterized=True)

ax.axhline(1.0,ls='--',color='k')
ax.axvline(1.0,ls='--',color='k')

ax.set_xlabel('$R_2/R_1$')
ax.set_ylabel('$J_2/J_1$')
ax.set_title('Joined (IS)')

plt.tight_layout()
pp.savefig()

nbins=100
x = RR_j
y = SBR_T_j
k = kde.gaussian_kde([x,y],bw_method=0.2)
xi, yi = np.mgrid[np.min(x):np.max(x):nbins*1j, np.min(y):np.max(y):nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

ks = kde.gaussian_kde([x,y])
xis, yis = np.mgrid[np.min(x):np.max(x):nbins*1j, np.min(y):np.max(y):nbins*1j]
zis = ks(np.vstack([xis.flatten(), yis.flatten()]))

y_T = SBR_T_j
kde_rr_sbr_T = kde.gaussian_kde([x,y_T])
y_I = SBR_I_j
kde_rr_sbr_I = kde.gaussian_kde([x,y_I])
y_rp = SBR_rp_j
kde_rr_sbr_rp = kde.gaussian_kde([x,y_rp])

pkl.dump(kde_rr_sbr_T,open('kde_rr_sbr_T.p','wb'))
pkl.dump(kde_rr_sbr_I,open('kde_rr_sbr_I.p','wb'))
pkl.dump(kde_rr_sbr_rp,open('kde_rr_sbr_rp.p','wb'))

kde_rr_sbr_T = pkl.load(open('kde_rr_sbr_T.p','rb'))
kde_rr_sbr_I = pkl.load(open('kde_rr_sbr_I.p','rb'))
kde_rr_sbr_rp = pkl.load(open('kde_rr_sbr_rp.p','rb'))

#T1_lo,T1_med,T1_hi = np.percentile(T1,[16,50,84])
#T2_lo,T2_med,T2_hi = np.percentile(T2,[16,50,84])
#
#print(T1_med,T1_hi-T1_med,T1_med-T1_lo)
#print(T2_med,T2_hi-T2_med,T2_med-T2_lo)

fig,ax = plt.subplots(1)

ax.hist2d(x,y,bins=[200,200],cmap='bone_r')

ax.set_xlabel('$R_2/R_1$')
ax.set_ylabel('$J_2/J_1$')

ax.set_xlim(0.9,1.2)
ax.set_ylim(0.8,1.2)

plt.tight_layout()
pp.savefig()

print('R2/R1:',np.percentile(x,50),np.percentile(x,84)-np.percentile(x,50),np.percentile(x,50)-np.percentile(x,16))
print('J2/J1 (TESS):',np.percentile(y_T,50),np.percentile(y_T,84)-np.percentile(y_T,50),np.percentile(y_T,50)-np.percentile(y_T,16))
print('J2/J1 (rp):',np.percentile(y_rp,50),np.percentile(y_rp,84)-np.percentile(y_rp,50),np.percentile(y_rp,50)-np.percentile(y_rp,16))
print('J2/J1 (I):',np.percentile(y_I,50),np.percentile(y_I,84)-np.percentile(y_I,50),np.percentile(y_I,50)-np.percentile(y_I,16))

print('T1:',np.percentile(T1_j,50),np.percentile(T1_j,84)-np.percentile(T1_j,50),np.percentile(T1_j,50)-np.percentile(T1_j,16))
print('T2:',np.percentile(T2_j,50),np.percentile(T2_j,84)-np.percentile(T2_j,50),np.percentile(T2_j,50)-np.percentile(T2_j,16))
print('R1:',np.percentile(R1_j,50),np.percentile(R1_j,84)-np.percentile(R1_j,50),np.percentile(R1_j,50)-np.percentile(R1_j,16))
print('R2:',np.percentile(R2_j,50),np.percentile(R2_j,84)-np.percentile(R2_j,50),np.percentile(R2_j,50)-np.percentile(R2_j,16))


fig,ax = plt.subplots(1)

ax.contourf(xi, yi, zi.reshape(xi.shape),zorder=1,cmap='bone_r')

ax.set_xlabel('$R_2/R_1$')
ax.set_ylabel('$J_2/J_1$')

plt.tight_layout()
pp.savefig()

fig,ax = plt.subplots(1)

ax.contourf(xis, yis, zis.reshape(xis.shape),zorder=1,cmap='bone_r')

ax.set_xlabel('$R_2/R_1$')
ax.set_ylabel('$J_2/J_1$')

plt.tight_layout()
pp.savefig()

fig = plt.figure()

ax = plt.axes(projection='3d')

ax.contour3D(xi, yi, zi.reshape(xi.shape),50,cmap='bone_r')

ax.set_xlabel('$R_2/R_1$')
ax.set_ylabel('$J_2/J_1$')
ax.set_zlabel('KDE')

plt.tight_layout()
pp.savefig()
fig = plt.figure()

ax = plt.axes(projection='3d')

ax.contour3D(xis, yis, zis.reshape(xis.shape),50,cmap='bone_r')

ax.set_xlabel('$R_2/R_1$')
ax.set_ylabel('$J_2/J_1$')
ax.set_zlabel('KDE')

plt.tight_layout()
pp.savefig()

pp.close()



fig,ax = plt.subplots(2,figsize=(4,6),dpi=600,sharex=True,sharey=True)

ax[1].plot(RR_j,SBR_T_j,'.',color='grey',alpha=0.1,rasterized=True,ms=1,zorder=0)

H,X_hist,Y_hist = np.histogram2d(RR_j,SBR_T_j,bins=30)

X_hist = (X_hist + (X_hist[1]-X_hist[0])/2.0)[:-1]
Y_hist = (Y_hist + (Y_hist[1]-Y_hist[0])/2.0)[:-1]

H=H/H.sum()

n = 1000
t = np.linspace(0, H.max(), n)
integral = ((H >= t[:, None, None]) * H).sum(axis=(1,2))
f = interpolate.interp1d(integral, t)

t_contours = f(np.array([0.997, 0.95, 0.68, 0.5]))

ax[1].contour(X_hist,Y_hist,H.T,levels=t_contours,colors='k')

ax[1].set_xlabel('$R_2/R_1$')
ax[0].set_ylabel('$J_2/J_1 \ (TESS)$')
ax[1].set_ylabel('$J_2/J_1 \ (TESS)$')

ax[0].set_xlim(0.8,1.4)
ax[0].set_ylim(0.6,1.3)

ax[0].text(0.815,0.62,'Individual Epochs')
ax[1].text(0.815,0.62,'Combined Posterior')

for i in range(f_names.size):
	T1,T2,R1,R2,SBR_T,SBR_I,SBR_rp = np.loadtxt(path+f_names[i],unpack=True,delimiter=',')

	RR = R2/R1

	H,X_hist,Y_hist = np.histogram2d(RR,SBR_T,bins=30)

	X_hist = (X_hist + (X_hist[1]-X_hist[0])/2.0)[:-1]
	Y_hist = (Y_hist + (Y_hist[1]-Y_hist[0])/2.0)[:-1]

	H=H/H.sum()

	n = 1000
	t = np.linspace(0, H.max(), n)
	integral = ((H >= t[:, None, None]) * H).sum(axis=(1,2))
	f = interpolate.interp1d(integral, t)
	
	t_contours = f(np.array([0.68]))
	
	ax[0].contour(X_hist,Y_hist,H.T,levels=t_contours,colors='k',alpha=0.5)

plt.tight_layout(pad=0.3)
plt.savefig('spec_prior_paper.pdf',dpi=600)
plt.close()


#Corner Plot
labels = ['$T_{eff,1} (K)$', '$T_{eff,2} (K)$', '$R_1 (R_\odot)$', '$R_2 (R_\odot)$']

corner_samples = np.reshape(np.concatenate([T1_j,T2_j,R1_j,R2_j]),(T1_j.size,4),order='F')

mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['xtick.labelsize'] = 11

fig = corner.corner(corner_samples, labels=labels, 
            		plot_density=False,plot_contours=True,
            		data_kwargs={'ms':0.5,'alpha':0.1,'color':'grey'},
            		color='k',truth_color='k',quantiles=(0.16, 0.50, 0.84),
            		show_titles=False,rasterized=True,
            		fill_contours=True, title_kwargs={"fontsize": 14},
            		title_fmt='.2f',hist_kwargs={"linewidth": 2.5})
#plt.tight_layout(pad=0.1)
plt.savefig('spec_corner_paper.pdf',dpi=600)
plt.close()




#pkl.dump([(T2/T1)[-10000:],(R2/R1)[-10000:],((R2**2*T2**4)/(R1**2*T1**4))[-10000:],(T2**4/T1**4)[-10000:],(SBR_T)[-10000:],(SBR_I)[-10000:],(SBR_rp)[-10000:]],open('spec_prior_dev_tail.p','wb'))

#fig,ax = plt.subplots(2,figsize = (8,5),dpi=600)

# -- I crop this Figure before uploading so I don't want to remake it every time I run this code for other reasons -- # 
#fig = plt.figure(figsize=(5,4))
#
#ax = plt.axes(projection='3d')
#
#ax.contour3D(xi, yi, zi.reshape(xi.shape),50,cmap='bone_r')
#
#ax.set_xlabel('$R_2/R_1$',labelpad=15)
#ax.set_ylabel('$J_2/J_1$',labelpad=15)
#ax.set_zlabel('KDE',labelpad=15)
#
#ylabels = ax.get_yticks()
#print(ylabels)
#for i in range(ylabels.size):
#	ylabels[i] = '{:1.2f}'.format(np.float(ylabels[i]))
#ax.set_yticklabels(ylabels,va='center', ha='left',rotation=-15)
##ax.set_yticklabels(ax.get_yticks(), rotation = -45)
#
##plt.tight_layout(pad=0.1)
#plt.savefig('KDE.png',dpi=600)
#plt.close()


