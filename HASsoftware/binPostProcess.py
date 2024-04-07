# Read MCMC output from a file, then compute and plot the auto-covariance function

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import acor
import ChainOutput
import time

# performance of post-processing improved by reading the MCMC output from a binary file

chain_test      = True
thetas_test     = True
CV_test         = True

start = time.time()
# read chain from binary file, cast it to vector called X
dtype = np.dtype('float64')
try:
   with open("chain.bin", "rb") as f:
      X = np.fromfile(f,dtype)
except IOError:
   chain_test = False
   
   
# read thetas from binary file, cast it to vector called theta
dtype = np.dtype('double')
try:
   with open("thetas.bin", "rb") as f:
       theta = np.fromfile(f,dtype)
       theta = theta[ theta != 0 ]
except IOError:
   thetas_test = False
   
   
# read chainCV from binary file, cast it to vector called CV
dtype = np.dtype('double')
try:
   with open("chainCV.bin", "rb") as f:
       CV = np.fromfile(f,dtype)
       CV = CV[ CV != 0 ]
except IOError:
   CV_test = False
   
   
   
   
Ts = ChainOutput.Ts
d  = ChainOutput.d
X = np.reshape(X, (Ts,d))
end = time.time()



print("---------------------------------------------------")
OutputLine = " chain has " + str(Ts) + " rows and " + str(d) + " columns "
print(OutputLine)
print(" Time to read chain from binary =  %.4f seconds" %(end-start))
eps = ChainOutput.eps
beta_eps = 1.0 / (2.0*eps*eps)
ac  = acor.acov(X[:,0])
tau = acor.act(ac, Nmin = 20, w = 7)
print(" tau = %.2f" %tau)
print("---------------------------------------------------")

np1 = 5*int(tau)

#-------------------------------Plot of Autocovariance-------------------------
if (chain_test==True):
   fig, ax = plt.subplots()
   ax.plot(range(np1), ac[0:np1], 'bo', label = 'auto-covariance')
   ax.set_ylabel('covariance')
   ax.set_xlabel('lag')
   title = ChainOutput.ModelName
   ax.set_title(title)
   ax.grid()

   ymin, ymax = ax.get_ylim()
   tauLabel = 'tau = {0:8.2f}'.format(tau)
   ax.vlines(tau,  ymin, ymax, label = tauLabel, colors='r')
   ax.legend()

   runInfo = r"$\beta_{\varepsilon}=\frac{1}{2\varepsilon^2}=%.1f$" % (beta_eps) + ", "
   runInfo = runInfo + r"$\varepsilon=%.3f$" % (ChainOutput.eps) + "\n"
   runInfo = runInfo + r"$N_{soft}=%d$" % (ChainOutput.Nsoft) + ", "
   runInfo = runInfo + r"$N_{rattle}=%d$" % (ChainOutput.Nrattle) + "\n"
   runInfo = runInfo + r"$s_q=%.2f$" % (ChainOutput.kq) + r"$\varepsilon$" + ", "
   runInfo = runInfo + r"$s_p=%.2f$" % (ChainOutput.kp) + r"$\varepsilon$"  + ", "
   runInfo = runInfo + r"$\Delta t=%.2f$" % (ChainOutput.dt)  + "\n"
   runInfo = runInfo + r"$A_s=%.3f$" % (ChainOutput.As) + ", "
   runInfo = runInfo + r"$A_r=%.3f$" % (ChainOutput.Ar) + "\n"
   runInfo = runInfo + r"$T_s=%d$" % (Ts) + ", "
   runInfo = runInfo + r"$T=%d$" % (ChainOutput.T)

   # these are matplotlib.patch.Patch properties
   props = dict(boxstyle='round', facecolor='white', alpha=0.8)

   # place a text box in upper left in axes coords
   ax.text(0.55, 0.8, runInfo, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)


   plt.savefig('AutoCovariance.pdf')
   #plt.show()





#-------------------------------CV histogram-----------------------------
if (CV_test==True):
   nCVbins = 80
   beta_q = ChainOutput.beta_q  # Assuming this comes from somewhere
   beta_s = ChainOutput.beta_s  # Assuming this comes from somewhere
   T_q = 1. / beta_q
   T_s = 1./ beta_s
   
   # Free Energy parameters
   D0      = ChainOutput.D0
   a       = ChainOutput.a
   kappa   = ChainOutput.kappa
   lambda_ = ChainOutput.lambda_

   # Compute the histogram of CV values
   hist, bin_edges = np.histogram(CV, bins=nCVbins, density=True)

   # Calculate the midpoints of the bins
   bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

   # Calculate the estimated free energy and normalize so the minimum is at zero
   F_hat = -(1. / beta_q) * np.log(hist)
   F_hat -= np.min(F_hat)

   # Define the true free energy profile function
   def F(x):
       return D0 * (x*x - a*a)**2 - (lambda_**2 * x*x / (2.*kappa))

   # Calculate the true free energy at the bin midpoints and normalize
   true_F = F(bin_midpoints)
   true_F -= np.min(true_F)

   # Compute the error between the estimated and true free energy at bin midpoints
   error = np.abs(true_F - F_hat)

   # Summarize the error into a single percentage number
   mae = np.mean(error)  # Mean Absolute Error
   average_true_F = np.mean(true_F)
   error_percentage = (mae / average_true_F) * 100.

   # Print the summarized error percentage
   print(" ")
   print(f"Relative Absolute Error: {error_percentage:.2f}%")
   print(" ")

   # Plot the estimated and true free energy profiles
   fig, ax = plt.subplots(figsize=(10, 8))
   ax.plot(bin_midpoints, F_hat, label='Estimated Free Energy')
   ax.plot(bin_midpoints, true_F, label='True Free Energy', linestyle='--')
   ax.set_xlabel('CV')
   ax.set_ylabel('Free Energy')
   ax.set_title('Estimated and True Free Energy Profiles')

   # Add annotation for the number of samples and relative error
   samples_text  = f"Number of Samples: {ChainOutput.Ts}"
   error_text    = f"Relative Error: {error_percentage:.2f}%"
   beta_eps_text = f"Constraining Potential Inverse Temperature  $\\beta_{{\\varepsilon}}$: {beta_eps:.2f}"
   beta_q_text   = f"Physical Temperature  $T$: {T_q:.2f}"
   beta_s_text   = f"Artificial Temperature  $T_s$: {T_s:.2f}"
   tau_text      = f"Auto-correlation Time  $\\tau$: {tau:.2f}"
   ax.text(0.1, 0.99, samples_text, transform=ax.transAxes, verticalalignment='top')
   ax.text(0.1, 0.96, error_text, transform=ax.transAxes, verticalalignment='top')
   ax.text(0.1, 0.93, beta_eps_text, transform=ax.transAxes, verticalalignment='top')
   ax.text(0.1, 0.90, beta_q_text, transform=ax.transAxes, verticalalignment='top')
   ax.text(0.1, 0.87, beta_s_text, transform=ax.transAxes, verticalalignment='top')
   ax.text(0.1, 0.84, tau_text, transform=ax.transAxes, verticalalignment='top')
      
   ax.legend(loc='upper right')
   plt.savefig('freeEnergy.pdf')  # Save the figure
   plt.close(fig)  # Close the figure

   # Plot the histogram itself
   fig, ax = plt.subplots(figsize=(10, 8))
   ax.bar(bin_midpoints, hist, width=(bin_edges[1] - bin_edges[0]), alpha=0.75, color='blue')
   ax.set_xlabel('CV')
   ax.set_ylabel('Probability Density of CV')
   ax.set_title('Histogram of CV')
   
   # Add annotation for the number of samples to the histogram plot
   ax.text(0.35, 0.99, samples_text, transform=ax.transAxes, verticalalignment='top')
   ax.text(0.35, 0.96, beta_eps_text, transform=ax.transAxes, verticalalignment='top')
   ax.text(0.35, 0.93, beta_q_text, transform=ax.transAxes, verticalalignment='top')
   ax.text(0.35, 0.90, beta_s_text, transform=ax.transAxes, verticalalignment='top')
   ax.text(0.35, 0.87, tau_text, transform=ax.transAxes, verticalalignment='top')

   plt.savefig('CVhist.pdf')  # Save the histogram figure
   plt.close(fig)  # Close the figure
#-------------------------------------------------------------------------------------------


#-------------------------------Theta histogram-----------------------------
if (thetas_test==True):
   ntbins = 100
   truepdf   = 1. / (math.pi)

   fig, ax = plt.subplots(figsize=(10,12))
   ax.hist(theta, ntbins, density=True, alpha=0.5, histtype='bar', ec='black', label=r"$\theta$ sample distr.")
   ax.set_ylabel('Density', fontsize=25)
   ax.set_xlabel(r'$\theta$', fontsize=25)
   ax.set_xlim(left=theta.min(), right=theta.max())
   plt.axhline(y = truepdf, color = 'r', linestyle = '-', label=r"$\theta$ theoretical pdf")
   ax.legend(bbox_to_anchor=(0.755, 1.14), fontsize=22)
   ax.tick_params(axis='both', which='major', labelsize=15)
   ax.tick_params(axis='both', which='minor', labelsize=15)

   plt.savefig('ThetaCount.pdf')



with open('output', 'w') as OutputFile:
   OutputFile.write('estimated tau is {0:8.2f}'.format(tau))
   






# Check that values agree with old PostProcess.py
#for i in range(0, 20):
   #for j in range(0,3):
      #print('X[    %.d' %i +',    %.d' %j + '] = %.8f' %X[i,j] )

