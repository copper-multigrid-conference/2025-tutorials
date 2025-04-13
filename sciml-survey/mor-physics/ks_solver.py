# To run: pip install numpy scipy matplotlib

# Markdown description
##################################
#
# # Kuramoto–Sivashinsky Equation solver
#
# From Ravi Patel (SNL): https://proceedings.mlr.press/v190/patel22a/patel22a.pdf
#
# $$ u_t = - C_1 u_{xx} - C_2 u_{xxxx} - C_3 (u^2)_x $$
#
# or
#
# $$ u_t = \mathcal{L}(u) + \mathcal{N}(u) $$
# where
# $$\mathcal{L}(u) = -C_1 u_{xx} - C_2 u_{xxxx} $$ 
# $$\mathcal{N}(u) = - C_3 (u^2)_x $$
#
# ## Low storage RK-3 Method by Spalart, Moser and Rogers (1991)
# https://doi.org/10.1016/0021-9991(91)90238-G
#
# $$ 
# \begin{aligned}
# u' &= u_n + \Delta t [ \mathcal{L}(\alpha_1 u_n + \beta_1 u') + \gamma_1 \mathcal{N}_n] \\
# u'' &= u' + \Delta t [ \mathcal{L}(\alpha_2 u' + \beta_2 u'') + \gamma_2 \mathcal{N}' + \zeta_1 N_n] \\
# u_{n+1} &= u'' + \Delta t [ \mathcal{L}(\alpha_3 u'' + \beta_3 u_{n+1}) + \gamma_3 \mathcal{N}'' + \zeta_2 \mathcal{N}'] 
# \end{aligned}
# $$
#
# $$
# \begin{matrix}
# \alpha_1 = \frac{29}{96} & \alpha_2 = -\frac{3}{40} & \alpha_3 = \frac16 \\
# \beta_1 = \frac{37}{160} & \beta_2 = \frac{5}{24} & \beta_3 = \frac16\\
# \gamma_1 = \frac{8}{15} & \gamma_2 = \frac{5}{12} & \gamma_3 = \frac34 \\
# \zeta_1 = - \frac{17}{60} & \zeta_2 = -\frac{5}{12}
# \end{matrix}
# $$
#
# $$ \widehat{u} = \mathcal{F}(u) $$
# 
# 
# $$ \widehat{u}_t = C_1 k^2 \widehat{u}_t  - C_2k^4 \widehat{u}_t  - C_3 ik \widehat{u^2} $$
#
# **NOTE**
#
# > Number of grid points should be a multiplication of "4". e.g. 4,8,12,16,...
##################################
import numpy as np
import scipy.special as special
from numpy.fft import rfft,irfft

#Calculate nonlinear term with 3/2 dealiasing
def cal_nonlinear(u,k):
    Np = len(u)
    pad_size = int((Np-1)*1.5) + 1
    u_pad = np.zeros((pad_size,),dtype=u.dtype)
    u_pad[:Np-1] = u[:Np-1]
    u_sq_pad = rfft(irfft(u_pad)**2)
    u_sq = np.zeros(Np,dtype=u.dtype)
    u_sq[:Np] = u_sq_pad[:Np]
    u_sq[-1] = 0.0
    
    return 1j*k*u_sq*1.5

def step(u,dt,kx,C1,C2,C3):
    """
    Produces the time update in fourier space using
    a low storage RK3 scheme. 

    u - initial condition in fourier space
    dt - time step
    kx - wave numbers method is discretized on
    C1,C2,C3 - model parameters

    returns time evolved solution in fourier space.
    """
    
    alpha1 = 29./96.; alpha2=-3./40.; alpha3 = 1./6.
    beta1 = 37./160.; beta2 = 5./24.; beta3 = 1./6.
    gamma1= 8./15.; gamma2 = 5./12.; gamma3 = 3./4.
    zeta1 = -17./60.; zeta2 = -5./12.
    
    N    = -C3*cal_nonlinear(u,kx)
    normalization = 1./ (1-dt*beta1*(C1*kx**2 - C2*kx**4))
    u_p  = normalization*((1 + dt*alpha1*(C1*kx**2 - C2*kx**4))*u    + dt*gamma1 * N)
    
    N_p  = -C3*cal_nonlinear(u_p,kx)
    normalization = 1./ (1-dt*beta2*(C1*kx**2 - C2*kx**4))
    u_pp = normalization*((1 + dt*alpha2*(C1*kx**2 - C2*kx**4))*u_p  + dt*gamma2 * N_p  + dt*zeta1*N)

    N_pp = -C3*cal_nonlinear(u_pp,kx)
    normalization = 1./ (1-dt*beta3*(C1*kx**2 - C2*kx**4))
    u_n  = normalization*((1 + dt*alpha3*(C1*kx**2 - C2*kx**4))*u_pp + dt*gamma3 * N_pp + dt*zeta2*N_p)

    return u_n

def generate_trajectory():
  Lx = np.pi*16 # Domain size
  N = 256  # Better to be even number

  kx = np.arange(int(N/2)+1)*np.pi*2/Lx # Wavenumber

  C1 = 1.
  C2 = 1.
  C3 = 0.5 
  dt = 0.01
  Nt = 10000

  u_data = np.zeros((Nt,N))
  u = np.random.random(N)

  mode_scaling = special.erfc(10.*(kx-1.))
  u_hat = rfft(u)*mode_scaling
  T = Nt*dt
  for i in range(Nt):
      u_hat = step(u_hat,dt,kx,C1,C2,C3)
      u_data[i] = irfft(u_hat)

  x = np.arange(N)*Lx/N # grid points
  return x, u_data

def generate_data(sub_sample=1):
  results = []
  for _ in range(10):
    x,u_data = generate_trajectory()
    results.append(np.expand_dims(u_data[::sub_sample],-1))
  
  results = np.stack(results)
  return x,results

def main():
  import pickle 

  # load up sample data
  filename = 'ks_samples.pkl'
  
  x,ks_sims = generate_data()
  with open(filename, 'wb') as file:
    pickle.dump([x,ks_sims], file)

if __name__=='__main__':
  main()
