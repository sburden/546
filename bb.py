
from systems import Hybrid

import numpy as np
import pylab as plt

import matplotlib 
from matplotlib import rc
#
# Say, "the default sans-serif font is Helvetica"
rc('font',**{'sans-serif':'Helvetica','family':'sans-serif','size':12})
rc('text',usetex=False)

class BouncingBall(Hybrid):

  def __init__(self):
    """
    bouncing ball hybrid system

    bb = BouncingBall()

    parameters:
      m - scalar - mass of ball
      g - scalar - gravitational constant
      c - scalar - coefficient of restitution
    """
    Hybrid.__init__(self)

  def F(self, (k,t), (j,x), **p):
    if j == 1:
      h,dh = x
      dx = np.array([dh, -p['g']])

    return dx  

  def G(self, (k,t), (j,x), **p):
    if j == 1:
      g = x[0]

    else:
      raise RuntimeError,'G -- unknown discrete mode'

    return g

  def R(self, (k,t), (j,x), **p):
    if j == 1:
      h,dh = x

      k_ = k+1
      t_ = t
      j_ = j
      x_ = np.array([0, -p['c']*dh])

    return (k_,t_),(j_,x_)

  def O(self, (k,t), (j,x), **p):
    if j == 1:
      # height, velocity
      h,dh = x
      # kinetic energy, potential energy
      KE = 0.5 * p['m'] * dh**2
      PE = p['m'] * p['g'] * h
      #
      o = [h,dh,KE,PE]

    return o

if __name__ == "__main__":

  N = 10
  T = 10.
  j = 1
  x = np.array([1., 0.])
  m = 1.
  g = 10.
  c = 1.
  c = 0.66
  #c = 0
  debug = True
  debug = False
  Zeno = True
  Zeno = False

  p = dict(m=m, g=g, c=c, debug=debug, Zeno=Zeno)

  1/0

  bb = BouncingBall()

  h  = 1e-2
  rx = 1e-6

  trjs = bb.sim((N,T), (j,x), h, rx, **p);

  (k,t),(j,o) = bb.obs(trjs, insert_nans=True, **p)
  (kd,td),(jd,od) = bb.obs(trjs, only_disc=True, **p)

  h,dh,KE,PE = o.T
  hd,dhd,KEd,PEd = od.T

  lw = 2
  mew = lw
  ms = 10
  lt = '-'
  fs = (8,8)

  plt.figure(1,figsize=fs)

  # continuous transitions
  ax = plt.subplot(1,1,1); plt.grid('on')
  ax.plot(dh,h,'k.-',lw=lw,ms=ms)
  # discrete transitions
  ax.plot(dhd[::2], hd[::2], 'o',ms=ms,mew=mew,mec='g',mfc='None')
  ax.plot(dhd[1::2],hd[1::2],'o',ms=ms,mew=mew,mec='r',mfc='None')
  # dashed line at discrete transitions
  ax.plot(np.vstack((dhd[1:-1:2],dhd[2::2])), 
           np.vstack(( hd[1:-1:2], hd[2::2])), 'k--',lw=lw)
  ax.set_xlabel(r'$\dot{h}$')
  ax.set_ylabel(r'$h$')

  plt.figure(2,figsize=fs)

  ax = plt.subplot(3,1,1); ax.grid('on')
  # continuous transitions
  ax.plot(t,h,'k'+lt,lw=lw,ms=ms)
  # discrete transitions -- height is continuous, so these don't appear
  #ax.plot(np.vstack(( td[1:-1:2], td[2::2])), 
  #         np.vstack(( hd[1:-1:2], hd[2::2])), 'k--',lw=lw)
  ax.set_ylabel(r'$h$')
  ax.set_xticklabels([])

  ax = plt.subplot(3,1,2); ax.grid('on')
  # continuous transitions
  ax.plot(t,dh,'k'+lt,lw=lw,ms=ms)
  # dashed line at discrete transitions
  ax.plot(np.vstack(( td[1:-1:2], td[2::2])), 
           np.vstack((dhd[1:-1:2],dhd[2::2])), 'k--',lw=lw)
  ax.set_ylabel(r'$\dot{h}$')
  ax.set_xticklabels([])

  ax = plt.subplot(3,1,3); ax.grid('on')
  # continuous transitions
  ax.plot(t,KE,'b'+lt,lw=lw,ms=ms)
  ax.plot(t,PE,'g'+lt,lw=lw,ms=ms)
  # dashed line at discrete transitions -- KE is discontinuous when c != 1
  ax.plot(np.vstack(( td[1:-1:2], td[2::2])), 
           np.vstack((KEd[1:-1:2],KEd[2::2])), 'b--',lw=lw)
  ax.legend(('KE','PE'))
  ax.set_ylabel(r'energy')
  ax.set_xlabel('time (sec)')

