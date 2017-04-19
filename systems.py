"""
system classes

classes:
  Discrete - discrete system (i.e. finite-state machine)
  Continuous - continuous system (i.e. differential equation)
  Hybrid - hybrid system (i.e. combination of discrete and continuous systems)

methods:
  sim - numerical simulation 
  obs - aggregate data from trajectories

Sam Burden, UW Seattle 2017
"""

import numpy as np
from numpy import linalg as la
import pylab as plt
import util

dbg = lambda _ : 1/0
import sys
from copy import deepcopy

class Discrete(object):

  def __init__(self):
    """
    ds = Discrete()

    discrete system superclass

    derived classes override
      .R,
    though trivial implementations are provided.
    """
    pass

  def R(self, s, j, **params):
    """
    discrete dynamics (i.e. reset map)
      s_,j_ = R(s,j) 
    """
    s_,j_ = (deepcopy(s)+1,deepcopy(j))
    return s_,j_

  def sim(self, K, j0, **params):
    """
    numerical simulation

    trjs = sim(K, j0, [params])

    inputs:
      K   - int    - maximum discrete simulation time
      j0  - object - initial discrete state

    optional inputs:
      (all named arguments will be passed to methods as params dict)

    outputs:
      trj - trajectory dict
        .k - list - discrete times
        .j - list - discrete states
    """
    k = [0]
    j = deepcopy(j0)

    while ( trj['k'][-1] < K # don't exceed max discrete simulation time
            and not trj['j'] is None ): # don't allow discrete state is None
      # . . .
      pass

    trj = dict(k=k,j=j)

    return trj

class Continuous(object):

  def __init__(self):
    """
    cs = Continuous()

    continuous system superclass

    derived classes override
      .F,
    though trivial implementations are provided.
    """
    pass

  def F(self, t, x, **params):
    """
    continuous dynamics (i.e. vector field)
      dx = F(t,x) 
    """
    return 0.*x

  def sim(self, T, x0, **params):
    """
    numerical simulation

    trjs = sim(T, x0, [params])

    inputs:
      T   - scalar - maximum simulation time
      x0  - object - initial discrete state

    optional inputs:
      (all named arguments will be passed to methods as params dict)

    outputs:
      trj - trajectory dict
        .t - list - continuous times
        .x - list - continuous states
    """
    t = [0]
    x = deepcopy(x0)

    while ( trj['t'][-1] < T # don't exceed max continuous simulation time
            and not np.any(np.isnan(trj['x'][-1]))): # don't allow np.nan in state
      # . . .
      pass

    trj = dict(t=t,x=x)

    return trj

class Hybrid(object):

  def __init__(self):
    """
    hs = Hybrid()

    hybrid system superclass

    derived classes override
      .F, .G, .R, .O,
    though trivial implementations are provided.
    """
    pass

  def F(self, (k,t), (j,x), **params):
    """
    continuous dynamics (i.e. vector field)
      dx = F((k,t),(j,x)) 
    """
    return 0.*x

  def G(self, (k,t), (j,x), **params):
    """
    guard function
      g = G((k,t),(j,x))
      --
      g > 0  : guard inactive
      g <= 0 : guard active
    """
    return 0

  def R(self, (k,t), (j,x), **params):
    """
    discrete dynamics (i.e. reset map)
      t_,(j_,x_) = R(t,(j,x)) 
    """
    (k_,t_),(j_,x_) = (deepcopy(k)+1,deepcopy(t)),(deepcopy(j),deepcopy(x))
    return (k_,t_),(j_,x_)


  def O(self, (k,t), (j,x), **params):
    """
    observation (i.e. outputs)
      o = O((k,t),(j,x))
    """
    return x

  def obs(self, trjs, insert_nans=False, only_disc=False, include=None, **p):
    """
    observations

    (k,t),(j,o) = obs(trjs, [insert_nans, only_disc, include, params])

    inputs:
      trjs - struct array - trajectories
    
    optional inputs:
      insert_nans - bool - insert np.nan at discrete transitions ?
      only_disc - bool - only include discrete transitions ?
      include - list - discrete modes to include

    outputs:
      k - Nt list - observation indices
      t - 1  x Nt - observation times
      j - Nt list - discrete states
      o - Nt x No - observations of continuous states
    """
    K = []; T = []; J = []; O = []

    for trj in trjs:
      if include is None or trj.j in include:
        k = trj['k']; j = trj['j']
        if only_disc: # only include discrete transitions
          # initial trj state
          (t,x) = (trj['t'][0],trj['x'][0])
          K.append( k )
          T.append( t )
          J.append( j )
          # final trj state
          O.append( self.O((k,t),(j,x), **p) )
          (t,x) = (trj['t'][-1],trj['x'][-1])
          K.append( k )
          T.append( t )
          J.append( j )
          O.append( self.O((k,t),(j,x), **p) )
        else:
          for (t,x) in zip(trj['t'],trj['x']):
            K.append( k )
            T.append( t )
            J.append( j )
            O.append( self.O((k,t),(j,x), **p) )
          if insert_nans:
            K.append( K[-1] )
            T.append( T[-1] )
            J.append( J[-1] )
            O.append( list(np.nan*np.asarray(O[-1])) )

    return (K,np.asarray(T)),(J,np.asarray(O))

  def sim(self, (K,T), (j0,x0), dt, rx, **params):
    """
    numerical simulation

    trjs = sim((K,tf), (j0,x0), dt, rx, [params])

    inputs:
      K   - int    - maximum discrete simulation time (np.inf OK)
      T   - scalar - maximum continuous simulation time
      j0  - object - initial discrete state
      x0  - array  - initial continuous state
      dt  - scalar - numerical simulation stepsize
      rx  - scalar - relaxation parameter
    
    optional inputs:
      (all named arguments will be passed to methods as params dict)
      params['debug'] - bool - print debugging info ?
      params['Zeno']  - bool - quit executions with short modes ?

    outputs:
      trjs - list of trajectory dicts
      trj - trajectory dict
        .k - discrete time
        .t - continuous times
        .j - discrete state
        .x - continuous states
    """
    dt0 = dt 

    k = 0
    t = [0.]
    j = deepcopy(j0)
    x = [deepcopy(x0)]

    trj = dict(k=k,t=t,j=j,x=x)
    trjs = []

    while ( trj['t'][-1] <= T # don't exceed max continuous time
            and trj['k'] < K # don't exceed max discrete transitions
            and not trj['j'] is None ): # don't allow discrete state is None
      k0 = trj['k']
      t0 = trj['t'][-1]
      j0 = trj['j']
      x0 = trj['x'][-1]
      if 0: # forward Euler
        dx = self.F((k0,t0), (j0,x0), **params)
      else: # 4th-order Runge-Kutta 
        f = lambda t,x : self.F((k0,t), (j0,x), **params)
        dx1 = f( t0, x0 ) * dt
        dx2 = f( t0+.5*dt, x0+.5*dx1 ) * dt
        dx3 = f( t0+.5*dt, x0+.5*dx2 ) * dt
        dx4 = f( t0+dt, x0+dx3 ) * dt
        dx = (1./6.)*( dx1 + 2*dx2 + 2*dx3 + dx4 ) / dt

      k = k0
      j = j0
      t = t0 + dt
      x = x0 + dt * dx
      g = self.G((k,t), (j,x), **params)

      # halve step size until trajectory doesn't violate guard more than rx
      i = 0
      imax = 50
      while np.any(g < -rx) and (i <= imax):
        dt  = dt/2.
        t  = t0 + dt
        x = x0 + dt * dx
        g = self.G((k,t), (j,x), **params)
        i += 1

      #if (i >= imax):
      #  raise RuntimeError,'(sim)  guard iterations exceeded -- you probably have a buggy guard'

      # append state to trj
      trj['t'].append(t)
      trj['x'].append(x)

      if 0 and 'debug' in params and params['debug']:
        print '  : (k,t)=(%s,%0.3f), (j,x)=(%s,%s), dt=%0.2e, g = %s, x = %s, dx = %s' % (k,t,j,x,dt,g,x,dx)

      # if in guard 
      if np.any(g < 0):

        # spend time in guard
        if i >= imax:
          t = t + rx
        else:
          t = t + (rx + g.min())
        trj['t'].append(t)
        trj['x'].append(x)

        if 'debug' in params and params['debug']:
          print 'rx: (k,t)=(%s,%0.3f), (j,x)=(%s,%s), dt=%0.2e, g = %s, x = %s' % (k,t,j,x,dt,g,x)

        # append trj to trjs
        trjs.append(trj)

        if 'Zeno' in params and params['Zeno'] and (len(trj['t']) <= 4):

          print '(sim)  possible Zeno @ stepsize dt = %0.6f' % dt0
          print 'rx: (k,t)=(%s,%0.3f), (j,x)=(%s,%s), dt=%0.2e, g = %s, x = %s' % (k,t,j,x,dt,g,x)
          return trjs

        # apply reset to modify trj
        (k,t),(j,x) = self.R((k,t), (j,x), **params)
        trj = dict(k=k,t=[t],j=j,x=[x])

        # re-initialize step size
        dt = dt0

        if 'debug' in params and params['debug']:
          g = self.G((k,t), (j,x), **params)
          print 'rx: (k,t)=(%s,%0.3f), (j,x)=(%s,%s), dt=%0.2e, g = %s, x = %s' % (k,t,j,x,dt,g,x)

    trjs.append(trj)

    return trjs

class Mechanical(Hybrid):

  def __init__(self):
    """
    mech = Mechanical()

    mechanical system subject to unilateral constraints from:
      A. M. Pace and S. A. Burden 
      Piecewise--differentiable trajectory outcomes in mechanical systems 
      subject to unilateral constraints 
      Hybrid Systems: Computation and Control (HSCC), 2017
      
    cf. self-manipulation hybrid system from:
      A. M. Johnson, S. A. Burden, and D. E. Koditschek
      A hybrid systems model for simple manipulation 
        and self-manipulation systems
      The International Journal of Robotics Research (IJRR)
      volume 35, issue 11, pages 1354--1392, 2016
      http://dx.doi.org/10.1177/0278364916639380

    instances and derived classes override
      .M, .c, .f, .a, .Da, .DtDa,
    though trivial implementations are provided (except for a).
    """
    pass

  def M(self, (k,t), (J,q,dq), **params):
    """
    inertia tensor M(q) in:
      M(q) ddq = f_J(q,dq) + c(q,dq) * dq + lambda_J(q,dq) * Da_J(q)
    """
    d = len(q)
    M = np.identity(d)
    return M

  def Lambda(self, (k,t), (J,q,dq), **params):
    """
    inverse of constraint-space inertia tensor:
      Lambda_J = ( Da_J(q) M(q)^{-1} Da_J(q)^T )^{-1}
    """
    d = len(q)
    M = self.M( (k,t), (J,q,dq), **params )
    Da = self.Da( (k,t), (J,q), **params )
    Db = self.Db( (k,t), (J,q), **params )
    D = np.vstack((Da,Db))
    Lambda = -la.inv( util.dot( D, la.inv(M), D.T ) ) 
    return Lambda

  def c(self, (k,t), (J,q,dq), **params):
    """
    Coriolis forces c_J(q,dq) in:
      M(q) ddq = f_J(q,dq) + c(q,dq) * dq + lambda_J(q,dq) * Da_J(q)
    """
    d = len(q)
    c = np.zeros((d,d))
    return c

  def lambda_(self, (k,t), (J,q,dq), **params):
    """
    constraint forces lambda_J(q,dq) in:
      M(q) ddq = f_J(q,dq) + c(q,dq) * dq + lambda_J(q,dq) * Da_J(q)
    """
    d = len(q)
    M = self.M( (k,t), (J,q,dq), **params )
    f = self.f( (k,t), (J,q,dq), **params )
    c = self.c( (k,t), (J,q,dq), **params )
    Da = self.Da( (k,t), (J,q), **params )
    Db = self.Db( (k,t), (J,q), **params )
    DtDa = self.DtDa( (k,t), (J,q,dq), **params )
    DtDb = self.DtDb( (k,t), (J,q,dq), **params )
    D = np.vstack((Da,Db))
    DtD = np.vstack((DtDa,DtDb))
    Lambda = self.Lambda( (k,t), (J,q,dq), **params )
    lambda_ = util.dot( Lambda, util.dot( D, la.inv(M), f + np.dot(c,dq)) 
                                + util.dot( DtD, dq ) )
    return lambda_

  def f(self, (k,t), (J,q,dq), **params):
    """
    effort f_J(q,dq), i.e. applied and potential forces 
    (excluding Coriolis and constraint forces) in contact mode J:
      M(q) ddq = f_J(q,dq) + c(q,dq) * dq + lambda_J(q,dq) * Da_J(q)
    """
    f = 0.*q
    return f

  def F(self, (k,t), (j,x), **params):
    """
    continuous dynamics (i.e. vector field)
      dx = F((k,t),(J,x)) 
    """
    d = len(x)/2
    q,dq = x[:d],x[d:]
    J = j
    M = self.M( (k,t), (J,q,dq), **params )
    f = self.f( (k,t), (J,q,dq), **params )
    c = self.c( (k,t), (J,q,dq), **params )
    Da = self.Da( (k,t), (J,q), **params )
    Db = self.Db( (k,t), (J,q), **params )
    D = np.vstack((Da,Db))
    lambda_ = self.lambda_( (k,t), (J,q,dq), **params )
    ddq = util.dot( la.inv(M), f + util.dot(c,dq) + util.dot(lambda_, D) )
    dx = np.hstack((dq,ddq))
    return dx

  def a(self, (k,t), (J,q), **params):
    """
    unilateral constraints
      a_J(q) >= 0
    """
    a = np.asarray([])
    return a

  def Da(self, (k,t), (J,q), **params):
    """
    Derivative of unilateral constraints
    """
    Da = util.D(lambda q : self.a( (k,t), (J,q), **params ), q)
    return Da

  def DtDa(self, (k,t), (J,q,dq), **params):
    """
    Derivative with respect to time of derivative of unilateral constraints
    """
    DtDa = util.D(lambda dt : self.Da( (k,t+dt[0]), (J,q+dt[0]*dq), **params ), [0])[...,0].T
    return DtDa

  def b(self, (k,t), (J,q), **params):
    """
    bilateral constraints
      b(q) = 0
    """
    b = np.asarray([])
    return b

  def Db(self, (k,t), (J,q), **params):
    """
    Derivative of bilateral constraints
    """
    Db = util.D(lambda q : self.b( (k,t), (J,q), **params ), q)
    return Db

  def DtDb(self, (k,t), (J,q,dq), **params):
    """
    Derivative with respect to time of derivative of bilateral constraints
    """
    DtDb = util.D(lambda dt : self.Db( (k,t+dt[0]), (J,q+dt[0]*dq), **params ), [0])[...,0].T
    return DtDb

  def G(self, (k,t), (j,x), **params):
    """
    guard function
      g = G((k,t),(j,x))
      --
      g > 0  : guard inactive
      g <= 0 : guard active
    """
    d = len(x)/2
    q,dq = x[:d],x[d:]
    J  = (j == True)
    _J = np.logical_not(J)
    # number of constraints
    n = len(J)    
    # number of active constraints
    m = np.sum(J) # = n - len(a)
    a = self.a( (k,t), (_J,q), **params)
    lambda_ = self.lambda_( (k,t), (J,q,dq), **params)
    # unilateral constraint forces
    lambda_ = lambda_[:m] 
    g = np.nan*np.zeros(n)
    g[_J] = a
    g[J] = lambda_
    return g


  def gamma(self, (k,t), (J,q,dq), **params):
    """
    coefficient of restitution gamma_J(d,dq) in:
      Delta_J(q) = I - (1 + gamma_J(q,dq)) M(q)^{-1} Da_J(q)^T Lambda_J(q) Da_J(q)
    """
    return 0.

  def Delta(self, (k,t), (J,q,dq), **params):
    """
    orthogonal projection onto tangent space of constraint manifold A_J:
      Delta_J(q) = I - (1 + gamma_J(q,dq)) M(q)^{-1} Da_J(q)^T Lambda_J(q) Da_J(q)
    """
    d = len(q)
    I = np.identity(d)
    gamma = self.gamma( (k,t), (J,q,dq), **params )
    M = self.M( (k,t), (J,q,dq), **params )
    Da = self.Da( (k,t), (J,q), **params )
    Db = self.Db( (k,t), (J,q), **params )
    D = np.vstack((Da,Db))
    Lambda = self.Lambda( (k,t), (J,q,dq), **params )
    Delta = I + (1 + gamma) * util.dot( la.inv(M), D.T, Lambda, D )
    #1/0
    return Delta

  def R(self, (k,t), (j,x), **params):
    """
    discrete dynamics (i.e. reset map)
      t_,(j_,x_) = R(t,(j,x)) 
    """
    d = len(x)/2
    q,dq = x[:d],x[d:]
    g = self.G((k,t), (j,x), **params)
    I = (g < 0.).nonzero()[0] # constraint indices that are activating or deactivating
    J = deepcopy(j)
    J[I] = np.logical_not(J[I])
    Delta = self.Delta( (k,t), (J,q,dq), **params )
    dq_ = util.dot( Delta, dq )
    k_ = k+1
    t_ = t
    j_ = 1 * J
    x_ = np.hstack([q,dq_])
    return (k_,t_),(j_,x_)

  def O(self, (k,t), (j,x), **params):
    """
    observation (i.e. outputs)
      o = O((k,t),(j,x))
    """
    return x

