import copy
import numpy as np
import scipy as sp
import scipy.optimize as op

"""
utility library

functions:
  fp - find fixed point of a func
  jac - numerically approximate Jacobian of a func 

Sam Burden, UW Seattle 2017
"""

def fp(f,x0,eps=1e-6,modes=[1,2],Ni=4,N=10,
       dist=lambda x,y : np.max(np.abs(x-y)) ):
  """
  .fp  Find fixed point of f near x0 to within tolerance eps

  The algorithm has two modes:
    1. iterate f; keep result if error decreases initially
    2. run fsolve on f(x)-x; keep result if non-nan

  Inputs:
    f : R^n --> R^n
    x0 - n vector - initial guess for fixed point  f(x0) ~= x0

  Outputs:
    x - n vector - fixed point  f(x) ~= x
  """
  # compute initial error
  xx = f(x0); e = dist(xx,x0)
  suc = False
  # 1. iterate f; keep result if error decreases initially
  if 1 in modes:
    # Iterate orbit map several times, compute error
    x = reduce(lambda x, y : f(x), [x0] + range(Ni))
    xx = f(x); e = dist(xx,x); e0 = dist(x,x0)
    # If converging to fixed point
    if e < e0:
      suc = True
      # Iterate orbit map
      n = 0
      while n < N-Ni and e > eps:
        n = n+1; x = xx; xx = f(x)
        e = dist(xx,x)
      x0 = xx
  # 2. run fsolve on f(x)-x; keep result if non-nan
  if 2 in modes:
    x = x0
    # Try to find fixed point using op.fsolve
    xx = op.fsolve(lambda x : f(x)-x, x)
    # If op.fsolve succeeded
    if not np.isnan(xx).any() or self.dist(xx,x) > e:
      suc = True
      x0 = xx
  # if all methods failed, return nan
  if not suc:
    x0 = np.nan*x0
  
  return x0

def central(f,x,fx,d):
  """
  df = central()  compute central difference 

  df = 0.5*(f(x+d) - f(x-d))/np.linalg.norm(d)
  """
  return 0.5*(f(x+d) - f(x-d))/np.linalg.norm(d)

def forward(f,x,fx,d):
  """
  df = forward()  compute forward difference 

  df = (f(x+d) - fx)/np.linalg.norm(d)
  """
  return (f(x+d) - fx)/np.linalg.norm(d)

def D(f, x, fx=None, d=1e-6, D=None, diff=forward):
  """
  Numerically approximate derivative of f at x

  Inputs:
    f : R^n --> R^m
    x - n vector
    d - scalar or (1 x n) - displacement in each coordinate
  (optional)
    fx - m vector - f(x)
    D - k x n - directions to differentiate (assumes D.T D invertible)
    diff - func - numerical differencing method

  Outputs:
    Df - m x n - Jacobian of f at x
  """
  if fx is None:
    fx = f(x)
  if D is None:
    J = map(lambda dd : diff(f,x,fx,dd), list(d*np.identity(len(x))))
  else:
    J = map(lambda dd : diff(f,x,fx,dd), list(D))
    dbg()

  return np.array(J).T

def dot(*arrays):
  return reduce(np.dot, arrays)

if __name__ == "__main__":
    import doctest
    doctest.testmod()


