import numpy as np 
from scipy.interpolate import CubicSpline 




def partition_envelope(f: Callable, threshold: float, interval: Tuple = (0, 1), lower: bool = False):
  """
  Partitions the domain of a real-valued function 'f' into intervals by evaluating the pointwise maximum of 'f' and the constant function g(x) = threshold. 
  The name 'envelope' is because this can be seen as intersecting the upper-envelope of 'f' with 'g'

  Parameters: 
    f := Callable that supports f.operator(), f.derivative(), and f.roots() (such as the Splines from scipy)
    threshold := cutoff-threshold
    interval := interval to evalulate f over 
    lower := whether to evaluate the lower envelope instead of the upper 

  Return: 
    intervals := (m x 3) nd.array giving the intervals that partition the corresponding envelope
  
  Each row (b,e,s) of intervals has the form: 
    b := beginning of interval
    e := ending of interval
    s := 1 if in the envelope, 0 otherwise 

  """
  assert isinstance(interval, Tuple) and len(interval) == 2

  ## Partition a curve into intervals at some threshold
  in_interval = lambda x: x >= interval[0] and x <= interval[1]
  crossings = np.fromiter(filter(in_interval, f.solve(threshold)), float)

  ## Determine the partitioning of the upper-envelope (1 indicates above threshold)
  intervals = []
  if len(crossings) == 0:
    is_above = f(0.50).item() >= threshold
    intervals.append((0.0, 1.0, 1 if is_above else 0))
  else:
    if crossings[-1] != 1.0: 
      crossings = np.append(crossings, 1.0)
    b = 0.0
    df = f.derivative(1)
    df2 = f.derivative(2)
    for c in crossings:
      grad_sign = np.sign(df(c))
      if grad_sign == -1:
        intervals.append((b, c, 1))
      elif grad_sign == 1:
        intervals.append((b, c, 0))
      else: 
        accel_sign = np.sign(df2(c).item())
        if accel_sign > 0: # concave 
          intervals.append((b, c, 1))
        elif accel_sign < 0: 
          intervals.append((b, c, 0))
        else: 
          raise ValueError("Unable to detect")
      b = c
  
  ### Finish up and return 
  intervals = np.array(intervals)
  if lower: 
    intervals[:,2] = 1 - intervals[:,2]
  return(intervals)
