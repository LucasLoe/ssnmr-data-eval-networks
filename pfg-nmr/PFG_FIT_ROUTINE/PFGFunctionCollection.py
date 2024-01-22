import numpy as np

class PFGFunctionCollection:

    def __init__(self) -> None:
        pass

        
    def _dexp(self, b, amp, D):
        return abs(amp)*np.exp(-b*D)


    def _y0exp2(self, b,amp1, amp2, D1, D2, y0):
        return abs(y0) + self._dexp(b, amp1, D1) + self._dexp(b, amp2, D2)


    def p0exp2(self, b,parDict):
        amp1, amp2, offset = parDict['a1'] , parDict['a2'], parDict['offset']
        diff1, diff2 = parDict['D1'] , parDict['D2']
        return abs(offset) + abs(amp1)*np.exp(-b*diff1) + abs(amp2)*np.exp(-b*diff2)
