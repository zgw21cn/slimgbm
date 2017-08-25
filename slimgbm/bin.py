__author__ = 'ZhangGuowei'


class Bin(object):
    def __init__(self,lower_bound=0,upper_bound=0,grad_sum=0,hess_sum=0):
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.grad_sum=grad_sum
        self.hess_sum=hess_sum


    def __str__(self):
        return str(self.lower_bound) + " " + str(self.upper_bound) + " "+\
               str(self.grad_sum)+" "+str(self.hess_sum)
