# Paralle Tempering Random Walk MCMC for Weighted Mixture of Distributions for  Curve Fitting.
# Ratneel Deo and Rohitash Chandra (2017)).
# SCIMS, USP.  deo.ratneel@gmail.com
# CTDS, UniSYD. c.rohitash@gmail.com
# Simulated data is used.


# Ref: https://en.wikipedia.org/wiki/Dirichlet_process
# https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.random.dirichlet.html




from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import matplotlib.mlab as mlab
import threading
import time


def fx_func(nModels, x, mu, sig, w):

    #print (w)
    fx = np.zeros(x.size)
    for i in range(nModels):
        fx = fx + w[i] * mlab.normpdf(x, mu[i], np.sqrt(sig[i]))
    return fx

class BayesMCMC:  # MCMC class

    def __init__(self, samples, nModels, ydata, tempr,  x):
        
        self.temprature = tempr
        print(tempr)
        self.fx_samples = np.ones((samples, ydata.size))
        #print(len(self.fx_samples))
        # memory for posterior mu, sig, w and tau - also fx
        self.pos_mu = np.ones((samples, nModels))
        self.pos_sig = np.ones((samples, (nModels)))
        self.pos_w = np.ones((samples, (nModels)))
        self.pos_tau = np.ones((samples,  1))

        #self.fx_samples = np.ones((samples, ydata.size))

        self.mu_current = np.zeros(nModels)
        

        self.sig_current = np.zeros(nModels)  # to get sigma
        self.nu_current = np.zeros(nModels)  # to get sigma
        
        self.w_current = np.zeros(nModels)
        

        self.step_size_mu = 0.1  # need to choose these values according to the problem
        self.step_size_nu = 0.2
        self.step_size_eta = 0.1

        for i in range(nModels):
            self.sig_current[i] = np.var(x)
            self.mu_current[i] = np.mean(x)
            self.nu_current[i] = np.log(self.sig_current[i])
            self.w_current[i] = 1.0 / nModels

        fx = fx_func(nModels, x, self.mu_current, self.sig_current, self.w_current)

        t = np.var(fx - ydata)
        self.tau_current = t
        self.eta_current = np.log(t)
        
        self.lhood = 0
        self.naccept = 0
        
        self.likelihood_current, self.fx = self.likelihood_func_pt(nModels, x, ydata, self.mu_current,
                                                 self.sig_current, self.w_current,
                                                 self.tau_current)
        

    def likelihood_func(self, nModels, x, y, mu, sig, w, tau):
        tausq = tau
        fx = fx_func(nModels, x, mu, sig, w)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return [(np.sum(loss)), fx]

    def likelihood_func_pt(self, nModels, x, y, mu, sig, w, tau):
        tausq = tau
        fx = fx_func(nModels, x, mu, sig, w)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        #print(np.sum(loss))
        likelihood = (np.sum(loss))* (1.0/self.temprature)
        return [likelihood, fx]
        
    def propose_parameters(self, nModels):
        weights = []
        if nModels == 1:
                weights = [1]
        elif nModels == 2:
            # (genreate vector that  adds to 1)
            weights = np.random.dirichlet((1, 1), 1)
        elif nModels == 3:
            weights = np.random.dirichlet((1, 1, 1), 1)  # (vector adds to 1)
        elif nModels == 4:
            weights = np.random.dirichlet(
                (1, 1, 1, 1), 1)  # (vector adds to 1)
        elif nModels == 5:
            weights = np.random.dirichlet(
                (1, 1, 1, 1, 1), 1)  # (vector adds to 1)
                
        return weights       
                
    

    def sample (self, samples, nModels, x, ydata, start, end):
        mu_proposal = np.zeros(nModels)
        sig_proposal = np.zeros(nModels)  # sigma
        nu_proposal = np.zeros(nModels)
        w_proposal = np.zeros(nModels)
        eta_proposal = 0.1
        tau_proposal = 0   # vector notation is used although size is 1
        

        for i in range(start, end, 1):
            #print (i, 'i val \n')
   
            ##create a new set of parapamers (theta)
            weights = self.propose_parameters(nModels)

            nu_proposal = self.nu_current + np.random.normal(0, self.step_size_nu, nModels)
            sig_proposal = np.exp(nu_proposal)
            mu_proposal = self.mu_current + np.random.normal(0, self.step_size_mu, nModels)


            ##propose a new set of weights (randomly generate the values of theta)
            for j in range(nModels):
                # ensure they stay between a range
                if mu_proposal[j] < 0 or mu_proposal[j] > 1:
                    mu_proposal[j] = random.uniform(np.min(x), np.max(x))

                w_proposal[j] = weights[0, j]  # just for vector consistency

            eta_proposal = self.eta_current + np.random.normal(0, self.step_size_eta, 1)
            tau_proposal = math.exp(eta_proposal)

            likelihood_proposal, fx = self.likelihood_func_pt(nModels, x, ydata,
                                                      mu_proposal, sig_proposal,
                                                      w_proposal, tau_proposal)

            diff = likelihood_proposal - self.likelihood_current

            mh_prob = min(1, math.exp(diff))

            u = random.uniform(0, 1)

            

            if u < mh_prob:
                # Update position
                #print(i, ' is accepted sample')
                self.naccept += 1
                self.likelihood_current = likelihood_proposal
                self.mu_current = mu_proposal
                self.nu_current = nu_proposal
                self.eta_current = eta_proposal

                #print(self.likelihood_current, self.mu_current, self.nu_current,  self.eta_current, 'accepted')
                #print(self.mu_proposal)
                
                self.pos_mu[i + 1, ] = mu_proposal
                self.pos_sig[i + 1, ] = sig_proposal
                self.pos_w[i + 1, ] = w_proposal
                self.pos_tau[i + 1, ] = tau_proposal
                self.fx_samples[i + 1, ] = fx
                #print (self.pos_mu)

            else:
                #print('here', self.pos_mu[i+1,])
                
                self.pos_mu[i + 1, ] = self.pos_mu[i, ]
                self.pos_sig[i + 1, ] = self.pos_sig[i, ]
                self.pos_w[i + 1, ] = self.pos_w[i, ]
                self.pos_tau[i + 1, ] = self.pos_tau[i, ]
                self.fx_samples[i + 1, ] = self.fx_samples[i, ]
                
                #print('here', self.pos_mu[i+1,])



        self.lhood = self.likelihood_current
        print(self.naccept / samples, '% was accepted')        

        return (self.pos_mu, self.pos_sig, self.pos_w, self.pos_tau, self.fx_samples, self.lhood)


 
        

class ParallelTempering:

    def __init__(self, num_chains, maxtemp,NumSample,ydata,nModels):
        
        self.maxtemp = maxtemp
        self.num_chains = num_chains
        self.chains = []
        self.tempratures = []
        self.NumSamples = int(NumSample/self.num_chains)
        self.sub_sample_size = int( 0.1* self.NumSamples)
        #self.sub_sample_size =  100
        
              
        self.fx_samples = np.ones((num_chains,self.NumSamples, ydata.size))
        self.pos_mu = np.ones((num_chains,self.NumSamples, nModels))
        self.pos_sig = np.ones((num_chains,self.NumSamples, (nModels)))
        self.pos_w = np.ones((num_chains,self.NumSamples, (nModels)))
        self.pos_tau = np.ones((num_chains,self.NumSamples,  1))
          
    
    # assigin tempratures dynamically   
    def assign_temptarures(self):
        tmpr_rate = (self.maxtemp /self.num_chains)
        temp = 1
        for i in xrange(0, self.num_chains):            
            self.tempratures.append(temp)
            temp += temp #tmpr_rate
            print(self.tempratures[i])
            
    
    # Create the chains.. Each chain gets its own temprature
    def initialize_chains (self, nModels, ydata,x):
        self.assign_temptarures()
        for i in xrange(0, self.num_chains):
            self.chains.append(BayesMCMC(self.NumSamples, nModels,ydata, self.tempratures[i],x))
            
    # Propose swapping between adajacent chains        
    def propose_swap (self, swap_proposal):
         for l in range( self.num_chains-1, 0, -1):            
                u = 1
		swap_prob = swap_proposal[l-1]
                if u < swap_prob : 
                    self.swap_info(self.chains[l],self.chains[l-1])
                    print('chains swapped')     
            
            
    # Swap configuration of two chains    
    def swap_info(self, chain_cooler, chain_warmer):  
        
        temp_chain = chain_cooler;
        
        chain_cooler.fx_samples = chain_warmer.fx_samples
        chain_cooler.pos_mu = chain_warmer.pos_mu
        chain_cooler.pos_sig = chain_warmer.pos_sig
        chain_cooler.pos_w = chain_warmer.pos_w
        chain_cooler.pos_tau = chain_warmer.pos_tau
        
        chain_warmer.fx_samples = temp_chain.fx_samples
        chain_warmer.pos_mu = temp_chain.pos_mu
        chain_warmer.pos_sig = temp_chain.pos_sig
        chain_warmer.pos_w = temp_chain.pos_w
        chain_warmer.pos_tau = temp_chain.pos_tau
        
    # Merge different MCMC chains y stacking them on top of each other       
    def merge_chain (self, chain):
        comb_chain = []
        for i in xrange(0, self.num_chains):
            for j in xrange(0, self.NumSamples):
                comb_chain.append(chain[i][j].tolist())		
        return np.asarray(comb_chain)
		

    def run_chains (self, nModels, x, ydata):
        self.initialize_chains ( nModels, ydata,x)
        swap_proposal = np.ones(self.num_chains-1) # only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
        
        print (self.NumSamples,self.sub_sample_size,self.NumSamples/self.sub_sample_size)
        #input ()
        start = 0
        end =  start + self.sub_sample_size
        #for i in range(0, int(self.NumSamples/self.sub_sample_size)):
        while (end < self.NumSamples):
            
            
            print (start, end)
            print ('--------------------------------------\n\n')

            lhood = np.zeros(self.num_chains)
            
            #run each chain for a fixed number of SAMPLING Period along the MCMC Chain
            for j in range(0,self.num_chains):        
                self.pos_mu[j], self.pos_sig[j], self.pos_w[j], self.pos_tau[j], self.fx_samples[j], lhood[j] = self.chains[j].sample(self.NumSamples, nModels, x, ydata, start, end)
                print (j, lhood[j])
                
            
            #calculate the swap acceptance rate for parallel chains    
            for k in range(0, self.num_chains-1): 
                 swap_proposal[k]=  (lhood[k]/lhood[k+1])*(1/self.tempratures[k] * 1/self.tempratures[k+1])  
                
            
            #propose swapping
            self.propose_swap(swap_proposal)
            
            #update the starting and ending positon within one chain
            start =  end
            end =  start + self.sub_sample_size
        
        
        
        
        
        end =  self.NumSamples-1   
        for j in range(0,self.num_chains):        
            self.pos_mu[j], self.pos_sig[j], self.pos_w[j], self.pos_tau[j], self.fx_samples[j], lhood[j] = self.chains[j].sample(self.NumSamples, nModels, x, ydata, start, end)
            print (j, lhood[j])    
  
        #concatenate all chains into one complete chain by stacking them on each other 
        chain_mu = self.merge_chain(self.pos_mu)
        chain_sig = self.merge_chain(self.pos_sig)
        chain_w = self.merge_chain(self.pos_w)
        chain_tau = self.merge_chain(self.pos_tau)
        chain_fx = self.merge_chain(self.fx_samples)
        print(self.pos_mu)   
             
            
        return chain_mu,chain_sig,chain_w,chain_tau,chain_fx
      


#plot a figure of posterior distributions
def plot_figure(list_points,title,ylabel,xlabel):
    len(list_points)
    print (list_points)
    bins = np.linspace(0, 1, 100)
    plt.clf()
    plt.hist(list_points, bins)
    plt.savefig( title + '.png')
    plt.show()
    
    




def main():
    random.seed(time.time())
    nModels = 2

    # load univariate data in same format as given
    modeldata = np.loadtxt('simdata.txt')
    ydata = modeldata
  
    x = np.linspace(1 / ydata.size, 1, num=ydata.size)  # (input x for ydata)

    NumSamples = 50000  # need to pick yourself
    
    #Number of chains of MCMC required to be run
    num_chains = 6

    #Maximum tempreature of hottest chain  
    maxtemp = 100
    
    #Create A a Patratellel Tempring object instance 
    pt = ParallelTempering(num_chains, maxtemp, NumSamples,ydata,nModels)

    #run the chains in a sequence in ascending order
    pos_mu, pos_sig, pos_w, pos_tau, fx_samples = pt.run_chains( nModels, x, ydata)
    
    print('sucessfully sampled')

    #remove the initial burnin period of MCMC
    burnin = 0.05 * NumSamples   # use post burn in samples
    pos_mu = pos_mu[int(burnin):]
    pos_sig = pos_sig[int(burnin):]
    pos_w = pos_w[int(burnin):]
    pos_tau = pos_tau[int(burnin):]
    
    #plot posterior distribution of MU 
    plot_figure(pos_mu[:,0],'Posterior_MU_Distrubution','y','x')
    

    
    #plot the accepted samples over the target points
    fx_mu = fx_samples.mean(axis=0)
    fx_high = np.percentile(fx_samples, 95, axis=0)
    fx_low = np.percentile(fx_samples, 5, axis=0)

    plt.plot(x, ydata)
    plt.plot(x, fx_mu)
    plt.plot(x, fx_low)
    plt.plot(x, fx_high)
    plt.fill_between(x, fx_low, fx_high, facecolor='g', alpha=0.4)

    plt.title("Plot of Data vs MCMC Uncertainty")
    plt.savefig('results.png')
    plt.clf()


if __name__ == "__main__":
    main()
