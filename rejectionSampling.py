#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:08:40 2020

@author: julia milton jmilton@mit.edu
"""
#
# MIT License
#
# Copyright (c) 2020 Julia Milton
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#imports
# RUN THIS COMMAND FIRST! 
#%run ./my_script.py
#%matplotlib qt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.stats as st
import seaborn as sns
import pandas as pd



class AnimatedScatter(object):
    """Draws the scatter plot as an animation using FuncAnimation"""

    def __init__(self, numpoints, targetFunction, proposalFunction, k, domain, ymax):
        self.k = k
        self.numpoints = numpoints+2
        self.targetFunction = targetFunction
        self.proposalFunction = proposalFunction
        self.domain = domain
        self.ymax = ymax
        self.stream = self.data_stream()

        # Setup the figure and axes
        self.fig, self.ax = plt.subplots()
        self.textvar = self.ax.text(0.05, 0.95, "Init", transform=self.ax.transAxes, fontsize=10, verticalalignment='top')
        
        # Setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1, 
                                          init_func=self.setup_plot, blit=False, cache_frame_data=False, save_count=self.numpoints)

    def setup_plot(self):
        """Initial drawing of the scatterplot"""

        x, y, s, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, c=c, s=s, vmin=0, vmax=1,
                                    cmap="Set3", edgecolor="none")
        self.ax.axis([min(self.domain), max(self.domain), 0, self.ymax])

        #Draw lines showing the target and proposal distributions
        line = self.ax.plot(self.domain, self.targetFunction(self.domain), 
                                    'y:', label = "Target Distribution")
        line2 = self.ax.plot(self.domain, k*self.proposalFunction(self.domain), 
                                    'b:', label = "Proposal Distribution")

        #Add legend and title
        self.ax.legend( loc = 'upper right', prop={'size': 6}, borderpad=1)
        self.fig.suptitle('Rejection Sampling \n Using proposal draws from a scaled distribution', fontsize=10)
        
        return self.scat,
    

    def data_stream(self):
        """Generate data corresponding to selections from the target distribution and the proposal"""        

        #initialize
        self.acceptedSamples = []
        self.acceptPx = []
        self.acceptX = []
        self.s = []
        self.ExArray = []
        
        j=1

        #run rejection sampling as many times as the user has defined
        while j < self.numpoints+1:
            
            def rejection_sampling(iter):
                
                #sample from f(x), which is the scaled gaussian N(0,1)
                proposalPoint = np.random.normal(0,1)

                #calculate probability that the proposal point came from the target function
                probProposal = self.targetFunction(proposalPoint)

                #calculate probability ratio
                alpha = probProposal/(self.k*self.proposalFunction(proposalPoint))

                #generate a uniform random nunmber between 0 and 1
                u = np.random.uniform()

                #accept the point if u<alpha
                if u <= alpha:
                    #if the point is accepted, add it to the list to find the expected value
                    self.ExArray.append(proposalPoint)

                    #the rest of this is to plot
                    self.acceptedSamples.append(9) #marking samples as accepted or rejected to change color
                    self.s.append(15)
                    self.acceptPx.append(u*self.k*self.proposalFunction(proposalPoint))
                    self.acceptX.append(proposalPoint)
                    
                else:
                    # the rest of this is to plot
                    self.acceptedSamples.append(0)
                    self.s.append(15)
                    self.acceptPx.append(u*self.k*self.proposalFunction(proposalPoint))
                    self.acceptX.append(proposalPoint)
                
                return [self.acceptedSamples, self.s, self.acceptPx, self.acceptX]
            
            [c, s, acceptPx, acceptX] = rejection_sampling(1)
            
            s = np.array(s)
            c = np.array(c)
            y = np.array(acceptPx)
            x = np.array(acceptX)
 

            #for calculating expected value
            self.Ex = np.array(self.ExArray)
            self.meanPEst = np.mean(self.Ex)         #calculate current expected value
            self.varPEst = np.var(self.Ex)/len(self.Ex)         #calculate current variance measure
            

            j = j+1

            yield np.c_[x[:int(j-1)], y[:int(j-1)], s[:int(j-1)], c[:int(j-1)]]
            
    

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)
        self.textvar.remove()
        # Set x and y data
        self.scat.set_offsets(data[:, :2])
        # Set sizes
        self.scat.set_sizes(data[:,2])
        # Set colors
        self.scat.set_array(data[:, 3])
        # Set text
        textstr = '\n'.join((
            r'Number of draws: %d' % (i),
            r'E(x)= %.4f ' % self.meanPEst,
            r'$\sigma^2/n$= %.4f' % self.varPEst))
        self.textvar = self.ax.text(0.05, 0.95, textstr, transform=self.ax.transAxes, fontsize=10, verticalalignment='top')

        # FuncAnimation expects a sequence of artists in return, leave the comma in 
        return self.scat,


if __name__ == '__main__':
    
    """user should define the following section"""

    # p(x) is the target distribution
    def p(x):
            return 0.2*st.norm.pdf(x, loc=1, scale=0.3)+0.7*st.norm.pdf(x, loc=-0.5, scale=0.4) #Gaussian mixture
            
            # uncomment below and comment out the above line to see an alternative example
            #return np.exp(-x**2/2)*(np.sin(6*x)**2 + 3*np.cos(x)**2.*np.sin(4*x)**2+1)
    
    # f(x) is the proposal distribution        
    def f(x):
        return st.norm.pdf(x, loc=0, scale=1)
    
    #change x and y axes below if necessary
    xmin = -4
    xmax = 4
    ymax = 1
    
    #change number of runs below (recommended minimum 1000)
    MCruns = 3000
    
    """end user defined"""

    
    domain = np.linspace(xmin, xmax,10000, endpoint=True)

    # k is the scaling factor for the proposal distribution
    k = max(p(domain)/f(domain))


    """Uncomment the two lines below and the last line if you would like to save the animation as a movie file """
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps =120, metadata=dict(artist='Me'))
        
    a = AnimatedScatter(MCruns, p, f, k, domain, ymax)
    plt.show()


    """Uncomment the line below if you would like to save the animation as a movie file """
    #a.ani.save('MCMC3.mp4', writer=writer)