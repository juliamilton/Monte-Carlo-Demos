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
    """An animated scatter plot with proposals and targets using matplotlib.animations.FuncAnimation."""

    def __init__(self, numpoints, targetFunction, proposalFunction, k, domain):
        self.k = k
        self.numpoints = numpoints+2
        self.targetFunction = targetFunction
        self.proposalFunction = proposalFunction
        self.domain = domain
        self.stream = self.data_stream()

        # Setup the figure and axes
        self.fig, self.ax = plt.subplots()
        self.textvar = self.ax.text(0.05, 0.95, "Init", transform=self.ax.transAxes, fontsize=10, verticalalignment='top')
        # Setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1, 
                                          init_func=self.setup_plot, blit=False, cache_frame_data=False, save_count=self.numpoints)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""

        x, y, s, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, c=c, s=s, vmin=0, vmax=1,
                                    cmap="Set3", edgecolor="none")
        self.ax.axis([min(self.domain), max(self.domain), -2, 30])

        #Draw lines showing the target and proposal distributions
        line = self.ax.plot(self.domain, 40*self.targetFunction(self.domain), 
                                    'y:', label = "Target Distribution")
        #initialize arrow and hist
        self.arrow = self.ax.arrow(0,0, 0.1,0.1, head_width = 0.01,  
          width = 0.006)

        self.countH, self.binsH, self.barsH = self.ax.hist(self.Ex, bins=20, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
        #Add legend and title
        self.ax.legend( loc = 'upper right', prop={'size': 6}, borderpad=1)
        #self.arrow = self.ax.arrow(data[i+1,0],data[i+1,1], data[i,0]-data[i-1,0],data[i,1]-data[i-1,1], head_width = 0.02,  
          #width = 0.005)
        self.fig.suptitle('Markov Chain Monte Carlo', fontsize=10)
        
        # FuncAnimation expects a sequence of artists in return, leave the comma t 
        return self.scat,
    

    def data_stream(self):
        """Generate data corresponding to selections from the target distribution and the proposal"""        
        
        def logpost(x):
                #return np.log(0.2*st.norm.pdf(x, loc=1, scale=0.3)+0.7*st.norm.pdf(x, loc=-0.5, scale=0.4))
                return 0.2*st.norm.pdf(x, loc=1, scale=0.3)+0.7*st.norm.pdf(x, loc=-0.5, scale=0.4)
                #return np.exp(-x**2/2)*(np.sin(6*x)**2 + 3*np.cos(x)**2.*np.sin(4*x)**2+1)

        self.state = np.zeros(self.numpoints)
        self.acceptprob = np.zeros(self.numpoints)
        self.apost = logpost(self.state[0])

        self.numaccept = 0
        self.ExArray = []

        self.acceptedSamples = []
        self.acceptPx = []
        self.acceptX = []
        self.s = []
        
        j=1
        while j < self.numpoints-1:
        #while True:
        
            def rejection_sampling(iter):

                #acceptedSamples = []
                #acceptPx = []
                #acceptX = []
                #s = []
                
                #for i in range(j):
                    
                acandidate = np.random.normal(self.state[j], 2.2)
                acandpost = logpost(acandidate)
                #print(acandpost)

                #self.acceptprob = min(1, np.exp(acandpost - self.apost))
                self.acceptprob = min(1, (acandpost/self.apost))

                u = np.random.uniform()

                if u <= self.acceptprob:
                    self.state[j+1] = acandidate

                    self.apost = acandpost
                    self.numaccept = self.numaccept + 1
                    self.acceptedSamples.append(9) #marking samples as accepted or rejected to change color
                    self.s.append(35)
                    #self.acceptPx.append(acandpost)
                    self.acceptPx.append(-0.5)
                    #self.acceptPx.append(np.random.normal(0, 4)*st.norm.pdf(acandidate, loc=self.state[j+1], scale=2.2))
                    #self.acceptX.append(st.norm.pdf(acandidate, loc=self.state[j], scale=2.2))
                    self.acceptX.append(acandidate)
                    self.ExArray.append(acandidate)
                    
                else:
                    self.state[j+1] = self.state[j]
                    self.acceptedSamples.append(0)
                    self.s.append(20)
                    #self.acceptPx.append(acandidate*st.norm.pdf(acandidate, loc=self.state[j+1], scale=2.2))
                    #self.acceptPx.append(acandpost)
                    self.acceptPx.append(-0.5)
                    #self.acceptPx.append(np.random.uniform(0, 1)*st.norm.pdf(acandidate, loc=self.state[j+1], scale=2.2))
                    self.acceptX.append(acandidate)
                    #self.acceptX.append(acandidate)
                    #self.acceptX.append(st.norm.pdf(acandidate, loc=self.state[j], scale=2.2))

                    #print(state[i+1])
                    #print(acceptX)
                
                return [self.acceptedSamples, self.s, self.acceptPx, self.acceptX]

            
            [c, s, acceptPx, acceptX] = rejection_sampling(1)

            self.Ex = np.array(self.ExArray)
            
            s = np.array(s)
            c = np.array(c)
            y = np.array(acceptPx)
            x = np.array(acceptX)
            #PEstArray = np.array(acceptPx)
            XEstArray = np.array(x)
            PEstArray = np.square(XEstArray) #for testing, change this
            meanXEst2Array = np.mean(np.square(XEstArray))

            #self.meanPEst = np.mean(PEstArray)
            self.meanPEst = np.mean(self.Ex)
            #self.varPEst = np.std(PEstArray)
            self.varPEst = np.var(self.Ex)/len(self.Ex)
            
            #print(x[:int(j-1)])
   
            j = j+1
            #yield np.c_[self.acceptedSamples, y[:int(j-1)], s[:int(j-1)], c[:int(j-1)]]
            yield np.c_[x[:int(j-1)], y[:int(j-1)], s[:int(j-1)], c[:int(j-1)]]
            
    

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)
        self.textvar.remove()
        self.arrow.remove()
        t = [b.remove() for b in self.barsH]
        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        #self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
        self.scat.set_sizes(data[:,2])
        # Set colors..
        self.scat.set_array(data[:, 3])

        self.countH, self.binsH, self.barsH = self.ax.hist(self.Ex, bins=20, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
        #Draw line?
        self.arrow = self.ax.arrow(data[i,0],data[i,1], data[i+1,0]-data[i,0],data[i+1,1]-data[i,1], head_width = 0.15,  
          width = 0.003, length_includes_head= True, color='b')
        # Set text
        textstr = '\n'.join((
            r'Number of draws: %d' % (i),
            r'Percent accepted: %.2f' % np.divide(self.numaccept,i),
            r'E(x)= %.4f ' % self.meanPEst,
            r'$\sigma^2/n$= %.4f' % self.varPEst))
        self.textvar = self.ax.text(0.05, 0.95, textstr, transform=self.ax.transAxes, fontsize=10, verticalalignment='top')
        #self.textvar = self.ax.text(max(self.domain)-2, 0.3*self.k, "E(p(x)): %d" % (i)

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,


if __name__ == '__main__':
    
    #user should define the following
    
    def p(x):
            #return np.exp(-x**2/2)*(np.sin(6*x)**2 + 3*np.cos(x)**2.*np.sin(4*x)**2+1)
            return 0.2*st.norm.pdf(x, loc=1, scale=0.3)+0.7*st.norm.pdf(x, loc=-0.5, scale=0.4)
            
            
    def q(x):
        return st.norm.pdf(x, loc=0, scale=1)
    
    xmin = -3
    xmax = 3
    
    #end user defined
    
    domain = np.linspace(xmin, xmax,10000, endpoint=True)
    k = max(p(domain)/q(domain))

    # Set up formatting for the movie files
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps =120, metadata=dict(artist='Me'))
        
    a = AnimatedScatter(1003, p, q, k, domain)
    plt.show()
    #a.ani.save('MCMC3.mp4', writer=writer)