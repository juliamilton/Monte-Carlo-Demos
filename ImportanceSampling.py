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
import random




class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numpoints, targetFunction, proposalFunction, domain):
        self.numpoints = numpoints+2
        self.targetFunction = targetFunction
        self.proposalFunction = proposalFunction
        self.domain = domain
        self.scaling = max(self.targetFunction(self.domain))
    #def __init__(self):
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        self.textvar = self.ax.text(0.05, 0.95, "Init", transform=self.ax.transAxes, fontsize=12, verticalalignment='top')
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1, 
                                          init_func=self.setup_plot, blit=False, cache_frame_data=False, save_count=self.numpoints)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, c=c, s=s, vmin=0, vmax=1,
                                    cmap="viridis", edgecolor="none")
        self.ax.axis([min(self.domain), max(self.domain), -0.2, self.scaling+0.1])
        line = self.ax.plot(self.domain, self.targetFunction(self.domain), 'b:', label = "Target Distribution")
        self.ax.legend( loc = 'upper right', prop={'size': 6}, borderpad=1)
        self.fig.suptitle('Importance Sampling \n Using proposal draws from a Uniform distribution', fontsize=10)
        #self.hist = plt.hist(x,bins =np.linspace(-4, 4, 50))
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,
    

    def data_stream(self):
        """Generate data corresponding to selections from the target distribution and the proposal"""        
        
        zerosForPlotting = np.zeros(self.numpoints)
        proposalDraws = np.random.uniform(-4,4, self.numpoints)
        #U = np.random.uniform(0, 1, self.numpoints)*self.k*self.proposalFunction(Z)
        
        proposalDraws= np.array(proposalDraws)
        #U = np.array(U)
        y = self.proposalFunction(proposalDraws)
        xyFull = np.column_stack((proposalDraws,y))
        
        
        j=1
        while j < self.numpoints+1:
        #while True:
            
            def rejection_sampling(iter):
                targetProbs = []
                proposalProbs = []
                c = []
                s = []
                acceptPx = []
                acceptX = []
                p_i_array = []

                t_x = self.targetFunction(proposalDraws)
                #p_x = self.proposalFunction(proposalDraws)
                p_x = 1/self.numpoints

                numer = np.divide(t_x,p_x)
                #denom = np.sum(np.divide(t_x,p_x))
                denom = np.sum(t_x)

                p_i = np.divide(numer, denom)

                weightedSamples = random.choices(population = proposalDraws, weights = p_i, k=self.numpoints)

                s =np.multiply(p_i,300)

                c = np.random.uniform(0,1,self.numpoints)
                
                return [c, s, p_i, weightedSamples]
            
            [c, s, p_i_array, weightedSamples] = rejection_sampling(1)
            
            s = np.array(s)
            c = np.array(c)
            xy = xyFull
            pi = np.array(p_i_array)
            #PEstArray = np.array(acceptPx)
            #XEstArray = np.multiply(np.square(proposalDraws[:int(j-1)]),pi[:int(j-1)])
            XEstArray = np.square(np.array(weightedSamples[:int(j-1)]))
            #PEstArray = np.square(XEstArray) #for testing, change this
            meanXEst2Array = np.mean(XEstArray)

            self.meanPEst = np.mean(XEstArray)
            #self.varPEst = np.std(PEstArray)
            self.varPEst = np.var(XEstArray)/len(XEstArray)
            
            #print(self.varPEst)

            print(XEstArray)

            #pi = np.multiply(pi, proposalDraws[:int(j)])
   
            j = j+1
            #yield np.c_[xy[:int(j-1),0], pi[:int(j-1)], s[:int(j-1)], c[:int(j-1)]]
            yield np.c_[xy[:int(j-1),0], zerosForPlotting[:int(j-1)], s[:int(j-1)], c[:int(j-1)]]
    

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)
        self.textvar.remove()
        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        #self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
        self.scat.set_sizes(data[:,2])
        # Set colors..
        self.scat.set_array(data[:, 3])

        #self.hist = plt.hist(data[:, 0],bins =np.linspace(-4, 4, 50))
        # Set text
        textstr = '\n'.join((
            r'Number of draws: %d' % (i),
            r'E(p(x))= %.4f ' % self.meanPEst,
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
        return st.uniform.pdf(x,-4,8)
        #return st.norm.pdf(x, loc=0, scale=1)
    
    xmin = -4
    xmax = 4
    
    #end user defined
    
    domain = np.linspace(xmin, xmax,10000, endpoint=True)
    #k = max(p(domain)/q(domain))

    # Set up formatting for the movie files
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps =200, metadata=dict(artist='Me'))
        
    a = AnimatedScatter(1000, p, q, domain)
    plt.show()
    #a.ani.save('im.mp4', writer=writer)
