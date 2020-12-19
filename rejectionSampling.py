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
        self.ax.axis([min(self.domain), max(self.domain), 0, 0.5*self.k])

        #Draw lines showing the target and proposal distributions
        line = self.ax.plot(self.domain, self.targetFunction(self.domain), 
                                    'y:', label = "Target Distribution")
        ine2 = self.ax.plot(self.domain, k*self.proposalFunction(self.domain), 
                                    'b:', label = "Proposal Distribution")

        #Add legend and title
        self.ax.legend( loc = 'upper right', prop={'size': 6}, borderpad=1)
        self.fig.suptitle('Rejection Sampling \n Using proposal draws from a scaled Gaussian distribution', fontsize=10)
        
        # FuncAnimation expects a sequence of artists in return, leave the comma in 
        return self.scat,
    

    def data_stream(self):
        """Generate data corresponding to selections from the target distribution and the proposal"""        
        
        Z = np.random.normal(0, 1, self.numpoints)
        U = np.random.uniform(0, 1, self.numpoints)*self.k*self.proposalFunction(Z)
        
        Z = np.array(Z)
        U = np.array(U)
        
        xyFull = np.column_stack((Z,U))
        
        
        j=1
        while j < self.numpoints+1:
            
            def rejection_sampling(iter):
                acceptedSamples = []
                acceptPx = []
                acceptX = []
                s = []
                
                for i in range(j):
                    
                    xy = xyFull[:int(i+1),:]
                    p_Z = self.targetFunction(Z[i])
                    
                    if U[i] <= p_Z:
                        acceptedSamples.append(9) #marking samples as accepted or rejected to change color
                        s.append(15)
                        acceptPx.append(U[i])
                        acceptX.append(xyFull[i,0])
                    else:
                        acceptedSamples.append(0)
                        s.append(15)
                
                return [xy, acceptedSamples, s, acceptPx, acceptX]
            
            [xy, c, s, acceptPx, acceptX] = rejection_sampling(1)
            
            s = np.array(s)
            c = np.array(c)
            xy = np.array(xy)
            #PEstArray = np.array(acceptPx)
            XEstArray = np.array(acceptX)
            PEstArray = np.square(XEstArray) #for testing, change this
            meanXEst2Array = np.mean(np.square(XEstArray))

            self.meanPEst = np.mean(XEstArray)
            self.varPEst = np.var(PEstArray)/len(PEstArray)
            
   
            j = j+1
            yield np.c_[xy[:,0], xy[:,1], s, c]
            
    

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
    
    #user should define the following
    
    def p(x):
            #return np.exp(-x**2/2)*(np.sin(6*x)**2 + 3*np.cos(x)**2.*np.sin(4*x)**2+1)
            return 0.2*st.norm.pdf(x, loc=1, scale=0.3)+0.7*st.norm.pdf(x, loc=-0.5, scale=0.4) #Gaussian mixture
            
            
    def q(x):
        return st.norm.pdf(x, loc=0, scale=1)
    
    xmin = -4
    xmax = 4
    
    #end user defined
    
    domain = np.linspace(xmin, xmax,10000, endpoint=True)
    k = max(p(domain)/q(domain))

    # Set up formatting for the movie files - uncomment this if you want to save it as a movie

    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps =200, metadata=dict(artist='Me'))
        
    a = AnimatedScatter(1003, p, q, k, domain)
    plt.show()
    #a.ani.save('rejectionSamp1.mp4', writer=writer)