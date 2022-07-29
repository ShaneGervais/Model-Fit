#ModelFit by @Shane Gervais
#For the curiculum of PHYS3336
#This python code is to test a certain data set
#provided by the lab. 

#Import methods that will help us for the analysis
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np

#Integrates the excel sheet containing our data
df = pd.read_excel('Model1.xlsx')


#Defining our variables
x = df.x
t = df.t
x = np.array(x)
t = np.array(t)
x = x - x.mean()

#Plots our data from the excel sheet
plt.figure(figsize = (1000, 10))
plt.plot(t,x,'b-o')
plt.ylabel("x9 (/m)")
plt.xlabel("t9 (/s)")
plt.show()

#Finds our peaks for each wave segments
#and appends them in an array called peaks
peaks = []
n = 2
for i in range(n, len(x)-n):
    if x[i] == max(x[i - n:i + n]):
        peaks.append(i)


#Gives us the plot of our data but shows the
#point of each peaks in our plot.
plt.figure(figsize = (1000, 10))
plt.plot(t,x,'b-o')
plt.plot(t[peaks],x[peaks],"ys")
plt.ylabel("x9 (/m)")
plt.xlabel("t9 (/s)")
plt.show()


#Finds the periods for each wave segment and 
#stores in variable Period
Period = t[peaks[1:]] - t[peaks[0:np.size(peaks) -1]]


#Finds the most frequent Period and displays it
T = np.median(Period)
print('The period is (/seconds): ')
print(T)

#Plots peak amplitudes at there times
plt.figure()
plt.plot(t[peaks], x[peaks])
plt.ylabel("Peak amplitude (/m)")
plt.xlabel("Time at peak amplitude (/s)")
plt.show()


#Uses python polyfit in order to find the best
#parameters to describe the log of the peak
#amplitudes. 
M, B = np.polyfit(t[peaks], np.log(x[peaks]), 1)
plt.figure()
plt.plot(t[peaks], np.log(x[peaks]), 'b-o')
plt.xlabel("Time at peak amplitude (/s)")
plt.ylabel("log(Peak amplitude) (/m)")
plt.plot(t[peaks], M*t[peaks]+B, 'r')
plt.show()

#Displays these parameters.
print("The parameter M in our linear regression is: ")
print(M)
print("The parameter B in our linear regression is: ")
print(B)

#Define these parameters for an exponential decay
#model.
a = np.exp(B)
b = np.exp(M)
w = 2*np.pi/T

#Defines our model and plots it against our
#data in order to see its fit.
model = a*np.power(b, t)*np.cos(w*t)
plt.figure(figsize = (1000, 10))
plt.plot(t, x, 'b')
plt.plot(t, model, 'r')
plt.ylabel("x9 (/m)")
plt.xlabel("t9 (/s)")
plt.show()

#Finds the phase difference for our model
#and prints it.
phase_Diff = np.arccos(x[0]/(np.exp(M*t[0] +B))) - (w*t[0])
print("The phase difference for this model is: ")
print(phase_Diff)