# datacamp-notes
print 

#types of variables str, or string: a type to represent text. bool, or boolean: a type to represent logical values. Can only be True or False. Float to represent ational number
#Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

#Print out second element from areas
print(areas[1])

#Print out last element from areas
print(areas[-1])

#Print out the area of the living room
print(areas[5])

#Print out length of var1
print(len(var1))
#to now how it works one function in python help(name of the function)
#Sort full in descending order: full_sorted
full_sorted=sorted(full, reverse=True)

#string to experiment with: place
place = "poolhouse"

#Use upper() on place: place_up
place_up=place.upper()

#Print out place and place_up
print(place); print(place_up)

#Print out the number of o's in place
print(place.count("o"))

#Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

#Print out the index of the element 20.0
print(areas.index(20.0))

#Print out how often 9.50 appears in areas
print(areas.count(9.50))

#Definition of radius
r = 0.43

#Import the math package
import math

#Calculate C
C = 2*math.pi*r

#Calculate A
A = math.pi*r**2

#to intall numpy: pip3 install numpy

#Create a numpy array from height: np_height
np_height=np.array(height)

#Print out np_height
print(np_height)

#Convert np_height to m: np_height_m
np_height_m=np_height*0.0254

#Print np_height_m
print(np_height_m)
bmi = np_weight_kg / np_height_m ** 2

#Create the light array
light=bmi<21

#Print out light
print(light)

#Print out BMIs of all baseball players whose BMI is below 21
print(bmi[light])
#Create np_baseball (2 cols)
np_baseball = np.array(baseball)

#Print out the 50th row of np_baseball
print(np_baseball[49:])

#Select the entire second column of np_baseball: np_weight
np_weight=np_baseball[:,1]

#Print out height of 124th player
print(np_baseball[123:0])
#Print out addition of np_baseball and updated
print(np_baseball+updated)

#Create numpy array: conversion
conversion=np.array([0.0254,0.453592,1])

#Print out product of np_baseball and conversion
print(np_baseball*conversion)

# GRAPHS
Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

#Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(year,pop)

#Display the plot with plt.show()
plt.show()

#Put the x-axis on a logarithmic scale
plt.xscale('log')
#Build histogram with 20 bins
plt.hist(life_exp,bins=20)
#Strings
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2007'

#Add axis labels
plt.xlabel(xlab)
plt.ylabel(ylab)

#Add title
plt.title(title)
#Definition of tick_val and tick_lab
tick_val = [1000, 10000, 100000]
tick_lab = ['1k', '10k', '100k']

#Adapt the ticks on the x-axis
plt.xticks(tick_val,tick_lab)
import numpy as np

# --making graphs with bubbles
Store pop as a numpy array: np_pop
np_pop=np.array(pop)

#Double np_pop
np_pop=np_pop*2

#Update: set s argument to np_pop
plt.scatter(gdp_cap, life_exp, s= np_pop)

#Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])

#Display the plot
plt.show()
#Additional customizations
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

Add grid() call
plt.grid(True)

# Dictionaries
#Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }


#Print out the capital of France
print(europe['france']['capital'])

#Create sub-dictionary data
data = {'capital':'rome','population':5983}

#Add data to europe under key 'italy'
europe ['italy']=data

#Print europe
print(europe)

# PANDAS

#Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

#Import pandas as pd
import pandas as pd

#Create dictionary my_dict with three key:value pairs: my_dict
my_dict = {'country':names, 'drives_right':dr, 'cars_per_cap':cpc}

#Build a DataFrame cars from my_dict: cars
cars = pd.DataFrame(my_dict)

#Print cars
print(cars)
#Definition of row_labels
row_labels = ['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG']

#Specify row labels of cars
cars.index=row_labels
#Import the cars.csv data: cars
cars = pd.read_csv('cars.csv')
#Fix import by including index_col
cars = pd.read_csv('cars.csv', index_col=0)
#Print out drives_right value of Morocco
print(cars.loc['MOR','drives_right'])

#Print sub-DataFrame
print(cars.loc[['RU','MOR'],['country','drives_right']])
#print out drives_right column as Series
print(cars['drives_right'])

#Print out drives_right column as DataFrame
print(cars.loc[:,['drives_right']])

#Print out cars_per_cap and drives_right as DataFrame
print(cars.loc[:, ['cars_per_cap','drives_right']])


#Create medium: observations with cars_per_cap between 100 and 500
cpc=cars['cars_per_cap']
between=np.logical_and(cpc>100, cpc<500)
medium=cars[between]
print(medium)

# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Change for loop to use enumerate() and update print()
for index, a in enumerate(areas) :
    print("room"+str(index)+":"+str(a))
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
          
#Iterate over europe
for key, value in europe.items():
    print("the capital of"+key+" is "+str(value))
    
# For loop over np_baseball
for x in np.nditer(np_baseball):
    print(x)
#Roll the dice
dice=np.random.randint(1,7)

# Initialize random_walk
random_walk=[0]

# Complete the ___
for x in range(100) :
    # Set step: last element in random_walk
    step=random_walk[-1]

    # Roll the dice
    dice = np.random.randint(1,7)

    # Determine next step
    if dice <= 2:
        step = step - 1
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    # append next_step to random_walk
    random_walk.append(step)

# Print random_walk
print(random_walk)
# Initialization
random_walk = [0]

for x in range(100) :
    step = random_walk[-1]
    dice = np.random.randint(1,7)

    if dice <= 2:
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    random_walk.append(step)

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Plot random_walk
plt.plot(random_walk)


# Show the plot
plt.show()
