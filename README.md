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

# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Print out the index of the element 20.0
print(areas.index(20.0))

# Print out how often 9.50 appears in areas
print(areas.count(9.50))

# Definition of radius
r = 0.43

# Import the math package
import math

# Calculate C
C = 2*math.pi*r

# Calculate A
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
