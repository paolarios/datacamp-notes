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

#Plot random_walk
plt.plot(random_walk)


#Show the plot
plt.show()

#Import numpy
import numpy as np

#Create array of DataFrame values: np_vals
np_vals = df.values

#Create new array of base 10 logarithm values: np_vals_log10
np_vals_log10=np.log10(np_vals)

#Create array of new DataFrame by passing df to np.log10(): df_log10
df_log10 = np.log10(df)

#Print original and new data containers
[print(x, 'has type', type(eval(x))) for x in ['np_vals', 'np_vals_log10', 'df', 'df_log10']]

# Zip the 2 lists together into one list of (key,value) tuples: zipped
zipped = list(zip(list_keys, list_values))

#Build a dictionary with the zipped list: data
data = dict(zipped)

#Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)
print(df)
Build a list of labels: list_labels
list_labels = ['year', 'artist','song','chart weeks']

Assign the list of labels to the columns attribute: df.columns
df.columns = list_labels
# Reading csv files
df1 = pd.read_csv(filepath_or_buffer='/usr/local/share/datasets/world_population.csv')

#Create a list of the new column labels: new_labels
new_labels = ['year', 'population']

#Read in the file, specifying the header and names parameters: df2
df2 = pd.read_csv(filepath_or_buffer='/usr/local/share/datasets/world_population.csv', header=0, names=new_labels)

# Plotting with pandas 
Create a plot with color='red'
df.plot(color='red')

#Add a title
plt.title('Temperature in Austin')
#Specify the x-axis label
plt.xlabel('Hours since midnight August 1, 2010')
#Specify the y-axis label
plt.ylabel('Temperature (degrees F)')
#Display the plot
plt.show()

# Generate a scatter plot
df.plot(kind='scatter', x='hp', y='mpg', s=sizes)

# Generate the box plots
df[cols].plot(subplots=True,kind='box')
#Print the minimum value of the Engineering column
print(df['Engineering'].min())

#Print the maximum value of the Engineering column
print(df['Engineering'].max())

#Construct the mean percentage per year: mean
mean = df.mean(axis='columns')

#Plot the average percentage per year
mean.plot(x='mean', y='Year')
#Filter the US population from the origin column: us
us=df.loc[df['origin']=='US',:]

#This formats the plots such that they appear on separate rows
fig, axes = plt.subplots(nrows=2, ncols=1)

#Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', normed=True, bins=30, range=(0,.3))
plt.show()

#Plot the CDF
df.fraction.plot(ax=axes[1], kind='hist', cumulative=True, normed=True, bins=30, range=(0,.3))
plt.show()

# Date time series 
#Prepare a format string: time_format
time_format = '%Y-%m-%d %H:%M'

#Convert date_list into a datetime object: my_datetimes
my_datetimes = pd.to_datetime(date_list, format=time_format)  

#Construct a pandas Series using temperature_list and my_datetimes: time_series
time_series = pd.Series(temperature_list, index=my_datetimes)
#Extract the hour from 9pm to 10pm on '2010-10-11': ts1
ts1 = ts0.loc['2010-10-11 21:00:00':'2010-10-11 22:00:00']

#Extract '2010-07-04' from ts0: ts2
ts2 = ts0.loc['2010-07-04']

#Extract data from '2010-12-15' to '2010-12-31': ts3
ts3 = ts0.loc['2010-12-15':'2010-12-31']
# Reindex without fill method: ts3
ts3 = ts2.reindex(ts1.index)

# Reindex with fill method, using forward fill: ts4
ts4 = ts2.reindex(ts1.index, method='ffill')

# Combine ts1 + ts2: sum12
sum12 = ts1+ts2

# Combine ts1 + ts3: sum13
sum13 = ts1+ts3

# Combine ts1 + ts4: sum14
sum14 = ts1+ts4

# Downsample to 6 hour data and aggregate by mean: df1
df1 = df.loc[:,'Temperature'].resample('6h').mean()

# Downsample to daily data and count the number of data points: df2
df2 = df.loc[:,'Temperature'].resample('D').count()
august = df.loc['2010-August','Temperature']
#another way
#Extract the August 2010 data: august
august = df['Temperature']['2010-August']
# Strip extra whitespace from the column names: df.columns
df.columns = df.columns.str.strip()

# Extract data for which the destination airport is Dallas: dallas
dallas = df['Destination Airport'].str.contains('DAL')

# Compute the total number of Dallas departures each day: daily_departures
daily_departures = dallas.resample('D').sum()

# Generate the summary statistics for daily Dallas departures: stats
stats = daily_departures.describe()

#Downsample to obtain only the daily highest temperatures in August: august_highs
august_highs = august.resample('D').max()

# Apply a rolling mean with a 24 hour window: smoothed moving averages
smoothed = unsmoothed.rolling(window=24).mean()

# Create a new DataFrame with columns smoothed and unsmoothed: august
august = pd.DataFrame({'smoothed':smoothed, 'unsmoothed':unsmoothed})

# Plot both smoothed and unsmoothed data using august.plot().
august.plot()
plt.show()
# Reset the index of ts2 to ts1, and then use linear interpolation to fill in the NaNs: ts2_interp
ts2_interp = ts2.reindex(ts1.index).interpolate(how='linear')

# Compute the absolute difference of ts1 and ts2_interp: differences 
differences = np.abs(ts1-ts2_interp)

# Generate and print summary statistics of the differences
print(differences.describe())
# Build a Boolean mask to filter out all the 'LAX' departure flights: mask
mask = df['Destination Airport'] == 'LAX'

# Use the mask to subset the data: la
la = df[mask]

# Combine two columns of data to create a datetime series: times_tz_none 
times_tz_none = pd.to_datetime(la['Date (MM/DD/YYYY)'] + ' ' +la['Wheels-off Time'] )

# Localize the time to US/Central: times_tz_central
times_tz_central =times_tz_none.dt.tz_localize('US/Central')

# Convert the datetimes from US/Central to US/Pacific
times_tz_pacific = times_tz_central.dt.tz_convert('US/Pacific')
# Plot the raw data before setting the datetime index
df.plot()
plt.show()


# Convert the 'Date' column into a collection of datetime objects: df.Date
df.Date = pd.to_datetime(df.Date)


# Set the index to be the converted 'Date' column
df.set_index('Date', inplace=True)

# Re-plot the DataFrame to see that the axis is now datetime aware!
df.plot()
plt.show()

# Plot the summer data
df.Temperature['2010-Jun':'2010-Aug'].plot()
plt.show()
plt.clf()

# Plot the one week data
df.Temperature['2010-06-10':'2010-06-17'].plot()
plt.show()
plt.clf()
# Convert the date column to string: df_dropped['date']
df_dropped['date'] = df_dropped['date'].astype(str)

# Pad leading zeros to the Time column: df_dropped['Time']
df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))

# Concatenate the new date and Time columns: date_string
date_string = df_dropped['date']+df_dropped['Time']

# Convert the date_string Series to datetime: date_times
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')

# Set the index to be the new date_times container: df_clean
df_clean = df_dropped.set_index(date_times)

# Print the output of df_clean.head()
print(df_clean.head())
# Print the dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc['2011-06-20 08:00':'2011-06-20 09:00', 'dry_bulb_faren'])

# Convert the dry_bulb_faren column to numeric values: df_clean['dry_bulb_faren']
df_clean['dry_bulb_faren'] = pd.to_numeric(df_clean['dry_bulb_faren'], errors='coerce')

# Print the transformed dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc['2011-06-20 08:00':'2011-06-20 09:00', 'dry_bulb_faren'])

# Convert the wind_speed and dew_point_faren columns to numeric values
df_clean['wind_speed'] = pd.to_numeric(df_clean['wind_speed'],errors='coerce')
df_clean['dew_point_faren'] = pd.to_numeric(df_clean['dew_point_faren'], errors='coerce')

# Print the median of the dry_bulb_faren column
print(df_clean['dry_bulb_faren'].median())

# Print the median of the dry_bulb_faren column for the time range '2011-Apr':'2011-Jun'
print(df_clean.loc['2011-Apr':'2011-Jun', 'dry_bulb_faren'].median())

# Print the median of the dry_bulb_faren column for the month of January
print(df_clean.loc['2011-Jan', 'dry_bulb_faren'].median())

# Select days that are sunny: sunny
sunny = df_clean.loc[df_clean['sky_condition']=='CLR']

# Select days that are overcast: overcast
overcast = df_clean.loc[df_clean['sky_condition'].str.contains('OVC')]

# Resample sunny and overcast, aggregating by maximum daily temperature
sunny_daily_max = sunny.resample('D').max()
overcast_daily_max = overcast.resample('D').max()

# Print the difference between the mean of sunny_daily_max and overcast_daily_max
print(sunny_daily_max.mean() - overcast_daily_max.mean())

# Downsample df_clean by day and aggregate by mean: daily_mean_2011
daily_mean_2011 = df_clean.resample('D').mean()

# Extract the dry_bulb_faren column from daily_mean_2011 using .values: daily_temp_2011
daily_temp_2011 = daily_mean_2011['dry_bulb_faren'].values

# Downsample df_climate by day and aggregate by mean: daily_climate
daily_climate = df_climate.resample('D').mean()

# Extract the Temperature column from daily_climate using .reset_index(): daily_temp_climate
daily_temp_climate = daily_climate.reset_index()['Temperature']

# Compute the difference between the two arrays and print the mean difference
difference = daily_temp_2011 - daily_temp_climate
print(difference.mean())
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Select the visibility and dry_bulb_faren columns and resample them: weekly_mean
weekly_mean = df_clean[['visibility','dry_bulb_faren']].resample('W').mean()

# Print the output of weekly_mean.corr()
print(weekly_mean.corr())

# Plot weekly_mean with subplots=True
weekly_mean.plot(subplots=True)
plt.show()

# Create a Boolean Series for sunny days: sunny
sunny = df_clean.loc[df_clean['sky_condition']=='CLR']

# Resample the Boolean Series by day and compute the sum: sunny_hours
sunny_hours = sunny.resample('D').sum()

# Resample the Boolean Series by day and compute the count: total_hours
total_hours = sunny.resample('D').count()

# Divide sunny_hours by total_hours: sunny_fraction
sunny_fraction = sunny_hours / total_hours

# Make a box plot of sunny_fraction
sunny_fraction.plot(kind='Box')
plt.show()

# Resample dew_point_faren and dry_bulb_faren by Month, aggregating the maximum values: monthly_max
monthly_max = df_clean[['dew_point_faren','dry_bulb_faren']].resample('M').max()

# Generate a histogram with bins=8, alpha=0.5, subplots=True
monthly_max.plot(kind='hist', bins=8, alpha=0.5, subplots=True)

# Show the plot
plt.show()

# Extract the maximum temperature in August 2010 from df_climate: august_max
august_max = df_climate.loc['2010-Aug','Temperature'].max()
print(august_max)

# Resample August 2011 temps in df_clean by day & aggregate the max value: august_2011
august_2011 = df_clean.loc['2011-Aug','dry_bulb_faren'].resample('D').max()

# Filter for days in august_2011 where the value exceeds august_max: august_2011_high

august_2011_high = august_2011.loc[august_2011 > august_max]

# Construct a CDF of august_2011_high
august_2011_high.plot(kind='hist', normed=True, cumulative=True, bins=25)

# Display the plot
plt.show()
