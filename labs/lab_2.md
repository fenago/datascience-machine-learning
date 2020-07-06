### Introduction

In this scenario, we are going to go through a few concepts of statistics and probability, which might be a refresher for some of you. These concepts are important to go through if you want to be a data scientist. We will see examples to understand these concepts better. We will also look at how to implement those examples using actual Python code.

We'll be covering the following topics in this scenario:

- Types of data you may encounter and how to treat them accordingly
- Statistical concepts of mean, median, mode, standard deviation, and variance
- Probability density functions and probability mass functions

### Jupyter Notebooks
  
We will run Jupyter Notebook as a Docker container. This setup will take some time because of the size of the image.

## Login
When the container is running, execute this statement:
`docker logs jupyter 2>&1 | grep -v "HEAD" `{{execute}}


This will show something like:
```
copy/paste this URL into your browser when you connect for the first time, to login with a token:
    http://localhost:8888/?token=f89b02dd78479d52470b3c3a797408b20cc5a11e067e94b8
    THIS IS NOT YOUR TOKEN.  YOU HAVE TO SEARCH THE LOGS TO GET YOUR TOKEN
```

The token is the value behind `/?token=`. You need that for logging in.

**Note:** You can also run following command to get token directly: 
`docker exec -it jupyter bash -c 'jupyter notebook list' | cut -d'=' -f 2 | cut -d' ' -f 1`{{execute}}

Next, you can open the Jupyter Notebook at 
 https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/

 ### Types of data

 Alright, if you want to be a data scientist, we need to talk about the types of data that you might encounter, how to categorize them, and how you might treat them differently. Let's dive into the different flavors of data you might encounter:


This will seem pretty basic, but we've got to start with the simple stuff and we'll work our way up to the more complicated data mining and machine learning things. It is important to know what kind of data you're dealing with because different techniques might have different nuances depending on what kind of data you're handling. So, there are several flavors of data, if you will, and there are three specific types of data that we will primarily focus on. They are:

- Numerical data
- Categorical data
- Ordinal data

Again, there are different variations of techniques that you might use for different types of data, so you always need to keep in mind what kind of data you're dealing with when you're analyzing it.

**Numerical data**

Let's start with numerical data. It's probably the most common data type. Basically, it represents some quantifiable thing that you can measure. Some examples are heights of people, page load times, stock prices, and so on. Things that vary, things that you can measure, things that have a wide range of possibilities. Now there are basically two kinds of numerical data, so a flavor of a flavor if you will.

**Discrete data**

There's discrete data, which is integer-based and, for example, can be counts of some sort of event. Some examples are how many purchases did a customer make in a year. Well, that can only be discrete values. They bought one thing, or they bought two things, or they bought three things. They couldn't have bought, 2.25 things or three and three-quarters things. It's a discrete value that has an integer restriction to it.

**Continuous data**

The other type of numerical data is continuous data, and this is stuff that has an infinite range of possibilities where you can go into fractions. So, for example, going back to the height of people, there is an infinite number of possible heights for people. You could be five feet and 10.37625 inches tall, or the time it takes to do something like check out on a website could be any huge range of possibilities, 10.7625 seconds for all you know, or how much rainfall in a given day. Again, there's an infinite amount of precision there. So that's an example of continuous data.

To recap, numerical data is something you can measure quantitatively with a number, and it can be either discrete, where it's integer-based like an event count, or continuous, where you can have an infinite range of precision available to that data.

**Categorical data**

The second type of data that we're going to talk about is categorical data, and this is data that has no inherent numeric meaning.

Most of the time, you can't really compare one category to another directly. Things like gender, yes/no questions, race, state of residence, product category, political party; you can assign numbers to these categories, and often you will, but those numbers have no inherent meaning.


So, for example, I can say that the area of Texas is greater than the area of Florida, but I can't just say Texas is greater than Florida, they're just categories. There's no real numerical quantifiable meaning to them, it's just ways that we categorize different things.

Now again, I might have some sort of numerical assignation to each state. I mean, I could say that Florida is state number 3 and Texas state number 4, but there's no real relationship between 3 and 4 there, right, it's just a shorthand to more compactly represent these categories. So again, categorical data does not have any intrinsic numerical meaning; it's just a way that you're choosing to split up a set of data based on categories.

**Ordinal data**

The last category that you tend to hear about with types of data is ordinal data, and it's sort of a mixture of numerical and categorical data. A common example is star ratings for a movie or music, or what have you.


In this case, we have categorical data in that could be 1 through 5 stars, where 1 might represent poor and 5 might represent excellent, but they do have mathematical meaning. We do know that 5 means it's better than a 1, so this is a case where we have data where the different categories have a numerical relationship to each other. So, I can say that 1 star is less than 5 stars, I can say that 2 stars is less than 3 stars, I can say that 4 stars is greater than 2 stars in terms of a measure of quality. Now you could also think of the actual number of stars as discrete numerical data. So, it's definitely a fine line between these categories, and in a lot of cases you can actually treat them interchangeably.

So, there you have it, the three different types. There is numerical, categorical, and ordinal data. Let's see if it's sunk in. Don't worry, I'm not going to make you hand in your work or anything.

### Mean, median, and mode

Let's do a little refresher of statistics 101. This is like elementary school stuff, but good to go through it again and see how these different techniques are used: Mean, median, and mode. I'm sure you've heard those terms before, but it's good to see how they're used differently, so let's dive in.

This should be a review for most of you, a quick refresher, now that we're starting to actually dive into some real statistics. Let's look at some actual data and figure out how to measure these things.

#### Mean
The mean, as you probably know, is just another name for the average. To calculate the mean of a dataset, all you have to do is sum up all the values and divide it by the number of values that you have.

```
Sum of samples/Number of samples
```

Let's take this example, which calculates the mean (average) number of children per house in my neighborhood.

Let's say I went door-to-door in my neighborhood and asked everyone, how many children live in their household. (That, by the way, is a good example of discrete numerical data; remember from the previous section?) Let's say I go around and I found out that the first house has no kids in it, and the second house has two children, and the third household has three children, and so on and so forth. I amassed this little dataset of discrete numerical data, and to figure out the mean, all I do is add them all together and divide it by the number of houses that I went to.

Number of children in each house on my street:

```
0, 2, 3, 2, 1, 0, 0, 2, 0
```

The mean is (0+2+3+2+1+0+0+2+0)/9 = `1.11`

It comes out as 0 plus 2 plus 3 plus all the rest of these numbers divided by the total number of houses that I looked at, which is 9, and the mean number of children per house in my sample is 1.11. So, there you have it, mean.

**Median**

Median is a little bit different. The way you compute the median of the dataset is by sorting all the values (in either ascending or descending order), and taking the one that ends up in the middle.

So, for example, let's use the same dataset of children in my neighborhood

```
0, 2, 3, 2, 1, 0, 0, 2, 0
```

I would sort it numerically, and I can take the number that's slap dab in the middle of the data, which turns out to be 1.

```
0, 0, 0, 0, 1, 2, 2, 2, 3
```

Again, all I do is take the data, sort it numerically, and take the center point.

**Note:** If you have an even number of data points, then the median might actually fall in between two data points. It wouldn't be clear which one is actually the middle. In that case, all you do is, take the average of the two that do fall in the middle and consider that number as the median.

**The factor of outliers**

Now in the preceding example of the number of kids in each household, the median and the mean were pretty close to each other because there weren't a lot of outliers. We had 0, 1, 2, or 3 kids, but we didn't have some wacky family that had 100 kids. That would have really skewed the mean, but it might not have changed the median too much. That's why the median is often a very useful thing to look at and often overlooked.

**Note:** Median is less susceptible to outliers than the mean.

People have a tendency to mislead people with statistics sometimes. I'm going to keep pointing this out throughout the book wherever I can.

For example, you can talk about the mean or average household income in the United States,and that actual number from last year when I looked it up was $72,000 or so, but that doesn't really provide an accurate picture of what the typical American makes. That is because, if you look at the median income, it's much lower at $51,939. Why is that? Well, because of income inequality. There are a few very rich people in America, and the same is true in a lot of countries as well. America's not even the worst, but you know those billionaires, those super-rich people that live on Wall Street or Silicon Valley or some other super-rich place, they skew the mean. But there's so few of them that they don't really affect the median so much.

This is a great example of where the median tells a much better story about the typical person or data point in this example than the mean does. Whenever someone talks about the mean, you have to think about what does the data distribution looks like. Are there outliers that might be skewing that mean? And if the answer is potentially yes, you should also ask for the median, because often, that provides more insight than the mean or the average.

### Mode

Finally, we'll talk about mode. This doesn't really come up too often in practice, but you can't talk about mean and median without talking about mode. All mode means, is the most common value in a dataset.

Let's go back to my example of the number of kids in each house.

```
0, 2, 3, 2, 1, 0, 0, 2, 0
```

How many of each value are there:

```
0: 4, 1: 1, 2: 3, 3: 1
```

The MODE is 0

If I just look at what number occurs most frequently, it turns out to be 0, and the mode therefore of this data is 0. The most common number of children in a given house in this neighborhood is no kids, and that's all that means.

Now this is actually a pretty good example of continuous versus discrete data, because this only really works with discrete data. If I have a continuous range of data then I can't really talk about the most common value that occurs, unless I quantize that somehow into discrete values. So we've already run into one example here where the data type matters.

**Note:** Mode is usually only relevant to discrete numerical data, and not to continuous data.

A lot of real-world data tends to be continuous, so maybe that's why I don't hear too much about mode, but we see it here for completeness.

There you have it: mean, median, and mode in a nutshell. Kind of the most basic statistics stuff you can possibly do, but I hope you gained a little refresher there in the importance of choosing between median and mean. They can tell very different stories, and yet people tend to equate them in their heads, so make sure you're being a responsible data scientist and representing data in a way that conveys the meaning you're trying to represent. If you're trying to display a typical value, often the median is a better choice than the mean because of outliers, so remember that. Let's move on.

 ### Using mean, median, and mode in Python

 Let's start doing some real coding in Python and see how you compute the mean, median, and mode using Python in an IPython Notebook file.

So go ahead and open up the `MeanMedianMode.ipynb` file from the data files for this section if you'd like to follow along, which I definitely encourage you to do. If you need to go back to that earlier section on where to download these materials from, please go do that, because you will need these files for the section. Let's dive in!


#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for notebooks. Open and run `MeanMedianMode.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/MeanMedianMode.ipynb

### Calculating mean using the NumPy package

What we're going to do is create some fake income data, getting back to our example from the previous section. We're going to create some fake data where the typical American makes around $27,000 a year in this example, we're going to say that's distributed with a normal distribution and a standard deviation of 15,000. All numbers are completely made up, and if you don't know what normal distribution and standard deviation means yet, don't worry. I'm going to cover that a little later in the chapter, but I just want you to know what these different parameters represent in this example. It will make sense later on.

In our Python notebook, remember to import the NumPy package into Python, which makes computing mean, median, and mode really easy. We're going to use the import numpy as np directive, which means we can use np as a shorthand to call numpy from now on.

Then we're going to create a list of numbers called incomes using the np.random.normal function.

```
import numpy as np 
 
incomes = np.random.normal(27000, 15000, 10000) 
np.mean(incomes) 
```

The three parameters of the np.random.normal function mean I want the data centered around 27000, with a standard deviation of 15000, and I want python to make 10000 data points in this list.

Once I do that, I compute the average of those data points, or the mean by just calling np.mean on incomes which is my list of data. It's just that simple.

Let's go ahead and run that. Make sure you selected that code block and then you can hit the play button to actually execute it, and since there is a random component to these income numbers, every time I run it, I'm going to get a slightly different result, but it should always be pretty close to 27000.

```
Out[1]: 27173.098561362742
```

OK, so that's all there is to computing the mean in Python, just using NumPy (np.mean) makes it super easy. You don't have to write a bunch of code or actually add up everything and count up how many items you had and do the division. NumPy mean, does all that for you.

### Visualizing data using matplotlib

Let's visualize this data to make it make a little more sense. So there's another package called matplotlib, and again we're going to talk about that a lot more in the future as well, but it's a package that lets me make pretty graphs in IPython Notebooks, so it's an easy way to visualize your data and see what's going on.

In this example, we are using matplotlib to create a histogram of our income data broken up into 50 different buckets. So basically, we're taking our continuous data and discretizing it, and then we can call show on matplotlib.pyplot to actually display this histogram in line. Refer to the following code:

```
%matplotlib inline 
import matplotlib.pyplot as plt 
plt.hist(incomes, 50) 
plt.show()
```

Go ahead and select the code block and hit play. It will actually create a new graph for us as follows:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/10/1.jpg)

If you're not familiar with histograms or you need a refresher, the way to interpret this is that for each one of these buckets that we've discretized our data into is showing the frequency of that data.

So, for example, around 27,000-ish we see there's about 600 data points in that neighborhood for each given range of values. There's a lot of people around the 27,000 mark, but when you get over to outliers like 80,000, there is not a whole lot, and apparently there's some poor souls that are even in debt at -40,000, but again, they're very rare and not probable because we defined a normal distribution, and this is what a normal probability curve looks like. Again, we're going to talk about that more in detail later, but I just want to get that idea in your head if you don't already know it.

### Calculating median using the NumPy package

Alright, so computing the median is just as simple as computing the mean. Just like we had NumPy mean, we have a NumPy median function as well.

We can just use the median function on incomes, which is our list of data, and that will give us the median. In this case, that came up to $26,911, which isn't very different from the mean of $26988. Again, the initial data was random, so your values will be slightly different.

```
np.median(incomes) 
```

The following is the output of the preceding code:

```
Out[4]: 26911.948365056276
```

We don't expect to see a lot of outliers because this is a nice normal distribution. Median and mean will be comparable when you don't have a lot of weird outliers.

### Analyzing the effect of outliers

Just to prove a point, let's add in an outlier. We'll take Donald Trump; I think he qualifies as an outlier. Let's go ahead and add his income in. So I'm going to manually add this to the data using np.append, and let's say add a billion dollars (which is obviously not the actual income of Donald Trump) into the incomes data.

```
incomes = np.append(incomes, [1000000000]) 
```

What we're going to see is that this outlier doesn't really change the median a whole lot, you know, that's still going to be around the same value $26,911, because we didn't actually change where the middle point is, with that one value, as shown in the following example:

```
np.median(incomes) 
```

This will output the following:

```
Out[5]: 26911.948365056276
```

This gives a new output of:

```
np.mean(incomes)
```

The following is the output of the preceding code:

```
Out[5]:127160.38252311043 
```

Aha, so there you have it! It is a great example of how median and mean, although people tend to equate them in commonplace language, can be very different, and tell a very different story. So that one outlier caused the average income in this dataset to be over $127160 a year, but the more accurate picture is closer to 27,000 dollars a year for the typical person in this dataset. We just had the mean skewed by one big outlier.

The moral of the story is: take anyone who talks about means or averages with a grain of salt if you suspect there might be outliers involved, and income distribution is definitely a case of that.

### Calculating mode using the SciPy package

Calculating mode using the SciPy package
Finally, let's look at mode. We will just generate a bunch of random integers, 500 of them to be precise, that range between 18 and 90. We're going to create a bunch of fake ages for people.

```
ages = np.random.randint(18, high=90, size=500) 
ages 
```

Your output will be random, but should look something like the following screenshot:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/13/1.png)

Now, SciPy, kind of like NumPy, is a bunch of like handy-dandy statistics functions, so we can import stats from SciPy using the following syntax. It's a little bit different than what we saw before.

```
from scipy import stats 
stats.mode(ages) 
```

The code means, from the scipy package import stats, and I'm just going to refer to the package as stats, Tha means that I don't need to have an alias like I did before with NumPy, just different way of doing it. Both ways work. Then, I used the stats.mode function on ages, which is our list of random ages. When we execute the above code, we get the following output:

```
Out[11]: ModeResult(mode=array([39]), count=array([12])) 
```

So in this case, the actual mode is 39 that turned out to be the most common value in that array. It actually occurred 12 times.

Now if I actually create a new distribution, I would expect a completely different answer because this data really is completely random what these numbers are. Let's execute the above code blocks again to create a new distribution.

```
ages = np.random.randint(18, high=90, size=500) 
ages

from scipy import stats 
stats.mode(ages) 
```

The output for randomizing the equation is as distribution is as follows:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/13/2.png)

Make sure you selected that code block and then you can hit the play button to actually execute it.

In this case, the mode ended up being the number 29, which occurred 14 times.

```
Out[11]: ModeResult(mode=array([29]), count=array([14]))
```

So, it's a very simple concept. You can do it a few more times just for fun. It's kind of like rolling the roulette wheel. We'll create a new distribution again.

There you have it, mean, median, and mode in a nutshell. It's very simple to do using the SciPy and NumPy packages.

### Some exercises

I'm going to give you a little assignment in this section. If you open up `MeanMedianExercise.ipynb` file, there's some stuff you can play with. I want you to roll up your sleeves and actually try to do this.

You can open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/MeanMedianExercise.ipynb

In the file, we have some random e-commerce data. What this data represents is the total amount spent per transaction, and again, just like with our previous example, it's just a normal distribution of data. We can run that, and your homework is to go ahead and find the mean and median of this data using the NumPy package. Pretty much the easiest assignment you could possibly imagine. All the techniques you need are in the `MeanMedianMode.ipynb` file that we used earlier.

The point here is not really to challenge you, it's just to make you actually write some Python code and convince yourself that you can actually get a result and make something happen here. So, go ahead and play with that. If you want to play with it some more, feel free to play around with the data distribution here and see what effect you can have on the numbers. Try adding some outliers, kind of like we did with the income data. This is the way to learn this stuff: master the basics and the advance stuff will follow. Have at it, have fun.

Once your're ready, let's move forward to our next concept, standard deviation and variance.

### Standard deviation and variance

Let's talk about standard deviation and variance. The concepts and terms you've probably heard before, but let's go into a little bit more depth about what they really mean and how you compute them. It's a measure of the spread of a data distribution, and that will make a little bit more sense in a few minutes.

Standard deviation and variance are two fundamental quantities for a data distribution that you'll see over and over again in this book. So, let's see what they are, if you need a refresher.

### Variance

Let's look at a histogram, because variance and standard deviation are all about the spread of the data, the shape of the distribution of a dataset. Take a look at the following histogram:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/15/1.png)

Let's say that we have some data on the arrival frequency of airplanes at an airport, for example, and this histogram indicates that we have around 4 arrivals per minute and that happened on around 12 days that we looked at for this data. However, we also have these outliers. We had one really slow day that only had one arrival per minute, we only had one really fast day where we had almost 12 arrivals per minute. So, the way to read a histogram is look up the bucket of a given value, and that tells you how frequently that value occurred in your data, and the shape of the histogram could tell you a lot about the probability distribution of a given set of data.

We know from this data that our airport is very likely to have around 4 arrivals per minute, but it's very unlikely to have 1 or 12, and we can also talk specifically about the probabilities of all the numbers in between. So not only is it unlikely to have 12 arrivals per minute, it's also very unlikely to have 9 arrivals per minute, but once we start getting around 8 or so, things start to pick up a little bit. A lot of information can be had from a histogram.

**Note:** Variance measures how spread-out the data is.

### Measuring variance

We usually refer to variance as sigma squared, and you'll find out why momentarily, but for now, just know that variance is the average of the squared differences from the mean.

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/16/1.png)

Let's look at what happens there, so (-3.4)2 is a positive 11.56 and (-0.4)2 ends up being a much smaller number, that is 0.16, because that's much closer to the mean of 4.4. Also (0.6)2 turned out to be close to the mean, only 0.36. But as we get up to the positive outlier, (3.6)2 ends up being 12.96. That gives us: (11.56, 0.16, 0.36, 0.16, 12.96).

To find the actual variance value, we just take the average of all those squared differences. So we add up all these squared variances, divide the sum by 5, that is number of values that we have, and we end up with a variance of 5.04.

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/16/2.png)

OK, that's all variance is.

### Standard deviation

Now typically, we talk about standard deviation more than variance, and it turns out standard deviation is just the square root of the variance. It's just that simple.

So, if I had this variance of 5.04, the standard deviation is 2.24. So you see now why we said that the variance = (σ)2. It's because σ itself represents the standard deviation. So,if I take the square root of (σ)2, I get sigma. That ends up in this example to be 2.24.

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/17/1.png)

#### Identifying outliers with standard deviation
Here's a histogram of the actual data we were looking at in the preceding example for calculating variance.

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/17/2.png)

Now we see that the number 4 occurred twice in our dataset, and then we had one 1, one 5, and one 8.

The standard deviation is usually used as a way to think about how to identify outliers in your dataset. If I say if I'm within one standard deviation of the mean of 4.4, that's considered to be kind of a typical value in a normal distribution. However, you can see in the preceding diagram, that the numbers 1 and 8 actually lie outside of that range. So if I take 4.4 plus or minus 2.24, we end up around 7 and 2, and 1 and 8 both fall outside of that range of a standard deviation. So we can say mathematically, that 1 and 8 are outliers. We don't have to guess and eyeball it. Now there is still a judgment call as to what you consider an outlier in terms of how many standard deviations a data point is from the mean.


You can generally talk about how much of an outlier a data point is by how many standard deviations (or sometimes how many-sigmas) from the mean it is.

So that's something you'll see standard deviation used for in the real world.

### Population variance versus sample variance

There is a little nuance to standard deviation and variance, and that's when you're talking about population versus sample variance. If you're working with a complete set of data, a complete set of observations, then you do exactly what I told you. You just take the average of all the squared variances from the mean and that's your variance.

However, if you're sampling your data, that is, if you're taking a subset of the data just to make computing easier, you have to do something a little bit different. Instead of dividing by the number of samples, you divide by the number of samples minus 1. Let's look at an example.

We'll use the sample data we were just studying for people standing in a line. We took the sum of the squared variances and divided by 5, that is the number of data points that we had, to get 5.04.

```
σ2 = (11.56 + 0.16 + 0.36 + 0.16 + 12.96) / 5 = 5.04
```

If we were to look at the sample variance, which is designated by S2, it is found by the sum of the squared variances divided by 4, that is (n - 1). This gives us the sample variance, which comes out to 6.3.

```
S2 = (11.56 + 0.16 + 0.36 + 0.16 + 12.96) / 4 = 6.3
```

So again, if this was some sort of sample that we took from a larger dataset, that's what you would do. If it was a complete dataset, you divide by the actual number. Okay, that's how we calculate population and sample variance, but what's the actual logic behind it?

 ### The Mathematical explanation

 As for why there is difference between population and sample variance, it gets into really weird things about probability that you probably don't want to think about too much, and it requires some fancy mathematical notation, I try to avoid notation in this book as much as possible because I think the concepts are more important, but this is basic enough stuff and that you will see it over and over again.

As we've seen, population variance is usually designated as sigma squared (σ2), with sigma (σ) as standard deviation, and we can say that is the summation of each data point X minus the mean, mu, squared, that's the variance of each sample squared over N, the number of data points , and we can express it with the following equation:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/19/1.png)

- X denotes each data point
- µ denotes the mean
- N denotes the number of data points

Sample variance similarly is designated as S2, with the following equation:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/19/2.png)

- X denotes each data point
- M denotes the mean
- N-1 denotes the number of data points minus 1

That's all there is to it.

### Analyzing standard deviation and variance on a histogram

The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for notebooks. Open and run `StdDevVariance.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/StdDevVariance.ipynb

We will discuss this notebook in the next steps.


```
%matplotlib inline 
import numpy as np 
import matplotlib.pyplot as plt 
incomes = np.random.normal(100.0, 20.0, 10000) 
plt.hist(incomes, 50) 
plt.show() 
```

We use matplotlib to plot a histogram of some normally distributed random data, and we call it incomes. We're saying it's going to be centered around 100 (hopefully that's an hourly rate or something and not annual, or some weird denomination), with a standard deviation of 20 and 10,000 data points.

Let's go ahead and generate that by executing that above code block and plotting it as shown in the following graph:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/21/1.png)

We have 10,000 data points centered around 100. With a normal distribution and a standard deviation of 20, a measure of the spread of this data, you can see that the most common occurrence is around 100, and as we get further and further from that, things become less and less likely. The standard deviation point of 20 that we specified is around 80 and around 120. You can see in the histogram that this is the point where things start to fall off sharply, so we can say that things beyond that standard deviation boundary are unusual.

### Using Python to compute standard deviation and variance

Now, NumPy also makes it incredibly easy to compute the standard deviation and variance. If you want to compute the actual standard deviation of this dataset that we generated, you just call the std function right on the dataset itself. So, when NumPy creates the list, it's not just a normal Python list, it actually has some extra stuff tacked onto it so you can call functions on it, like std for standard deviation. Let's do that now:

```
incomes.std() 
```

This gives us something like the following output (remember that we used random data, so your figures won't be exactly the same as mine):

```
20.024538249134373 
```

When we execute that, we get a number pretty close to 20, because that's what we specified when we created our random data. We wanted a standard deviation of 20. Sure enough, 20.02, pretty close.

The variance is just a matter of calling var.

```
incomes.var() 
```

This gives me the following:

```
400.98213209104557 
```

It comes out to pretty close to 400, which is 202. Right, so the world makes sense! Standard deviation is just the square root of the variance, or you could say that the variance is the standard deviation squared. Sure enough, that works out, so the world works the way it should.

### Try it yourself

I want you to dive in here and actually play around with it, make it real, so try out different parameters on generating that normal data. Remember, this is a measure of the shape of the distribution of the data, so what happens if I change that center point? Does it matter? Does it actually affect the shape? Why don't you try it out and find out?

Try messing with the actual standard deviation, that we've specified, to see what impact that has on the shape of the graph. Maybe try a standard deviation of 30, and you know, you can see how that actually affects things. Let's make it even more dramatic, like 50. Just play around with 50. You'll see the graph starting to get a little bit fatter. Play around with different values, just get a feel of how these values work. This is the only way to really get an intuitive sense of standard deviation and variance. Mess around with some different examples and see the effect that it has.

So that's standard deviation and variance in practice. You got hands on with some of it there, and I hope you played around a little bit to get some familiarity with it. These are very important concepts and we'll talk about standard deviations a lot throughout the book and no doubt throughout your career in data science, so make sure you've got that under your belt. Let's move on.

### Probability density function and probability mass function

So we've already seen some examples of a normal distribution function for some of the examples in this book. That's an example of a probability density function, and there are other types of probability density functions out there. So let's dive in and see what it really means and what some other examples of them are.

#### The probability density function and probability mass functions
We've already seen some examples of a normal distribution function for some of the code we've looked at in this book. That's an example of a probability density function, and there are other types of probability density functions out there. Let's dive in and see what that really means and what some other examples of them there are.

### Probability density functions


Let's talk about probability density functions, and we've used one of these already in the book. We just didn't call it that. Let's formalize some of the stuff that we've talked about. For example, we've seen the normal distribution a few times, and that is an example of a probability density function. The following figure is an example of a normal distribution curve

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/25/1.png)


It's conceptually easy to try to think of this graph as the probability of a given value occurring, but that's a little bit misleading when you're talking about continuous data. Because there's an infinite number of actual possible data points in a continuous data distribution. There could be 0 or 0.001 or 0.00001 so the actual probability of a very specific value happening is very, very small, infinitely small. The probability density function really tells the probability of a given range of values occurring. So that's the way you've got to think about it.

So, for example, in the normal distribution shown in the above graph, between the mean (0) and one standard deviation from the mean (1σ) there's a 34.1% chance of a value falling in that range. You can tighten this up or spread it out as much as you want, figure out the actual values, but that's the way to think about a probability density function. For a given range of values it gives you a way of finding out the probability of that range occurring.

- You can see in the graph, as you get close to the mean (0), within one standard deviation (**-1σ** and **1σ**), you're pretty likely to land there. I mean, if you add up 34.1 and 34.1, which equals to 68.2%, you get the probability of landing within one standard deviation of the mean.
- However, as you get between two and three standard deviations (-3σ to -2σ and 2σ to 3σ), we're down to just a little bit over **4%** (4.2%, to be precise).
- As you get out beyond three standard deviations (-3σ and 3σ) then we're much less than 1%.

So, the graph is just a way to visualize and talk about the probabilities of the given data point happening. Again, a probability distribution function gives you the probability of a data point falling within some given range of a given value, and a normal function is just one example of a probability density function. We'll look at some more in a moment.

### Probability mass functions

Now when you're dealing with discrete data, that little nuance about having infinite numbers of possible values goes away, and we call that something different. So that is a probability mass function. If you're dealing with discrete data, you can talk about probability mass functions. Here's a graph to help visualize this:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/26/1.png)

For example, you can plot a normal probability density function of continuous data on the black curve shown in the graph, but if we were to quantize that into a discrete dataset like we would do with a histogram, we can say the number 3 occurs some set number of times, and you can actually say the number 3 has a little over 30% chance of occurring. So a probability mass function is the way that we visualize the probability of discrete data occurring, and it looks a lot like a histogram because it basically is a histogram.

**Note:**

Terminology difference: A probability density function is a solid curve that describes the probability of a range of values happening with continuous data. A probability mass function is the probabilities of given discrete values occurring in a dataset.


