### Introduction

After going through some of the simpler concepts of statistics and probability in the previous chapter, we're now going to turn our attention to some more advanced topics that you'll need to be familiar with to get the most out of the remainder of this book. Don't worry, they're not too complicated.

We'll be covering the following topics in this scenario:

- Understanding conditional probability with examples
- Understanding Bayes' theorem and its importance

### Jupyter Notebooks

We will run Jupyter Notebook as a Docker container. This setup will take some time because of the size of the image.

## Login
When the container is running, execute this statement:
`docker logs jupyter 2>&1 | grep -v "HEAD" `


This will show something like:
```
copy/paste this URL into your browser when you connect for the first time, to login with a token:
    http://localhost:8888/?token=f89b02dd78479d52470b3c3a797408b20cc5a11e067e94b8
    THIS IS NOT YOUR TOKEN.  YOU HAVE TO SEARCH THE LOGS TO GET YOUR TOKEN
```

The token is the value behind `/?token=`. You need that for logging in.

**Note:** You can also run following command to get token directly: 
`docker exec -it jupyter bash -c 'jupyter notebook list' | cut -d'=' -f 2 | cut -d' ' -f 1`

Next, you can open the Jupyter Notebook at 
 https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/

 ### Conditional probability

 Next, we're going to talk about conditional probability. It's a very simple concept. It's trying to figure out the probability of something happening given that something else occurred. Although it sounds simple, it can be actually very difficult to wrap your head around some of the nuances of it. So get an extra cup of coffee, make sure your thinking cap's on, and if you're ready for some more challenging concepts here. Let's do this.

Conditional probability is a way to measure the relationship between two things happening to each other. Let's say I want to find the probability of an event happening given that another event already happened. Conditional probability gives you the tools to figure that out.

What I'm trying to find out with conditional probability is if I have two events that depend on each other. That is, what's the probability that both will occur?

In mathematical notation, the way we indicate things here is that P(A,B) represents the probability of both A and B occurring independent of each other. That is, what's the probability of both of these things happening irrespective of everything else.

Whereas this notation, P(B|A), is read as the probability of B given A. So, what is the probability of B given that event A has already occurred? It's a little bit different, and these things are related like this:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-03-02/steps/1.png)

The probability of B given A is equal to the probability of A and B occurring over the probability of A alone occurring, so this teases out the probability of B being dependent on the probability of A.

It'll make more sense with an example here, so bear with me.

Let's say that I give you, my readers, two tests, and 60% of you pass both tests. Now the first test was easier, 80% of you passed that one. I can use this information to figure out what percentage of readers who passed the first test also passed the second. So here's a real example of the difference between the probability of B given A and the probability of A and B.

I'm going to represent A as the probability of passing the first test, and B as the probability of passing the second test. What I'm looking for is the probability of passing the second test given that you passed the first, that is, P (B|A).

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-03-02/steps/2.png)

So the probability of passing the second test given that you passed the first is equal to the probability of passing both tests, P(A,B) (I know that 60% of you passed both tests irrespective of each other), divided by the probability of passing the first test, P(A), which is 80%. It's worked out to 60% passed both tests, 80% passed the first test, therefore the probability of passing the second given that you passed the first works out to 75%.

OK, it's a little bit tough to wrap your head around this concept. It took me a little while to really internalize the difference between the probability of something given something and the probability of two things happening irrespective of each other. Make sure you internalize this example and how it's really working before you move on.

### Conditional probability exercises in Python

Alright, let's move on and do another more complicated example using some real Python code. We can then see how we might actually implement these ideas using Python.

Let's put conditional probability into action here and use some of the ideas to figure out if there's a relationship between age and buying stuff using some fabricated data. Go ahead and open up the `ConditionalProbabilityExercise.ipynb` here and follow along with me if you like.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `ConditionalProbabilityExercise.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/ConditionalProbabilityExercise.ipynb


What I'm going to do is write a little bit of Python code that creates some fake data:

```
from numpy import random 
random.seed(0) 
 
totals = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0} 
purchases = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0} 
totalPurchases = 0 
for _ in range(100000): 
    ageDecade = random.choice([20, 30, 40, 50, 60, 70]) 
    purchaseProbability = float(ageDecade) / 100.0 
    totals[ageDecade] += 1 
    if (random.random() < purchaseProbability): 
        totalPurchases += 1 
        purchases[ageDecade] += 1 
```

What I'm going to do is take 100,000 virtual people and randomly assign them to an age bracket. They can be in their 20s, their 30s, their 40s, their 50s, their 60s, or their 70s. I'm also going to assign them a number of things that they bought during some period of time, and I'm going to weight the probability of purchasing something based on their age.

What this code ends up doing is randomly assigning each person to an age group using the random.choice() function from NumPy. Then I'm going to assign a probability of purchasing something, and I have weighted it such that younger people are less likely to buy stuff than older people. I'm going to go through 100,000 people and add everything up as I go, and what I end up with are two Python dictionaries: one that gives me the total number of people in each age group, and another that gives me the total number of things bought within each age group. I'm also going to keep track of the total number of things bought overall. Let's go ahead and run that code.


If you want to take a second to kind of work through that code in your head and figure out how it works, you've got the IPython Notebook. You can go back into that later too. Let's take a look what we ended up with.


Our totals dictionary is telling us how many people are in each age bracket, and it's pretty evenly distributed, just like we expected. The amount purchased by each age group is in fact increasing by age, so 20-year-olds only bought about 3,000 things and 70-year-olds bought about 11,000 things, and overall the entire population bought about 45,000 things.

Let's use this data to play around with the ideas of conditional probability. Let's first figure out what's the probability of buying something given that you're in your 30s. The notation for that will be P(E|F) if we're calling purchase E, and F as the event that you're in your 30s.

Now we have this fancy equation that gave you a way of computing P(E|F) given P(E,F), and P(E), but we don't need that. You don't just blindly apply equations whenever you see something. You have to think about your data intuitively. What is it telling us? I want to figure out the probability of purchasing something given that you're in your 30s. Well I have all the data I need to compute that directly.

```
PEF = float(purchases[30]) / float(totals[30]) 
```

I have how much stuff 30-year-olds purchased in the purchases[30] bucket, and I know how many 30-year-olds there are. So I can just divide those two numbers to get the ratio of 30-year-old purchases over the number of 30-year-olds. I can then output that using the print command:

```
print ("P(purchase | 30s): ", PEF) 
```

I end up with a probability of purchasing something given that you're in your 30s of being about 30%:

```
P(purchase | 30s): 0.2992959865211 
```

**Note:** that if you're using Python 2, the print command doesn't have the surrounding brackets, so it would be:

```
print "p(purchase | 30s): ", PEF 
```

If I want to find P(F), that's just the probability of being 30 overall, I can take the total number of 30-year-olds divided by the number of people in my dataset, which is 100,000:

```
PF = float(totals[30]) / 100000.0 
print ("P(30's): ", PF) 
```

Again, remove those brackets around the print statement if you're using Python 2. That should give the following output:

```
P(30's): 0.16619 
```

I know the probability of being in your 30s is about 16%.

We'll now find out P(E), which just represents the overall probability of buying something irrespective of your age:

```
PE = float(totalPurchases) / 100000.0 
print ("P(Purchase):", PE) 
 
P(Purchase): 0.45012 
```

That works out to be, in this example, about 45%. I can just take the total number of things purchased by everybody regardless of age and divide it by the total number of people to get the overall probability of purchase.


Alright, so what do I have here? I have the probability of purchasing something given that you're in your 30s being about 30%, and then I have the probability of purchasing something overall at about 45%.

Now if E and F were independent, if age didn't matter, then I would expect the P(E|F) to be about the same as P(E). I would expect the probability of buying something given that you're in your 30s to be about the same as the overall probability of buying something, but they're not, right? And because they're different, that tells me that they are in fact dependent, somehow. So that's a little way of using conditional probability to tease out these dependencies in the data.

Let's do some more notation stuff here. If you see something like P(E)P(F) together, that means multiply these probabilities together. I can just take the overall probability of purchase multiplied by the overall probability of being in your 30s:

```
print ("P(30's)P(Purchase)", PE * PF) 
 
P(30's)P(Purchase) 0.07480544280000001 
```

That worked out to about 7.5%.

Just from the way probabilities work, I know that if I want to get the probability of two things happening together, that would be the same thing as multiplying their individual probabilities. So it turns out that P(E,F) happening, is the same thing as P(E)P(F).

```
print ("P(30's, Purchase)", float(purchases[30]) / 100000.0) 
P(30's, Purchase) 0.04974 
```

Now because of the random distribution of data, it doesn't work out to be exactly the same thing. We're talking about probabilities here, remember, but they're in the same ballpark, so that makes sense, about 5% versus 7%, close enough.

Now that is different again from P(E|F), so the probability of both being in your 30s and buying something is different than the probability of buying something given that you're in your 30s.

Now let's just do a little sanity check here. We can check our equation that we saw in the Conditional Probability section earlier, that said that the probability of buying something given that you're in your 30s is the same as the probability of being in your 30s and buying something over the probability of buying something. That is, we check if P(E|F)=P(E,F)/P(F).

```
(float(purchases[30]) / 100000.0) / PF  
```

This gives us:

```
Out []:0.29929598652145134 
```

Sure enough, it does work out. If I take the probability of buying something given that you're in your 30s over the overall probability, we end up with about `30%`, which is pretty much what we came up with originally for `P(E|F)`. So the equation works, yay!

Alright, it's tough to wrap your head around some of this stuff. It's a little bit confusing, I know, but if you need to, go through this again, study it, and make sure you understand what's going on here. I've tried to put in enough examples here to illustrate different combinations of thinking about this stuff. Once you've got it internalized, I'm going to challenge you to actually do a little bit of work yourself here.

### Conditional probability assignment

What I want you to do is modify the following Python code which was used in the preceding section.

```
from numpy import random 
random.seed(0) 
 
totals = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0} 
purchases = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0} 
totalPurchases = 0 
for _ in range(100000): 
ageDecade = random.choice([20, 30, 40, 50, 60, 70]) 
purchaseProbability = 0.4 
totals[ageDecade] += 1 
if (random.random() < purchaseProbability): 
totalPurchases += 1 
purchases[ageDecade] += 1 
```

Modify it to actually not have a dependency between purchases and age. Make that an evenly distributed chance as well. See what that does to your results. Do you end up with a very different conditional probability of being in your 30s and purchasing something versus the overall probability of purchasing something? What does that tell you about your data and the relationship between those two different attributes? Go ahead and try that, and make sure you can actually get some results from this data and understand what's going on, and I'll run through my own solution to that exercise in just a minute.

So that's conditional probability, both in theory and in practice. You can see there's a lot of little nuances to it and a lot of confusing notation. Go back and go through this section again if you need to wrap your head around it. I gave you a homework assignment, so go off and do that now, see if you can actually modify my code in that IPython Notebook to produce a constant probability of purchase for those different age groups. Come back and we'll take a look at how I solved that problem and what my results were.

### My assignment solution

The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `ConditionalProbabilitySolution.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/ConditionalProbabilitySolution.ipynb

Did you do your homework? I hope so. Let's take a look at my solution to the problem of seeing how conditional probability tells us about whether there's a relationship between age and purchase probability in a fake dataset.

To remind you, what we were trying to do was remove the dependency between age and probability of purchasing and see if we could actually reflect that in our conditional probability values. Here's what I've got:

```
from numpy import random 
random.seed(0) 
 
totals = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0} 
purchases = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0} 
totalPurchases = 0 
for _ in range(100000): 
    ageDecade = random.choice([20, 30, 40, 50, 60, 70]) 
    purchaseProbability = 0.4 
    totals[ageDecade] += 1 
    if (random.random() < purchaseProbability): 
        totalPurchases += 1 
        purchases[ageDecade] += 1 
What I've done here is I've taken the original snippet of code for creating our dictionary of age groups and how much was purchased by each age group for a set of 100,000 random people. Instead of making purchase probability dependent on age, I've made it a constant probability of 40%. Now we just have people randomly being assigned to an age group, and they all have the same probability of buying something. Let's go ahead and run that.


Now this time, if I compute the P(E|F), that is, the probability of buying something given that you're in your 30s, I come up with about 40%.

```
PEF = float(purchases[30]) / float(totals[30]) 
print ("P(purchase | 30s): ", PEF) 
 
P(purchase | 30s):  0.398760454901 
If I compare that to the overall probability of purchasing, that too is about 40%.

```
PE = float(totalPurchases) / 100000.0 
print ("P(Purchase):", PE) 
 
P(Purchase): 0.4003 
```

I can see here that the probability of purchasing something given that you're in your 30s is about the same as the probability of purchasing something irrespective of your age (that is, P(E|F) is pretty close to P(E)). That suggests that there's no real relationship between those two things, and in fact, I know there isn't from this data.

Now in practice, you could just be seeing random chance, so you'd want to look at more than one age group. You'd want to look at more than one data point to see if there really is a relationship or not, but this is an indication that there's no relationship between age and probability of purchase in this sample data that we modified.

So, that's conditional probability in action. Hopefully your solution was fairly close and had similar results. If not, go back and study my solution. It's right there in the data files for this book, ConditionalProbabilitySolution.ipynb, if you need to open it up and study it and play around with it. Obviously, the random nature of the data will make your results a little bit different and will depend on what choice you made for the overall purchase probability, but that's the idea.

And with that behind us, let's move on to Bayes' theorem.

### Bayes' theorem

Now that you understand conditional probability, you can understand how to apply Bayes' theorem, which is based on conditional probability. It's a very important concept, especially if you're going into the medical field, but it is broadly applicable too, and you'll see why in a minute.

You'll hear about this a lot, but not many people really understand what it means or its significance. It can tell you very quantitatively sometimes when people are misleading you with statistics, so let's see how that works.

First, let's talk about Bayes' theorem at a high level. Bayes' theorem is simply this: the probability of A given B is equal to the probability of A times the probability of B given A over the probability of B. So you can substitute A and B with whatever you want.

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-03-02/steps/12/1.png)

**Note:**

The key insight is that the probability of something that depends on B depends very much on the base probability of B and A. People ignore this all the time.

One common example is drug testing. We might say, what's the probability of being an actual user of a drug given that you tested positive for it. The reason Bayes' theorem is important is that it calls out that this very much depends on both the probability of A and the probability of B. The probability of being a drug user given that you tested positive depends very much on the base overall probability of being a drug user and the overall probability of testing positive. The probability of a drug test being accurate depends a lot on the overall probability of being a drug user in the population, not just the accuracy of the test.

It also means that the probability of B given A is not the same thing as the probability of A given B. That is, the probability of being a drug user given that you tested positive can be very different from the probability of testing positive given that you're a drug user. You can see where this is going. That is a very real problem where diagnostic tests in medicine or drug tests yield a lot of false positives. You can still say that the probability of a test detecting a user can be very high, but it doesn't necessarily mean that the probability of being a user given that you tested positive is high. Those are two different things, and Bayes' theorem allows you to quantify that difference.

Let's nail that example home a little bit more.

Again, a drug test can be a common example of applying Bayes' theorem to prove a point. Even a highly accurate drug test can produce more false positives than true positives. So in our example here, we're going to come up with a drug test that can accurately identify users of a drug 99% of the time and accurately has a negative result for 99% of non-users, but only 0.3% of the overall population actually uses the drug in question. So we have a very small probability of actually being a user of a drug. What seems like a very high accuracy of 99% isn't actually high enough, right?

We can work out the math as follows:

- Event A = is a user of the drug
- Event B = tested positively for the drug

So let event A mean that you're a user of some drug, and event B the event that you tested positively for the drug using this drug test.

We need to work out the probability of testing positively overall. We can work that out by taking the sum of probability of testing positive if you are a user and the probability of testing positive if you're not a user. So, P(B) works out to 1.3% (0.99*0.003+0.01*0.997) in this example. So we have a probability of B, the probability of testing positively for the drug overall without knowing anything else about you.

Let's do the math and calculate the probability of being a user of the drug given that you tested positively.

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-03-02/steps/12/2.png)

So the probability of a positive test result given that you're actually a drug user works out as the probability of being a user of the drug overall (P(A)), which is 3% (you know that 3% of the population is a drug user) multiplied by P(B|A) that is the probability of testing positively given that you're a user divided by the probability of testing positively overall which is 1.3%. Again, this test has what sounds like a very high accuracy of 99%. We have 0.3% of the population which uses a drug multiplied by the accuracy of 99% divided by the probability of testing positively overall, which is 1.3%. So the probability of being an actual user of this drug given that you tested positive for it is only 22.8%. So even though this drug test is accurate 99% of the time, it's still providing a false result in most of the cases where you're testing positive.


**Note:**
Even though P(B|A) is high (99%), it doesn't mean P(A|B) is high.

People overlook this all the time, so if there's one lesson to be learned from Bayes' theorem, it is to always take these sorts of things with a grain of salt. Apply Bayes' theorem to these actual problems and you'll often find that what sounds like a high accuracy rate can actually be yielding very misleading results if you're dealing with a low overall incidence of a given problem. We see the same thing in cancer screening and other sorts of medical screening as well. That's a very real problem; there's a lot of people getting very, very real and very unnecessary surgery as a result of not understanding Bayes' theorem. If you're going into the medical profession with big data, please, please, please remember this theorem.

So that's Bayes' theorem. Always remember that the probability of something given something else is not the same thing as the other way around, and it actually depends a lot on the base probabilities of both of those two things that you're measuring. It's a very important thing to keep in mind, and always look at your results with that in mind. Bayes' theorem gives you the tools to quantify that effect. I hope it proves useful.