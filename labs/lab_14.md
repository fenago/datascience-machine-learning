<img align="right" src="../images/logo-small.png">


Lab : Apache Spark - Machine Learning on Big Data - Part 2
-------------------------------------


In this lab, we'll cover following topics:

- Working with Spark
- Decision Trees in Spark
- K-Means Clustering in Spark

### Terminal

Now, move in the directory which contains the source code.

`cd ~/work/datascience-machine-learning`

**Note:**
- The supplied commands in the next steps MUST be run from your `datascience-machine-learning` directory. 
- There should be terminal opened already. You can also open New terminal by Clicking `File` > `New` > `Terminal` from the top menu.
- To copy and paste: use **Control-C** and to paste inside of a terminal, use **Control-V**




### Searching wikipedia with Spark MLlib

We're going to build an actual working search algorithm for a piece of Wikipedia using Apache Spark in MLlib, and we're going to do it all in less than 50 lines of code. This might be the coolest thing we do in this entire book!

Go into your course materials and open up the TF-IDF.py script, and that should open up Canopy with the following code:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-09-02/steps/22/1.png)

Now, step back for a moment and let it sink in that we're actually creating a working search algorithm, along with a few examples of using it in less than 50 lines of code here, and it's scalable. I could run this on a cluster. It's kind of amazing. Let's step through the code.

## Import statements

We're going to start by importing the SparkConf and SparkContext libraries that we need for any Spark script that we run in Python, and then we're going to import HashingTF and IDF using the following commands.

```
from pyspark import SparkConf, SparkContext 
from pyspark.mllib.feature import HashingTF 
from pyspark.mllib.feature import IDF 
```

So, this is what computes the term frequencies (TF) and inverse document frequencies (IDF) within our documents.

#### Creating the initial RDD
We'll start off with our boilerplate Spark stuff that creates a local SparkConfiguration and a SparkContext, from which we can then create our initial RDD.

```
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF") 
sc = SparkContext(conf = conf) 
```

Next, we're going to use our SparkContext to create an RDD from subset-small.tsv.

```
rawData = sc.textFile("./subset-small.tsv") 
```


That gives me back an RDD where every document is in each line of the RDD. The tsv file contains one entire Wikipedia document on every line, and I know that each one of those documents is split up into tabular fields that have various bits of metadata about each article.


The next thing I'm going to do is split those up:

```
fields = rawData.map(lambda x: x.split("\t")) 
```

I'm going to split up each document based on their tab delimiters into a Python list, and create a new fields RDD that, instead of raw input data, now contains Python lists of each field in that input data.

Finally, I'm going to map that data, take in each list of fields, extract field number three x[3], which I happen to know is the body of the article itself, the actual article text, and I'm in turn going to split that based on spaces:

```
documents = fields.map(lambda x: x[3].split(" ")) 
```

What x[3] does is extract the body of the text from each Wikipedia article, and split it up into a list of words. My new documents RDD has one entry for every document, and every entry in that RDD contains a list of words that appear in that document. Now, we actually know what to call these documents later on when we're evaluating the results.

I'm also going to create a new RDD that stores the document names:

```
documentNames = fields.map(lambda x: x[1]) 
```

All that does is take that same fields RDD and uses this map function to extract the document name, which I happen to know is in field number one.

So, I now have two RDDs, documents, which contains lists of words that appear in each document, and documentNames, which contains the name of each document. I also know that these are in the same order, so I can actually combine these together later on to look up the name for a given document.

### Creating and transforming a HashingTF object

Now, the magic happens. The first thing we're going to do is create a HashingTF object, and we're going to pass in a parameter of 100,000. This means that I'm going to hash every word into one of 100,000 numerical values:

```
hashingTF = HashingTF(100000)  
```

Instead of representing words internally as strings, which is very inefficient, it's going to try to, as evenly as possible, distribute each word to a unique hash value. I'm giving it up to 100,000 hash values to choose from. Basically, this is mapping words to numbers at the end of the day.

Next, I'm going to call transform on hashingTF with my actual RDD of documents:

```
tf = hashingTF.transform(documents) 
```

That's going to take my list of words in every document and convert it to a list of hash values, a list of numbers that represent each word instead.

This is actually represented as a sparse vector at this point to save even more space. So, not only have we converted all of our words to numbers, but we've also stripped out any missing data. In the event that a word does not appear in a document where you're not storing the fact that word does not appear explicitly, it saves even more space.

#### Computing the TF-IDF score
To actually compute the TF-IDF score for each word in each document, we first cache this tf RDD.

```
tf.cache() 
```

We do that because we're going to use it more than once. Next, we use IDF(minDocFreq=2), meaning that we're going to ignore any word that doesn't appear at least twice:

```
idf = IDF(minDocFreq=2).fit(tf) 
```

We call fit on tf, and then in the next line we call transform on tf:

```
tfidf = idf.transform(tf) 
```

What we end up with here is an RDD of the TF-IDF score for each word in each document.

### Using the Wikipedia search engine algorithm

Let's try and put the algorithm to use. Let's try to look up the best article for the word Gettysburg. If you're not familiar with US history, that's where Abraham Lincoln gave a famous speech. So, we can transform the word Gettysburg into its hash value using the following code:

```
gettysburgTF = hashingTF.transform(["Gettysburg"]) 
gettysburgHashValue = int(gettysburgTF.indices[0]) 
```

We will then extract the TF-IDF score for that hash value into a new RDD for each document:

```
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])  
```

What this does is extract the TF-IDF score for Gettysburg, from the hash value it maps to for every document, and stores that in this gettysburgRelevance RDD.

We then combine that with the documentNames so we can see the results:

```
zippedResults = gettysburgRelevance.zip(documentNames)  
```

Finally, we can print out the answer:

```
print ("Best document for Gettysburg is:") 
print (zippedResults.max()) 
```

### Running the algorithm

So, let's go run that and see what happens. As usual, to run the Spark script, we're not going to just hit the play icon. We have to go to Tools>Canopy Command Prompt. In the Command Prompt that opens up, we will type in spark-submit TF-IDF.py, and off it goes.

#### Run Code
Now, run the python code by running: `spark-submit TF-IDF.py`

**Note:** We can also run python code by running:

```
python TF-IDF.py
```

We are asking it to chunk through quite a bit of data, even though it's a small sample of Wikipedia it's still a fair chunk of information, so it might take a while. Let's see what comes back for the best document match for Gettysburg, what document has the highest TF-IDF score?

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-09-02/steps/22/2.png)

It's Abraham Lincoln! Isn't that awesome? We just made an actual search engine that actually works, in just a few lines of code.

And there you have it, an actual working search algorithm for a little piece of Wikipedia using Spark in MLlib and TF-IDF. And the beauty is we can actually scale that up to all of Wikipedia if we wanted to, if we had a cluster large enough to run it.


#### Implementing linear regression

In this example, what we're going to do is implement linear regression, and linear regression is just a way of fitting a line to a set of data. What we're going to do in this exercise is take a bunch of fabricated data that we have in two dimensions, and try to fit a line to it with a linear model.

We're going to separate our data into two sets, one for building the model and one for evaluating the model, and we'll compare how well this linear model does at actually predicting real values. First of all, in Spark 2, if you're going to be doing stuff with the SparkSQL interface and using Datasets, you've got to be using a SparkSession object instead of a SparkContext. To set one up, you do the following:

```
spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("LinearRegression").getOrCreate() 
```

**Note:**

**Note:** that the middle bit is only necessary on Windows and in Spark 2.0. It kind of works around a little bug that they have, to be honest. So, if you're on Windows, make sure you have a C:/temp folder. If you want to run this, go create that now if you need to. If you're not on Windows, you can delete that whole middle section to leave: spark = SparkSession.builder.appName("LinearRegression").getOrCreate().

Okay, so you can say spark, give it an appName and getOrCreate().

This is interesting, because once you've created a Spark session, if it terminates unexpectedly, you can actually recover from that the next time that you run it. So, if we have a checkpoint directory, it can actually restart where it left off using getOrCreate.

### Using the Spark 2.0 DataFrame API for MLlib


Now, we're going to use this regression.txt file that I have included with the course materials:

```
inputLines = spark.sparkContext.textFile("regression.txt")  
```

That is just a text file that has comma-delimited values of two columns, and they're just two columns of, more or less randomly, linearly correlated data. It can represent whatever you want. Let's imagine that it represents heights and weights, for example. So, the first column might represent heights, the second column might represent weights.

**Note:**

In the lingo of machine learning, we talk about labels and features, where labels are usually the thing that you're trying to predict, and features are a set of known attributes of the data that you use to make a prediction from.


In this example, maybe heights are the labels and the features are the weights. Maybe we're trying to predict heights based on your weight. It can be anything, it doesn't matter. This is all normalized down to data between -1 and 1. There's no real meaning to the scale of the data anywhere, you can pretend it means anything you want, really.

To use this with MLlib, we need to transform our data into the format it expects:

```
data = inputLines.map(lambda x: x.split(",")).map(lambda x: (float(x[0]), Vectors.dense(float(x[1])))) 
```

The first thing we're going to do is split that data up with this map function that just splits each line into two distinct values in a list, and then we're going to map that to the format that MLlib expects. That's going to be a floating point label, and then a dense vector of the feature data.

In this case, we only have one bit of feature data, the weight, so we have a vector that just has one thing in it, but even if it's just one thing, the MLlib linear regression model requires a dense vector there. This is like a labeledPoint in the older API, but we have to do it the hard way here.

Next, we need to actually assign names to those columns. Here's the syntax for doing that:

```
colNames = ["label", "features"] 
df = data.toDF(colNames) 
```

We're going to tell MLlib that these two columns in the resulting RDD actually correspond to the label and the features, and then I can convert that RDD to a DataFrame object. At this point, I have an actual dataframe or, if you will, a Dataset that contains two columns, label and features, where the label is a floating point height, and the features column is a dense vector of floating point weights. That is the format required by MLlib, and MLlib can be pretty picky about this stuff, so it's important that you pay attention to these formats.


Now, like I said, we're going to split our data in half.

```
trainTest = df.randomSplit([0.5, 0.5]) 
trainingDF = trainTest[0] 
testDF = trainTest[1] 
```

We're going to do a 50/50 split between training data and test data. This returns back two dataframes, one that I'm going to use to actually create my model, and one that I'm going to use to evaluate my model.

I will next create my actual linear regression model with a few standard parameters here that I've set.

```
lir = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
``` 

We're going to call lir = LinearRegression, and then I will fit that model to the set of data that I held aside for training, the training data frame:

```
model = lir.fit(trainingDF) 
```

That gives me back a model that I can use to make predictions from.

Let's go ahead and do that.

```
fullPredictions = model.transform(testDF).cache() 
```

I will call model.transform(testDF), and what that's going to do is predict the heights based on the weights in my testing Dataset. I actually have the known labels, the actual, correct heights, and this is going to add a new column to that dataframe called predictions, that has the predicted values based on that linear model.


I'm going to cache those results, and now I can just extract them and compare them together. So, let's pull out the prediction column, just using select like you would in SQL, and then I'm going to actually transform that dataframe and pull out the RDD from it, and use that to map it to just a plain old RDD full of floating point heights in this case:

```
predictions = fullPredictions.select("prediction").rdd.map(lambda x: x[0]) 
```

These are the predicted heights. Next, we're going to get the actual heights from the label column:

```
labels = fullPredictions.select("label").rdd.map(lambda x: x[0]) 
```

Finally, we can zip them back together and just print them out side by side and see how well it does:

```
predictionAndLabel = predictions.zip(labels).collect() 
 
for prediction in predictionAndLabel: 
    print(prediction) 
 
spark.stop() 
```

#### Run Code
Now, run the python code by running: `spark-submit SparkLinearRegression.py`

There's a little bit more upfront time to actually run these APIs with Datasets, but once they get going, they're very fast. Alright, there you have it.

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-09-02/steps/26/2.png)

Here we have our actual and predicted values side by side, and you can see that they're not too bad. They tend to be more or less in the same ballpark. There you have it, a linear regression model in action using Spark 2.0, using the new dataframe-based API for MLlib. More and more, you'll be using these APIs going forward with MLlib in Spark, so make sure you opt for these when you can. Alright, that's MLlib in Spark, a way of actually distributing massive computing tasks across an entire cluster for doing machine learning on big Datasets. So, good skill to have. Let's move on.
