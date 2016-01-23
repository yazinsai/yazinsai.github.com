---
layout: post
title:  "First Post! The Types of Machine Learning"
date:   2015-12-22
categories: machine-learning
excerpt: |
  In this first post, I learn the different types of Machine Learning, getting
  an idea in the process of what Machine Learning can and cannot do.
---

Here's what I learned today:

- There are a bunch of different types of Machine Learning.
- A general impression of what can (and can't) be achieved using Machine Learning.

### Types of Machine Learning

There are two main types of Machine Learning:

1. Supervised
2. Unsupervised

You kinda get a general sense for what these two categories might entail, but let's talk about them abit more.

#### 1. Supervised Machine Learning

In supervised learning, we are given a data set and already know what our correct output should look like .. this usually means we have some sort of hunch about the relationship between the input and the output.

The Supervised category is further divided into 2 subcategories:

- **Regression problems**: it's where we are trying to map the inputs to a continuous output .. with infinite possible values. *(Example: trying to predict the price of a stock based on historical performance of that stock)*
- **Classification problems**: this is where the output we're trying to predict has a bunch of possible values, and our job is to figure out which category it falls under. *(Example: given a cute picture of a kittie, trying to figure out which species it belongs to)*

#### 2. Unsupervised Machine Learning

Unsupervised learning, on the other hand, allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results, i.e., there is no teacher to correct you. It’s not just about clustering. For example, associative memory is unsupervised learning.

There are a bunch of categories of unsupervised machine learning:

- **Associative**: Take a collection of 1000 essays written on the US Economy, and find a way to automatically group these essays into a small number that are somehow similar or related by different variables, such as word frequency, sentence length, page count, and so on.
- **Clustering**: Suppose a doctor over years of experience forms associations in his mind between patient characteristics and illnesses that they have. If a new patient shows up then based on this patient’s characteristics such as symptoms, family medical history, physical attributes, mental outlook, etc the doctor associates possible illness or illnesses based on what the doctor has seen before with similar patients. This is not the same as rule based reasoning as in expert systems. In this case we would like to estimate a mapping function from patient characteristics into illnesses.
