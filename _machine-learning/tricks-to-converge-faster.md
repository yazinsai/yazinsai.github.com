---
layout: post
title:  "Tricks to make Gradient Descent converge faster"
date:   2016-01-21 22:10:10 +0400
excerpt: |
  Here, we talk about a couple of tricks that you can use to optimize your gradient descent .. and make it run with fewer iterations, hence reducing the number of calculation cycles and doing your bit to save the planet.
---

### Trick #1: Normalize your data

You know how you were always trying to fit in as a kid? Well, it turns out that's good advice for our features to heed.

It can be proved that if your features have greatly varying ranges, then it would almost certainly take longer for gradient descent to find the local minima.

For example, let's take the features below:

\\(x_1\\) = Size of the bedroom (0 - 2000 ft\\(^2\\))

\\(x_2\\) = Number of bedrooms (1 - 5 rooms)

The ideal feature would fit within the range \\(-1 \le x \le +1\\)

Just keep in mind that this is an approximate range ( \\(-2 \le x \le 3 \\) is fine too .. but \\(-1,000 \le x \le 250\\) is **not**)

Knowing what we know now, let's normalize \\(x_1, x_2\\) to fit within that range:

$$x_1 = \frac{\text{Size of bedroom}}{2000}$$

$$x_2 = \frac{\text{Number of bedrooms}}{5}$$

### Trick #2: Picking a learning rate, \\(\alpha\\)

Plot # of iters vs. \\(J(\theta)\\)

Run this plot for these values of \\(\alpha\\) .. 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1
