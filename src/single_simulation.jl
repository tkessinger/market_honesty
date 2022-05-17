#!/usr/bin/env julia

## single_simulation.jl
##
## Authors: Hiroaki Okabe <roak0225@gmail.com>
##  and Taylor Kessinger <tkess@sas.upenn.edu>
##
## Single implementation of Markets.
## Initializes a market with platforms, buyers, and sellers,
## allows sellers to update strategy, price, and quantity,
## and plots trajectories.

using PyPlot
using Revise
using Markets

n_platforms = 2 # number of platforms
n_sellers = 100 # number of sellers
n_buyers = 10 # number of buyers

n_generations = 100000

# price premiums
a_h = 1.0
b_h = 0.9
a_d = 0.9
b_d = 1.0

sigma = 1.0 # strength of selection
seller_mu = 0.01 # seller mutation rate

I = 1.0 # price function intercept
C = 0.05 # production cost, good sellers
c = 0.025 # production cost, bad sellers

track_history = true
verbose = false

# initialize the market
mkt = uniform_market(n_platforms, n_sellers, n_buyers,
	a_h, b_h, a_d, b_d, I, C, c,
	sigma, seller_mu, track_history, verbose)

# evolve it (and update its state automatically)
evolve!(mkt, n_generations)

# pull out the attributes of the SellerHistory object
strategies, quantities, prices, fitnesses = mkt.seller_history.strategies,
	mkt.seller_history.quantities, mkt.seller_history.prices,
	mkt.seller_history.fitnesses

# recast them as 2-D arrays for easier plotting
strategies = permutedims(hcat(strategies...))
quantities = permutedims(hcat(quantities...))
prices = permutedims(hcat(prices...))
fitnesses = permutedims(hcat(fitnesses...))

plotcolors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
strat_label = ["gh", "bh", "gd", "bd"]

fig, axs = plt.subplots(2, 2, figsize = (8,8),
	sharex="col")

[axs[1].plot(0:n_generations, strategies[:,i], label=strat_label[i],c=plotcolors[i]) for i in 1:4]
[axs[2].plot(0:n_generations, prices[:,i], label=strat_label[i],c=plotcolors[i]) for i in 1:4]
[axs[3].plot(0:n_generations, fitnesses[:,i], label=strat_label[i],c=plotcolors[i]) for i in 1:4]
[axs[4].plot(0:n_generations, quantities[:,i], label=strat_label[i],c=plotcolors[i]) for i in 1:4]

axs[2].set_xlabel("time")
axs[4].set_xlabel("time")

axs[1].set_ylabel("strategy frequency")

axs[2].set_ylabel("price")

axs[3].set_ylabel("fitness")
axs[3].yaxis.set_label_position("right")
axs[3].yaxis.tick_right()

axs[4].set_ylabel("quantity")
axs[4].yaxis.set_label_position("right")
axs[4].yaxis.tick_right()

axs[1].legend(loc=1)

plt.tight_layout()
display(fig)
