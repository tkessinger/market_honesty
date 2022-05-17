#!/usr/bin/env julia

## plt_freqs.jl
##
## Authors: Hiroaki Okabe <roak0225@gmail.com>
##  and Taylor Kessinger <tkess@sas.upenn.edu>
## Plot results from Markets parallel simulations.

using CSV, PyPlot, Statistics
using DataFrames

# load simulation output as a dataframe
runs = CSV.read("output/test.csv",DataFrame)

# get number of generations and trials
n_generations = sort(unique(runs[!,:n_generations]))[1]
n_trials = sort(unique(runs[!,:n_trials]))[1]

# arrays to store frequencies and values
# we add 1 to n_generations
# because the generation 0 frequencies are stored
strategies = zeros(Float64, n_generations+1, 4)
quantities = zeros(Float64, n_generations+1, 4)
prices = zeros(Float64, n_generations+1, 4)
fitnesses = zeros(Float64, n_generations+1, 4)

for (ri, run) in enumerate(eachrow(runs[:,:]))
	for i in 1:4
		# the strategy (etc.) arrays are stored in the .csv file as strings
		# that is, they look something like
		# "[1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]"
		# to convert them "back" to 2-dim arrays, we perform some witchcraft
		# first, we strip the external brackets
		# then, we split them by semicolons, to get an array of values for each generation
		# then, we split that array by whitespace to get individual values
		# we parse those values as floats
		# we then add those arrays, generation-wise, to our master array
		# and divide by the number of trials to normalize
		strategies[:,i] += parse.(Float64,split(split.(runs[ri,:][:strategies][2:end-1],r"; ")[i], r" "))/n_trials
		prices[:,i] += parse.(Float64,split(split.(runs[ri,:][:prices][2:end-1],r"; ")[i], r" "))/n_trials
		fitnesses[:,i] += parse.(Float64,split(split.(runs[ri,:][:fitnesses][2:end-1],r"; ")[i], r" "))/n_trials
		quantities[:,i] += parse.(Float64,split(split.(runs[ri,:][:quantities][2:end-1],r"; ")[i], r" "))/n_trials
	end
end

plotcolors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
strat_label = ["gh", "bh", "gd", "bd"]

# generate a plot with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize = (6,6),
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

plt.savefig("figures/test.pdf")
