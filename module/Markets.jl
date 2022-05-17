#!/usr/bin/env julia

## Markets.jl
##
## Authors: Hiroaki Okabe <roak0225@gmail.com>
##  and Taylor Kessinger <tkess@sas.upenn.edu>
##
## Essential functions and structures for
## marketplace simulations.


module Markets

	using Random, Statistics, Revise

	export evolve!, uniform_market
	export update_market!


	mutable struct SellerHistory
		# structure for storing history of seller attributes
		# future additions could include platform identity/honesty
		strategies::Array{Array{Float64,1}}
		quantities::Array{Array{Float64,1}}
		prices::Array{Array{Float64,1}}
		fitnesses::Array{Array{Float64,1}}
		# status quo, these are all arrays of arrays
		# each element in the array is an array of (seller type-wise) values

	end

	mutable struct Sellers
		# structure for storing the current state of the market's sellers
		qualities::Array{Bool,1} # false for bad sellers, true for good sellers
		platforms::Array{Int64,1} # which platform each seller sells on
		quantities::Array{Float64,2} # how much each seller produces
		prices::Array{Float64,2} # how much each seller charges
		fitnesses::Array{Float64,2} # seller fitnesses
			# indexing is: [seller quality + 1, platform honesty + 1]
			# e.g., fitness[1,2] is the fitness
			# of BAD sellers on HONEST platforms
		n_Sbd::Int64 # number of bad sellers on dishonest platforms
		n_Sbh::Int64 # number of bad sellers on honest platforms
		n_Sgd::Int64 # number of good sellers on dishonest platforms
		n_Sgh::Int64 # number of good sellers on honest platforms
	end

	mutable struct Platforms
		# structure for storing the current state of the market's platforms
		# currently this is barebones: more can be added later when platforms evolve
		honesties::Array{Bool,1} # false for dishonest platforms, true for honest platforms
		fitnesses::Array{Float64,1} # platform fitnesses

	end

	mutable struct Buyers
		# structure for storing the current state of the market's buyers
		# currently this is barebones: more can be added later when buyers evolve
		platforms::Array{Int64,1}  # which platform each buyer buys on
		fitnesses::Array{Float64,1} # buyer fitnesses

		f_Ud::Float64 # fraction of buyers on dishonest platforms
		f_Uh::Float64 # fraction of buyers on honest platforms
	end

	mutable struct Market
		# structure for storing all information about the current sellers,
		# buyers, and platforms, as well as (seller) history

		n_platforms::Int64
		n_sellers::Int64
		n_buyers::Int64

		platforms::Platforms # struct for storing information about platforms
		sellers::Sellers # struct for storing information about sellers
		buyers::Buyers # struct for storing information about buyers

		seller_history::SellerHistory # structure for tracking history
			# of strategy frequencies, prices, etc.

		a_h::Float64 # price premium for good sellers, honest platform
		b_h::Float64 # price premium for bad sellers, honest platform
		a_d::Float64 # price premium for good sellers, dishonest platform
		b_d::Float64 # price premium for bad sellers, dishonest platform

		I::Float64 # price function intercept
		C::Float64 # production cost for good sellers
		c::Float64 # production cost for bad sellers

		sigma::Float64 # strength of selection
		seller_mu::Float64 # mutation rate for sellers

		generation::Int64 # current generation
		track_history::Bool # turn on to track history
		verbose::Bool # turn on for error tracking
	end


	function uniform_market(
		n_platforms::Int64,
		n_sellers::Int64,
		n_buyers::Int64,
		a_h::Float64,
		b_h::Float64,
		a_d::Float64,
		b_d::Float64,
		I::Float64,
		C::Float64,
		c::Float64,
		sigma::Float64,
		seller_mu::Float64,
		track_history::Bool=false,
		verbose::Bool=false
		)
		# initializes a ``uniform'' market: ``uniform'' means
		# buyers are uniformly distributed across platforms,
		# platforms are evenly divided into honest and dishonest,
		# and sellers are uniformly (but randomly) distributed across platforms

		# make every even-numbered platform honest
		platform_honesties = Bool[i%2 for i in 1:n_platforms]
		# initialize Platforms
		platforms = Platforms(platform_honesties, Float64[])

		# evenly distribute the buyers across the platforms
		buyer_platforms = Int64[(i%n_platforms)+1 for i in 1:n_buyers]
		# initialize Buyers
		buyers = Buyers(buyer_platforms, Float64[], 0.0, 0.0)

		seller_qualities = rand(Bool, n_sellers) # randomize seller quantities
		seller_platforms = rand(1:n_platforms, n_sellers) # randomize seller platform membership
		# we will set quantities, prices, etc. to their correct values later
		seller_quantities = zeros(Float64, (2,2))
		seller_prices = zeros(Float64, (2,2))
		seller_fitnesses = zeros(Float64, (2,2))
		# initialize Sellers
		# the zeros are the n_Sbd (etc.) values
		# we will fix them later
		sellers = Sellers(seller_qualities, seller_platforms,
			seller_quantities, seller_prices, seller_fitnesses,
			0, 0, 0, 0)

		# initialize the SellerHistory object
		seller_history = initialize_seller_history()

		# initialize the Market object
		mkt = Market(n_platforms, n_sellers, n_buyers,
			platforms, sellers, buyers,
			seller_history,
			a_h, b_h, a_d, b_d,
			I, C, c,
			sigma, seller_mu, 0,
			track_history, verbose)

		# this updates the relevant f and n values
		update_market!(mkt)

		return mkt

	end

	function update_market!(
		mkt::Market
		)
		# ensures that buyer and seller counts (f_Uh, n_Sbd, etc.)
		# are correctly calculated
		# and updates fitnesses, quantities, etc.
		# this should be run any time a Market is initialized
		# or any time the buyers, sellers, or platforms change
		update_buyer_counts!(mkt)
		update_seller_counts!(mkt)

		update_quantities!(mkt)
		update_prices!(mkt)
		update_seller_fitnesses!(mkt)

		# add the current state of sellers to the history
		if mkt.track_history
			append_seller_history!(mkt)
		end
	end

	function initialize_seller_history()
		# generates an empty SellerHistory object
		return SellerHistory(Array{Float64,1}[],
			Array{Float64,1}[],
			Array{Float64,1}[],
			Array{Float64,1}[])
	end


	function append_seller_history!(
		mkt::Market
		)
		# converts the current state of the sellers (including some 2-D arrays)
		# into 1-D arrays, then appends them to the SellerHistory object
		current_strategies = 1.0*[mkt.sellers.n_Sgh, mkt.sellers.n_Sbh,
			mkt.sellers.n_Sgd, mkt.sellers.n_Sbd]/mkt.n_sellers
		current_quantities = 1.0*[mkt.sellers.quantities[2,2], mkt.sellers.quantities[1,2],
			mkt.sellers.quantities[2,1], mkt.sellers.quantities[1,1]]
		current_prices = 1.0*[mkt.sellers.prices[2,2], mkt.sellers.prices[1,2],
			mkt.sellers.prices[2,1],mkt.sellers.prices[1,1],]
		current_fitnesses = 1.0*[mkt.sellers.fitnesses[2,2], mkt.sellers.fitnesses[1,2],
			mkt.sellers.fitnesses[2,1], mkt.sellers.fitnesses[1,1]]
		push!(mkt.seller_history.strategies, current_strategies)
		push!(mkt.seller_history.quantities, current_quantities)
		push!(mkt.seller_history.prices, current_prices)
		push!(mkt.seller_history.fitnesses, current_fitnesses)

	end

	function evolve!(
		mkt::Market,
		generations::Int64=1
		)
		# applies the selection and mutation functions to the market,
		# then updates its state
		# ``generations'' is an optional variable
		for g in 1:generations

			select_sellers!(mkt)
			mutate_sellers!(mkt)
			update_market!(mkt)

		end

	end

	function update_seller_counts!(
		mkt::Market
		)
		# computes the n values for Sellers
		mkt.sellers.n_Sgh = sum((mkt.platforms.honesties[mkt.sellers.platforms] .== true) .& (mkt.sellers.qualities .== true))
		mkt.sellers.n_Sbh = sum((mkt.platforms.honesties[mkt.sellers.platforms] .== true) .& (mkt.sellers.qualities .== false))
		mkt.sellers.n_Sgd = sum((mkt.platforms.honesties[mkt.sellers.platforms] .== false) .& (mkt.sellers.qualities .== true))
		mkt.sellers.n_Sbd = sum((mkt.platforms.honesties[mkt.sellers.platforms] .== false) .& (mkt.sellers.qualities .== false))
	end

	function update_buyer_counts!(
		mkt::Market
		)
		# computes the f values for Buyers
		mkt.buyers.f_Uh = 1.0*sum(mkt.platforms.honesties[mkt.buyers.platforms] .== true)/mkt.n_buyers
		mkt.buyers.f_Ud = 1.0*sum(mkt.platforms.honesties[mkt.buyers.platforms] .== false)/mkt.n_buyers
	end

	function update_quantities(
		a_h::Float64,
		b_h::Float64,
		a_d::Float64,
		b_d::Float64,
		I::Float64,
		C::Float64,
		c::Float64,
		n_Sbd::Int64,
		n_Sbh::Int64,
		n_Sgd::Int64,
		n_Sgh::Int64,
		f_Ud::Float64,
		f_Uh::Float64
		)
		# updates the quantity of goods produced by each type of seller
		quantities = zeros(Float64,2,2)
		if (f_Uh*((I*a_h - C)*(1 + n_Sbh) - (I*b_h - c)*(n_Sbh)))/(1 + n_Sgh + n_Sbh) >= 0 && (f_Uh*((I*b_h - c)*(1 + n_Sgh) - (I*a_h - C)*(n_Sgh)))/(1 + n_Sgh + n_Sbh) >= 0
			quantities[2,2] = (f_Uh*((I*a_h - C)*(1 + n_Sbh) - (I*b_h - c)*(n_Sbh)))/(1 + n_Sgh + n_Sbh)
			quantities[1,2] = (f_Uh*((I*b_h - c)*(1 + n_Sgh) - (I*a_h - C)*(n_Sgh)))/(1 + n_Sgh + n_Sbh)
		elseif (f_Uh*((I*b_h - c)*(1 + n_Sgh) - (I*a_h - C)*(n_Sgh)))/(1 + n_Sgh + n_Sbh) < 0
			quantities[2,2] = f_Uh*(I*a_h - C)/(1 + n_Sgh)
			quantities[1,2] = 0.0
		else
			quantities[2,2] = 0.0
			quantities[1,2] = f_Uh*(I*b_h - c)/(1 + n_Sbh)
		end

		if (f_Ud*((I*a_d - C)*(1 + n_Sbd) - (I*b_d - c)*(n_Sbd)))/(1 + n_Sgd + n_Sbd) >= 0 && (f_Ud*((I*b_d - c)*(1 + n_Sgd) - (I*a_d - C)*(n_Sgd)))/(1 + n_Sgd + n_Sbd) >= 0
			quantities[2,1] = (f_Ud*((I*a_d - C)*(1 + n_Sbd) - (I*b_d - c)*(n_Sbd)))/(1 + n_Sgd + n_Sbd)
			quantities[1,1] = (f_Ud*((I*b_d - c)*(1 + n_Sgd) - (I*a_d - C)*(n_Sgd)))/(1 + n_Sgd + n_Sbd)
		elseif (f_Ud*((I*b_d - c)*(1 + n_Sgd) - (I*a_d - C)*(n_Sgd)))/(1 + n_Sgd + n_Sbd) < 0
			quantities[2,1] = f_Ud*(I*a_d - C)/(1 + n_Sgd)
			quantities[1,1] = 0.0
		else
			quantities[2,1] = 0.0
			quantities[1,1] = f_Ud*(I*b_d - c)/(1 + n_Sbd)
		end
		return quantities
	end

	function update_quantities!(
		mkt::Market
		)
		# wrapper for update_quantities() above
		# passes Market variables into the above function and
		# updates the Seller quantities
		new_quantities = update_quantities(mkt.a_h, mkt.b_h, mkt.a_d, mkt.b_d,
			mkt.I, mkt.C, mkt.c,
			mkt.sellers.n_Sbd, mkt.sellers.n_Sbh,
			mkt.sellers.n_Sgd, mkt.sellers.n_Sgh,
			mkt.buyers.f_Ud, mkt.buyers.f_Uh)
		mkt.sellers.quantities = new_quantities
	end

	function update_prices!(
		mkt::Market
		)
		# updates the Seller prices
		Q_d = (mkt.sellers.n_Sgd*mkt.sellers.quantities[2,1]+mkt.sellers.n_Sbd*mkt.sellers.quantities[1,1])/mkt.n_sellers
		Q_h = (mkt.sellers.n_Sgh*mkt.sellers.quantities[2,2]+mkt.sellers.n_Sbh*mkt.sellers.quantities[1,2])/mkt.n_sellers
		mkt.sellers.prices[2,2] = mkt.I*mkt.a_h - Q_h/mkt.buyers.f_Uh
		mkt.sellers.prices[1,2] = mkt.I*mkt.b_h - Q_h/mkt.buyers.f_Uh
		mkt.sellers.prices[2,1] = mkt.I*mkt.a_d - Q_d/mkt.buyers.f_Ud
		mkt.sellers.prices[1,1] = mkt.I*mkt.b_d - Q_d/mkt.buyers.f_Ud
	end

	function update_seller_fitnesses!(
		mkt::Market
		)
		# updates the Seller fitnesses
		# mkt.sellers.fitnesses[2,2] = ((mkt.sellers.quantities[2,2])^2)/mkt.buyers.f_Uh
		# mkt.sellers.fitnesses[1,2] = ((mkt.sellers.quantities[1,2])^2)/mkt.buyers.f_Uh
		# mkt.sellers.fitnesses[2,1] = ((mkt.sellers.quantities[2,1])^2)/mkt.buyers.f_Uh
		# mkt.sellers.fitnesses[1,1] = ((mkt.sellers.quantities[1,1])^2)/mkt.buyers.f_Uh
		mkt.sellers.fitnesses[2,2] = (mkt.sellers.prices[2,2]-mkt.C)*mkt.sellers.quantities[2,2]
		mkt.sellers.fitnesses[1,2] = (mkt.sellers.prices[1,2]-mkt.c)*mkt.sellers.quantities[1,2]
		mkt.sellers.fitnesses[2,1] = (mkt.sellers.prices[2,1]-mkt.C)*mkt.sellers.quantities[2,1]
		mkt.sellers.fitnesses[1,1] = (mkt.sellers.prices[1,1]-mkt.c)*mkt.sellers.quantities[1,1]
	end

	function mutate_sellers!(
		mkt::Market
		)
		# checks to see if a seller should be mutated:
		# if yes, picks a random seller and randomizes their platform and quality
		if rand() < mkt.seller_mu
			rand_seller = rand(1:mkt.n_sellers) # pick a random seller
			mkt.sellers.platforms[rand_seller] = rand(1:mkt.n_platforms) # give them a random platform
			mkt.sellers.qualities[rand_seller] = rand(Bool) # give them a random quality
		end
	end

	function select_sellers!(
		mkt::Market
		)
		# picks a random seller, compares its fitness to another seller's,
		# and uses a sigmoid function to decide whether to copy its platform and quality

		# randomly choose sellers to compare
		i_seller = rand(1:mkt.n_sellers)
		j_seller = rand(1:mkt.n_sellers)

		# pull out relevant seller attributes (for readability)
		i_quality = mkt.sellers.qualities[i_seller]
		j_quality = mkt.sellers.qualities[j_seller]
		# below, we check to see which platform i
		# is on, then look up the honesty of that platform
		# then same for j
		i_honesty = mkt.platforms.honesties[mkt.sellers.platforms[i_seller]]
		j_honesty = mkt.platforms.honesties[mkt.sellers.platforms[j_seller]]

		# look up fitnesses
		i_fitness = mkt.sellers.fitnesses[i_quality+1, i_honesty+1]
		j_fitness = mkt.sellers.fitnesses[j_quality+1, j_honesty+1]

		# compute choice function
		choice_function = 1.0/(1.0 + exp(mkt.sigma*(i_fitness - j_fitness)))

		# use choice function to decide whether i_seller copies j_seller
		if rand() < choice_function

			mkt.sellers.platforms[i_seller] = mkt.sellers.platforms[j_seller]
			mkt.sellers.qualities[i_seller] = mkt.sellers.qualities[j_seller]

		end
	end

end
