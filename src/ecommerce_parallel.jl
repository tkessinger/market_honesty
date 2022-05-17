#!/usr/bin/env julia

## ecommerce_parallel.jl
##
## Authors: Hiroaki Okabe <roak0225@gmail.com>
##  and Taylor Kessinger <tkess@sas.upenn.edu>
## Parallelized implementation of Markets.
## Initializes a market with platforms, buyers, and sellers,
## allows sellers to update strategy, price, and quantity,
## and records trajectories.

using Random, Statistics
using ArgParse
using Distributed
using Revise
using CSV
using Dates
using DataFrames
import JSON

function read_parameters(
	defpars::Dict{String,Any},
    inputfile = nothing
	)
	# read and parse JSON file
	# to pass parameters to a worker

    pars = copy(defpars)

    # read JSON file
    if inputfile != nothing
        inpars = JSON.parsefile(inputfile)
    else
        inpars = Dict()
    end

    for parkey in keys(defpars)
        if "type" in keys(pars[parkey])
            if isprimitivetype(pars[parkey]["type"]) ||
                pars[parkey]["type"] == String
                T = pars[parkey]["type"]
            end
        else
            # default type is Float64
            T = Float64
        end
        #println(parkey, T)
        if T <: Int
            convertf = (val)->round(T, val)
        else
            convertf = (val)->convert(T, val)
        end

        # use defpars for list of usable parameters in JSON
        if parkey in keys(inpars)
            if "value" in keys(inpars[parkey])
                val = inpars[parkey]["value"]
            elseif "range" in keys(inpars[parkey])
                valr = inpars[parkey]["range"]
                if "log" in keys(valr)
                    b = valr["log"]
                    rf = (r)->b.^r
                    pop!(valr, "log")
                else
                    rf = (r)->r
                end
                start = pop!(valr, "start")
                rkws = Dict(zip(Symbol.(keys(valr)), values(valr)))
                val = rf(range(start; rkws...))
            end
        else
            val = pars[parkey]["value"]
        end

        if !isstructtype(typeof(val)) || typeof(val) == String || typeof(val) == Bool
            pars[parkey] = [convertf(val)]
        else
            pars[parkey] = convertf.(val)
        end
    end

    return pars
end

function main(args)
	# main simulation function:
	# parse parameters, take their Cartesian product,
	# run the actual simulation,
	# and record output

    s = ArgParseSettings(description =
        "run ReputationSets simulations across multiple cores")
    @add_arg_table! s begin
        "--ncpus"
            arg_type = Int64
            default = max(round(Int, Sys.CPU_THREADS), 1)
        "--input"
            default = nothing
        #"--output"
        #    default=nothing
    end
    parsed_args = parse_args(args, s)


    defpars = Dict{String,Any}([
	    "n_platforms"     => Dict("value" => 2, "type" => Int64),
	    "n_sellers"     => Dict("value" => 2, "type" => Int64),
	    "n_buyers"     => Dict("value" => 10, "type" => Int64),
	    "n_trials" => Dict("value" => 10, "type" => Int64),
	  	"n_generations" => Dict("value" => 10000, "type" => Int64),
		"a_h" => Dict("value" => 1.0, "type" => Float64),
		"b_h" => Dict("value" => 0.9, "type" => Float64),
		"a_d" => Dict("value" => 0.9, "type" => Float64),
		"b_d" => Dict("value" => 1.0, "type" => Float64),
		"sigma" => Dict("value" => 1.0, "type" => Float64),
		"seller_mu" => Dict("value" => 0.01, "type" => Float64),
		"I" => Dict("value" => 1.0, "type" => Float64),
		"C" => Dict("value" => 0.05, "type" => Float64),
		"c" => Dict("value" => 0.025, "type" => Float64),
	    "output" => Dict("value" => "output/test.csv", "type" => String)
    ])
    pars = read_parameters(defpars, parsed_args["input"])

    # take the Cartesian product of all parameter combinations
    parsets = collect(Base.product(values(pars)...))
    nsets = length(parsets)

    # setup workers assuming directory is manually added to LOAD_PATH
    addprocs(min(parsed_args["ncpus"], round(Int64, Sys.CPU_THREADS / 2)))
    wpool = WorkerPool(workers())
    #extradir = filter((p)->match(r"/", p) !== nothing, LOAD_PATH)[1]
    extradir = filter((p)->match(r"/", p) !== nothing, LOAD_PATH)
    #@everywhere workers() push!(LOAD_PATH, $extradir)
    [@everywhere workers() push!(LOAD_PATH, $x) for x in extradir]
    @everywhere workers() eval(:(using Random))
    @everywhere workers() eval(:(using Statistics))
    @everywhere workers() eval(:(using Dates))

    inputs  = RemoteChannel(()->Channel{Dict}(2 * nsets * maximum(pars["n_trials"])))
    results = RemoteChannel(()->Channel{Dict}(2 * nsets * maximum(pars["n_trials"])))

    @everywhere function run_worker(inputs, results)
        # save trial number and random seed
        seed = Dict(zip(["seed1", "seed2", "seed3", "seed4"], Random.MersenneTwister().seed))

        while true
            pard = take!(inputs)
            pard = merge(pard, seed)

		    n_platforms = pard["n_platforms"]
		    n_sellers = pard["n_sellers"]
		    n_buyers = pard["n_buyers"]
		    n_generations = pard["n_generations"]
		    a_h = pard["a_h"]
		    b_h = pard["b_h"]
		    a_d = pard["a_d"]
		    b_d = pard["b_d"]
		    sigma = pard["sigma"]
		    seller_mu = pard["seller_mu"]
		    I = pard["I"]
		    C = pard["C"]
		    c = pard["c"]

			# name of output file
            output = pard["output"]

            println("--- running ", pard["nrun"], " --- ")
            flush(stdout)


			marketplaces = []
			seller_marketplaces = []
			seller_quality = []
			buyer_marketplaces = []

			marketplaces = [true, false]

			seller_marketplaces = ones(Int64, n_sellers)
			for i in 1:n_sellers
				if rand() > 0.5
					seller_marketplaces[i] = 2
				end
			end

			seller_quality = zeros(Bool, n_sellers)
			for i in 1:n_sellers
				if rand() > 0.5
					seller_quality[i] = true
				end
			end


			buyer_marketplaces = ones(Int64,n_buyers)
			for i in 1:n_buyers
				if i%2 == 0
					buyer_marketplaces[i] = 2
				end
			end

			n_Sgh = sum((marketplaces[seller_marketplaces] .== true) .& (seller_quality .== true))
			n_Sbh = sum((marketplaces[seller_marketplaces] .== true) .& (seller_quality .== false))
			n_Sgd = sum((marketplaces[seller_marketplaces] .== false) .& (seller_quality .== true))
			n_Sbd = sum((marketplaces[seller_marketplaces] .== false) .& (seller_quality .== false))

			f_Uh = 1.0*sum(marketplaces[buyer_marketplaces] .== true)/n_buyers
			f_Ud = 1.0*sum(marketplaces[buyer_marketplaces] .== false)/n_buyers

			seller_quantities = zeros(Float64, 2, 2)
			seller_prices = zeros(Float64, 2, 2)
			seller_fitnesses = zeros(Float64, 2, 2)

			if (f_Uh*((I*a_h - C)*(1 + n_Sbh) - (I*b_h - c)*(n_Sbh)))/(1 + n_Sgh + n_Sbh) >= 0 && (f_Uh*((I*b_h - c)*(1 + n_Sgh) - (I*a_h - C)*(n_Sgh)))/(1 + n_Sgh + n_Sbh) >= 0
				seller_quantities[2,2] = (f_Uh*((I*a_h - C)*(1 + n_Sbh) - (I*b_h - c)*(n_Sbh)))/(1 + n_Sgh + n_Sbh)
				seller_quantities[1,2] = (f_Uh*((I*b_h - c)*(1 + n_Sgh) - (I*a_h - C)*(n_Sgh)))/(1 + n_Sgh + n_Sbh)
			elseif (f_Uh*((I*b_h - c)*(1 + n_Sgh) - (I*a_h - C)*(n_Sgh)))/(1 + n_Sgh + n_Sbh) < 0
				seller_quantities[2,2] = f_Uh*(I*a_h - C)/(1 + n_Sgh)
				seller_quantities[1,2] = 0
			else
				seller_quantities[2,2] = 0
				seller_quantities[1,2] = f_Uh*(I*b_h - c)/(1 + n_Sbh)
			end

			if (f_Ud*((I*a_d - C)*(1 + n_Sbd) - (I*b_d - c)*(n_Sbd)))/(1 + n_Sgd + n_Sbd) >= 0 && (f_Ud*((I*b_d - c)*(1 + n_Sgd) - (I*a_d - C)*(n_Sgd)))/(1 + n_Sgd + n_Sbd) >= 0
				seller_quantities[2,1] = (f_Ud*((I*a_d - C)*(1 + n_Sbd) - (I*b_d - c)*(n_Sbd)))/(1 + n_Sgd + n_Sbd)
				seller_quantities[1,1] = (f_Ud*((I*b_d - c)*(1 + n_Sgd) - (I*a_d - C)*(n_Sgd)))/(1 + n_Sgd + n_Sbd)
			elseif (f_Ud*((I*b_d - c)*(1 + n_Sgd) - (I*a_d - C)*(n_Sgd)))/(1 + n_Sgd + n_Sbd) < 0
				seller_quantities[2,1] = f_Ud*(I*a_d - C)/(1 + n_Sgd)
				seller_quantities[1,1] = 0
			else
				seller_quantities[2,1] = 0
				seller_quantities[1,1] = f_Ud*(I*b_d - c)/(1 + n_Sbd)
			end

			seller_prices[2,2] = I*a_h - (seller_quantities[2,2]+seller_quantities[1,2])/f_Uh
			seller_prices[1,2] = I*b_h - (seller_quantities[2,2]+seller_quantities[1,2])/f_Uh
			seller_prices[2,1] = I*a_d - (seller_quantities[2,1]+seller_quantities[1,1])/f_Uh
			seller_prices[1,1] = I*b_d - (seller_quantities[2,1]+seller_quantities[1,1])/f_Uh

			seller_fitnesses[2,2] = ((seller_quantities[2,2])^2)/f_Uh
			seller_fitnesses[1,2] = ((seller_quantities[1,2])^2)/f_Uh
			seller_fitnesses[2,1] = ((seller_quantities[2,1])^2)/f_Uh
			seller_fitnesses[1,1] = ((seller_quantities[1,1])^2)/f_Uh

			strategies = []
			quantities = []
			prices = []
			fitnesses = []

			current_strategies = 1.0*[n_Sgh, n_Sbh, n_Sgd, n_Sbd]/n_sellers
			current_quantities = 1.0*[seller_quantities[2,2], seller_quantities[1,2], seller_quantities[2,1], seller_quantities[1,1]]
			current_prices = 1.0*[seller_prices[2,2], seller_prices[1,2],seller_prices[2,1],seller_prices[1,1],]
			current_fitnesses = 1.0*[seller_fitnesses[2,2], seller_fitnesses[1,2], seller_fitnesses[2,1], seller_fitnesses[1,1]]
			push!(strategies, current_strategies)
			push!(quantities, current_quantities)
			push!(prices, current_prices)
			push!(fitnesses, current_fitnesses)

			for k in 1:n_generations
				i_seller = rand(1:n_sellers)
				j_seller = rand(1:n_sellers)
				i_fitness = seller_fitnesses[seller_quality[i_seller]+1, marketplaces[seller_marketplaces[i_seller]]+1]
				j_fitness = seller_fitnesses[seller_quality[j_seller]+1, marketplaces[seller_marketplaces[j_seller]]+1]



				choice_function = 1.0/(1.0 + exp(sigma*(i_fitness - j_fitness)))

				if rand() < choice_function

					seller_marketplaces[i_seller] = seller_marketplaces[j_seller]
					seller_quality[i_seller] = seller_quality[j_seller]

				end
				if rand() < seller_mu
					rand_seller = rand(1:n_sellers)
					seller_marketplaces[rand_seller] = rand(1:n_platforms)
					seller_quality[rand_seller] = rand(Bool)

				end
				n_Sgh = sum((marketplaces[seller_marketplaces] .== true) .& (seller_quality .== true))
				n_Sbh = sum((marketplaces[seller_marketplaces] .== true) .& (seller_quality .== false))
				n_Sgd = sum((marketplaces[seller_marketplaces] .== false) .& (seller_quality .== true))
				n_Sbd = sum((marketplaces[seller_marketplaces] .== false) .& (seller_quality .== false))

				f_Uh = 1.0*sum(marketplaces[buyer_marketplaces] .== true)/n_buyers
				f_Ud = 1.0*sum(marketplaces[buyer_marketplaces] .== false)/n_buyers

				if (f_Uh*((I*a_h - C)*(1 + n_Sbh) - (I*b_h - c)*(n_Sbh)))/(1 + n_Sgh + n_Sbh) >= 0 && (f_Uh*((I*b_h - c)*(1 + n_Sgh) - (I*a_h - C)*(n_Sgh)))/(1 + n_Sgh + n_Sbh) >= 0
					seller_quantities[2,2] = (f_Uh*((I*a_h - C)*(1 + n_Sbh) - (I*b_h - c)*(n_Sbh)))/(1 + n_Sgh + n_Sbh)
					seller_quantities[1,2] = (f_Uh*((I*b_h - c)*(1 + n_Sgh) - (I*a_h - C)*(n_Sgh)))/(1 + n_Sgh + n_Sbh)
				elseif (f_Uh*((I*b_h - c)*(1 + n_Sgh) - (I*a_h - C)*(n_Sgh)))/(1 + n_Sgh + n_Sbh) < 0
					seller_quantities[2,2] = f_Uh*(I*a_h - C)/(1 + n_Sgh)
					seller_quantities[1,2] = 0
				else
					seller_quantities[2,2] = 0
					seller_quantities[1,2] = f_Uh*(I*b_h - c)/(1 + n_Sbh)
				end

				if (f_Ud*((I*a_d - C)*(1 + n_Sbd) - (I*b_d - c)*(n_Sbd)))/(1 + n_Sgd + n_Sbd) >= 0 && (f_Ud*((I*b_d - c)*(1 + n_Sgd) - (I*a_d - C)*(n_Sgd)))/(1 + n_Sgd + n_Sbd) >= 0
					seller_quantities[2,1] = (f_Ud*((I*a_d - C)*(1 + n_Sbd) - (I*b_d - c)*(n_Sbd)))/(1 + n_Sgd + n_Sbd)
					seller_quantities[1,1] = (f_Ud*((I*b_d - c)*(1 + n_Sgd) - (I*a_d - C)*(n_Sgd)))/(1 + n_Sgd + n_Sbd)
				elseif (f_Ud*((I*b_d - c)*(1 + n_Sgd) - (I*a_d - C)*(n_Sgd)))/(1 + n_Sgd + n_Sbd) < 0
					seller_quantities[2,1] = f_Ud*(I*a_d - C)/(1 + n_Sgd)
					seller_quantities[1,1] = 0
				else
					seller_quantities[2,1] = 0
					seller_quantities[1,1] = f_Ud*(I*b_d - c)/(1 + n_Sbd)
				end

			    seller_prices[2,2] = I*a_h - (seller_quantities[2,2]+seller_quantities[1,2])/f_Uh
			    seller_prices[1,2] = I*b_h - (seller_quantities[2,2]+seller_quantities[1,2])/f_Uh
			    seller_prices[2,1] = I*a_d - (seller_quantities[2,1]+seller_quantities[1,1])/f_Uh
			    seller_prices[1,1] = I*b_d - (seller_quantities[2,1]+seller_quantities[1,1])/f_Uh

			    seller_fitnesses[2,2] = ((seller_quantities[2,2])^2)/f_Uh
			    seller_fitnesses[1,2] = ((seller_quantities[1,2])^2)/f_Uh
			    seller_fitnesses[2,1] = ((seller_quantities[2,1])^2)/f_Uh
			    seller_fitnesses[1,1] = ((seller_quantities[1,1])^2)/f_Uh

				current_strategies = 1.0*[n_Sgh, n_Sbh, n_Sgd, n_Sbd]/n_sellers
				current_quantities = 1.0*[seller_quantities[2,2], seller_quantities[1,2], seller_quantities[2,1], seller_quantities[1,1]]
				current_prices = 1.0*[seller_prices[2,2], seller_prices[1,2],seller_prices[2,1],seller_prices[1,1],]
				current_fitnesses = 1.0*[seller_fitnesses[2,2], seller_fitnesses[1,2], seller_fitnesses[2,1], seller_fitnesses[1,1]]
				push!(strategies, current_strategies)
				push!(quantities, current_quantities)
				push!(prices, current_prices)
				push!(fitnesses, current_fitnesses)
			end



			strategies = hcat(strategies...)
			quantities = hcat(quantities...)
			prices = hcat(prices...)
			fitnesses = hcat(fitnesses...)


			pard["strategies"] = strategies
			pard["quantities"] = quantities
			pard["prices"] = prices
			pard["fitnesses"] = fitnesses



            # return data to master process
            put!(results, pard)
        end
    end

    total_time_start = now()

    # load parameter sets into inputs channel
    nruns = 0
    for parset in parsets
        pard = Dict(zip(keys(pars), parset))
        #println(pard)
        println("--- queueing --- ")
        foreach(k->print(k, ": ", pard[k], ", "), sort(collect(keys(pard))))
        println()
        flush(stdout)
        for rep in 1:pard["n_trials"]
            nruns += 1
            rpard = copy(pard)
            rpard["rep"] = rep
            rpard["nrun"] = nruns
            put!(inputs, rpard)
        end
    end

    # start workers running on parameter sets in inputs
    for w in workers() # start tasks on the workers to process requests in parallel
        remote_do(run_worker, w, inputs, results)
    end

    # create output file name and data table
    output = pars["output"][1]
    println(output)
    file = occursin(r"\.csv$", output) ? output : output * ".csv"
    cols = push!(sort(collect(keys(pars))),
                 ["rep", "strategies", "quantities", "prices", "fitnesses", "seed1", "seed2", "seed3", "seed4"]...)
    dat = DataFrame(Dict([(c, Any[]) for c in cols]))

    # grab results and output to CSV
    for sim in 1:nruns
        # get results from parallel jobs
        flush(stdout)
        resd = take!(results)
        nrun = pop!(resd, "nrun")

        # add to table (must convert dict keys to symbols) and save
        push!(dat, Dict([(Symbol(k), resd[k]) for k in keys(resd)]))
        CSV.write(file, dat)
    end
    total_time_stop = now()
    total_time = Dates.canonicalize(Dates.CompoundPeriod(round(total_time_stop - total_time_start, Dates.Second(1))))
    println("total time elapsed: $total_time")
end

# specify input file here
main(["--input", "submit/ecommerce_parallel.json"])
