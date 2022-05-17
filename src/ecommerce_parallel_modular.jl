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
        println(parkey, T)
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
        "run Reputation_sets simulations across multiple cores")
    @add_arg_table! s begin
        "--n_cpus"
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
    n_sets = length(parsets)

    # setup workers assuming directory is manually added to LOAD_PATH
    addprocs(min(parsed_args["n_cpus"], round(Int64, Sys.CPU_THREADS / 2)))
    wpool = WorkerPool(workers())
    #extradir = filter((p)->match(r"/", p) !== nothing, LOAD_PATH)[1]
    extradir = filter((p)->match(r"/", p) !== nothing, LOAD_PATH)
	println(extradir)
    #@everywhere workers() push!(LOAD_PATH, $extradir)
    [@everywhere workers() push!(LOAD_PATH, $x) for x in extradir]
    @everywhere workers() eval(:(using Random))
    @everywhere workers() eval(:(using Statistics))
	@everywhere workers() eval(:(using Markets))
    @everywhere workers() eval(:(using Dates))

    inputs  = RemoteChannel(()->Channel{Dict}(2 * n_sets * maximum(pars["n_trials"])))
    results = RemoteChannel(()->Channel{Dict}(2 * n_sets * maximum(pars["n_trials"])))

    @everywhere function run_worker(inputs, results)
        # save trial number and random seed
        seed = Dict(zip(["seed1", "seed2", "seed3", "seed4"], Random.MersenneTwister().seed))

        while true
            pard = take!(inputs)
            pard = merge(pard, seed)

			# take the input parameters
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

			verbose = false
			track_history = true

            println("--- running ", pard["nrun"], " --- ")
            flush(stdout)

			# initialize the Market object
			mkt = uniform_market(n_platforms, n_sellers, n_buyers,
				a_h, b_h, a_d, b_h, I, C, c,
				sigma, seller_mu, track_history, verbose)

			# evolve the market and track its history
			evolve!(mkt, n_generations)

			# pull out the SellerHistory attributes
			strategies, quantities, prices, fitnesses = mkt.seller_history.strategies,
				mkt.seller_history.quantities, mkt.seller_history.prices,
				mkt.seller_history.fitnesses

			# recast the attributes as 2-D arrays
			# note: we do NOT permute dimensions because that
			# makes it rather onerous to parse it for plotting
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
        println(pard)
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
