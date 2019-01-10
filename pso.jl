# simple impleentation of "canonical" particle swarm optimisation algorithms

abstract struct Algorithm end

struct PSO <: Algorithm
	solver::PSOSolver
	num_points::Int
	fitnessFunction::Func
	discountRate::AbstractFloat
	a1::AbstractFloat
	a2::AbstractFloat

end

mutable struct PSOSolver
	points::Array{PSOPoint}
	globalBestFitness::AbstractFloat
end

mutable struct PSOPoint {T}
	coordinates::AbstractArray{T}
	velocity::AbstractArray{T}
	currentFitness::AbstractFloat
	bestFitness::AbstractFloat	
end


PSOPoint(d) = PSOPoint([rand() for i in d], [rand() for i in d], 0,0)
PSOSolver() = PSOSolver([PSOPoint() for i in num_points], 0)
PSO(f,n=50,d=0.7,a1=4, a2=4) = PSO(PSOSolver(), n,f,d,a1,a2)

function solve!(pso::PSO)
	for p in pso.solver.points
		r1 = rand()
		r2 = rand()
		p.velocity = (pso.d * p.velocity) + (pso.a1 * r1 * (p.bestFitness - p.currentFitness) + (pso.a2 * r2 * (pso.solver.globalBestFitness - p.currentFitness))
		p.coordinates += p.velocity# update position 
		p.currentFitness = pso.fitnessFunction(p.coordinates)
		if p.currentFitness > p.bestFitness
			p.bestFitness = p.currentFitness
			# check if better than globalBestFitness
			if p.bestFitness > pso.solver.globalBestFitness
				pso.solver.globalBestFitness = p.bestFitness
			end
		end
	end
	return pso
end




# how to improve this algorithm... first off, you want to make sure there is stuff sufficient exploration - i.e. ignoring premture convergence
# so some possibilities are to a.) whenever swarm is converging explode them again to start anew all over search space...
# I wonder how long this can be proven to work for as its essentially n operators on the search space
# it depends how big the basins of attractoin are right?
# but hopefully they would increase in higher dimensions
# b.) require some level of covariance -i.e. if the mean position of the swarm is known, then pushing the swarm away from the mean with smoe parameter would work
# this is obviously highly parralelisable insofar as it can run on separate machines with very little computatoin required
# i.e. fundamentally all you need to store is 1 number - the global optimium and this can be refreshed regularly but if it is slightly stale
# this should not matter so much
# secondarily you could keep the mean of the swarm and push the items away from that
# also regular swarm dispersals... perhaps calculate variance of swarm if they semi-regularly send their current positions
# andexlpode them again if getting too irregular...
# other possibilities is keeping track of interesting state spaceregions explored... I'm not totally sure how you do this
# but it would be really cool to think about how to actually get it figured out, so who knows?

