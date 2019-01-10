# simple impleentation of "canonical" particle swarm optimisation algorithms


struct PSO <: Algorithm
	solver::PSOSolver
	num_points::Int
	fitnessFunction::Func
	discountRate::AbstractFloat
	learningRate::AbstractFloat

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
PSO(f,n,d,l) = PSO(PSOSolver(), n,f,d,l)

function solve!(pso::PSO)
	for p in pso.solver.points
		r1 = rand()
		r2 = rand()
		p.velocity = (pso.d * p.velocity) + pso.l * (r1 * (p.bestFitness - p.currentFitness) + r2 * (pso.solver.globalBestFitness - p.currentFitness))
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





