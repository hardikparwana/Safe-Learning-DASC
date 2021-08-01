using ForwardDiff
using DifferentialEquations
using Plots

function f(x, u)
    return x + u
end

function g(x)
    return 0.1
end

function control(x, k)
    return -k * x
end

function closedLoop(x, k)
    return f(x, control(x, k))
end

function ode_f(state, params, t)
    return closedLoop(state, params[1])
end

x0 = 0.5
tspan = (0.0,1.0)
prob = ODEProblem(ode_f,x0,tspan, [3.0])
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

plot(sol)

function ode_g(state, params, time)
    return g(state)
end

x0 = 0.5
tspan = (0.0,1.0)
prob = SDEProblem(ode_f, ode_g, x0,tspan, [3.0])
sol = solve(prob, EM(), dt=0.01)

plot(sol)

ensembleprob = EnsembleProblem(prob)

sol = solve(ensembleprob,EnsembleThreads(),trajectories=1000)

using DifferentialEquations.EnsembleAnalysis
summ = EnsembleSummary(sol,0:0.01:1)
plot(summ,labels="Middle 95%")
summ = EnsembleSummary(sol,0:0.01:1;quantiles=[0.25,0.75])
plot!(summ,labels="Middle 50%",legend=true)

plot(summ.u)

plot(summ.v)

sol.t[end]

function cost(sol)
    sol.u[end]^2
end

cost(sol)

function cost(k, x0)
    tspan = (0.0,1.0)
    prob = ODEProblem(ode_f,x0,tspan, [k])
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
    cost(sol)
end
    

cost(2.0, 0.5)

cost(4.0, 0.5)

J(k) = cost(k, 0.5)

using ForwardDiff

ForwardDiff.derivative(k -> cost(k, 2.0), 1.0)

function costSDE(k)
    x0 = 0.5
    tspan = (0.0,1.0)
    prob = SDEProblem(ode_f, ode_g, x0,tspan, [k])
#     sol = solve(prob, EM(), dt=0.01)
    ensembleprob = EnsembleProblem(prob)
    sol = solve(ensembleprob,EnsembleThreads(),trajectories=1000)
    summ = EnsembleSummary(sol,0:0.01:1)
    return (summ.u[end])^2
end

cost(1.0, 0.5)

cost(3.0, 0.5)


costSDE(1.0)

costSDE(3.0)

@time ForwardDiff.derivative(costSDE, 1.0)

(2 * costSDE(1.0) - costSDE(1.001) - costSDE(1- 0.001))/(2 * 0.001)

(costSDE(1.1) - costSDE(1.))/(0.1)

(costSDE(1.0) - costSDE(0.9))/(0.1)

@time (costSDE(1.1) - costSDE(0.9))/(0.2)

ks = 1.0:0.01:1.1
outs = costSDE.(ks)


plot(ks, outs)


