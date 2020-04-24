using Pkg
using Revise # lets you change A2funcs without restarting julia!
includet("A2_src.jl")
using Plots
using LinearAlgebra
using Statistics: mean
using Zygote
using Test
using Distributions
using Logging
using .A2funcs: log1pexp # log(1 + exp(x)) stable
using .A2funcs: factorized_gaussian_log_density
using .A2funcs: skillcontour!
using .A2funcs: plot_line_equal_skill!

#Q1(a)
function log_prior(zs)
  logprob=factorized_gaussian_log_density(0, 0, zs)
  return logprob
end;

#Q1(b)
function logp_a_beats_b(za,zb)
  return -log1pexp.(zb .- za)
end;

#Q1(c)
function all_games_log_likelihood(zs,games)
  zs_a = zs[games[:,1],:]
  zs_b = zs[games[:,2],:]
  loglike_a_beats_b = logp_a_beats_b(zs_a,zs_b)
  likelihoods =  sum.(eachcol(loglike_a_beats_b))
  return  vec(likelihoods)'
end;

#Q1(d)
function joint_log_density(zs,games)
  prior = log_prior(zs)
  likelihoods = all_games_log_likelihood(zs,games)
  return prior + likelihoods
end;

test_zs = randn(4,15)
factorized_gaussian_log_density(0,0,test_zs)
test_games = [1 2; 3 1; 4 2]
joint_log_density(test_zs, test_games)
@testset "Test shapes of batches for likelihoods" begin
  B = 15 # number of elements in batch
  N = 4 # Total Number of Players
  test_zs = randn(4,15)
  test_games = [1 2; 3 1; 4 2] # 1 beat 2, 3 beat 1, 4 beat 2
  @test size(test_zs) == (N,B)
  #batch of priors
  @test size(log_prior(test_zs)) == (1,B)
  # loglikelihood of p1 beat p2 for first sample in batch
  @test size(logp_a_beats_b(test_zs[1,1],test_zs[2,1])) == ()
  # loglikelihood of p1 beat p2 broadcasted over whole batch
  @test size(logp_a_beats_b.(test_zs[1,:],test_zs[2,:])) == (B,)
  # batch loglikelihood for evidence
  @test size(all_games_log_likelihood(test_zs,test_games)) == (1,B)
  # batch loglikelihood under joint of evidence and prior
  @test size(joint_log_density(test_zs,test_games)) == (1,B)
end

ENV["GRDIR"]=""
Pkg.build("GR")

# Convenience function for producing toy games between two players.
two_player_toy_games(p1_wins, p2_wins) = vcat([repeat([1,2]',p1_wins), repeat([2,1]',p2_wins)]...)

# Example for how to use contour plotting code
plot(title="Example Gaussian Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
example_gaussian(zs) = exp(factorized_gaussian_log_density([-1.,2.],[0.,0.5],zs))
skillcontour!(example_gaussian)
plot_line_equal_skill!()
# savefig(joinpath("plots","example_gaussian.pdf"))

#Q2(a)
plot_log_prior(zs) = exp.(log_prior(zs));
plot(title="Skill Prior Q2(a)",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
skillcontour!(plot_log_prior)
plot_line_equal_skill!()


#Q2(b)
plot_likelihood(zs) = exp.(logp_a_beats_b(zs[1],zs[2]));
plot(title="Likelihood Q2(b)",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
skillcontour!(plot_likelihood)
plot_line_equal_skill!()


#Q2(c)
# player A wins 1 match
games=two_player_toy_games(1,0);
plot_joint_likelihood(zs) = exp.(joint_log_density(zs,games));
plot(title="Joint Likelihood Q2(c)",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
skillcontour!(plot_joint_likelihood)
plot_line_equal_skill!()

# Q2(d)
# plot joint contours with player A winning 10 games
games=two_player_toy_games(10,0);
plot_joint_likelihood(zs) = exp.(joint_log_density(zs,games));
plot(title="Joint Likelihood Q2(d)",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
skillcontour!(plot_joint_likelihood)
plot_line_equal_skill!()

# Q2(e)
# plot joint contours with player A winning 10 games and player B winning 10 games
games=two_player_toy_games(10,10);
plot_joint_likelihood(zs) = exp.(joint_log_density(zs,games));
plot(title="Joint Likelihood Q2(e)",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
skillcontour!(plot_joint_likelihood)
plot_line_equal_skill!()


# Q3(a)
function elbo(params,logp,num_samples)
  mu = params[1]
  logσ = params[2]
  N = size(mu)[1]
  a = Diagonal(exp.(logσ)) * randn(N, num_samples)
  b = reshape(mu, N, 1) * ones(1, num_samples)
  samples = a .+ mu
  logp_estimate = logp(samples)
  logq_estimate = factorized_gaussian_log_density(mu, logσ, samples)
  return mean(logp_estimate .- logq_estimate)
end;

# Q3(b)
# Conveinence function for taking gradients
function neg_toy_elbo(params; games = two_player_toy_games(1,0), num_samples = 100)
  logp(zs) = joint_log_density(zs,games)
  return -elbo(params,logp, num_samples)
end;

# Q3(c)
# Toy game
num_players_toy = 2;
toy_mu = [-2.,3.]; # Initial mu, can initialize randomly!
toy_ls = [0.5,0.]; # Initual log_sigma, can initialize randomly!
toy_params_init = (toy_mu, toy_ls);

# Q3(c)
function fit_toy_variational_dist(init_params, toy_evidence; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params
  for i in 1:num_itrs
    loss(params) = neg_toy_elbo(params; games=toy_evidence, num_samples=num_q_samples)
    # gradients of variational objective with respect to parameters
    grad_params = gradient(loss, params_cur)[1]
    mu_grad = grad_params[1]
    logσ_grad = grad_params[2]
    mu = params_cur[1] - lr .* mu_grad
    logσ = params_cur[2] - lr .* logσ_grad
    params_cur = (mu, logσ)
    println(loss(params_cur))
  end

  p(zs) = exp.(joint_log_density(zs, games));
  q(zs) = exp.(factorized_gaussian_log_density(params_cur[1], params_cur[2], zs));
  plot(title="Compare p & q",
      xlabel = "Player 1 Skill",
      ylabel = "Player 2 Skill"
     )
  display(plot_line_equal_skill!())
  display(skillcontour!(p; colour=:red))
  display(skillcontour!(q; colour=:blue))
  return params_cur
end;


# Q3(d)
# fit q with SVI observing player A winning 1 game
games = two_player_toy_games(1,0);
params_1 = fit_toy_variational_dist(toy_params_init, games);
# savefig("Q3_d.pgn")

# Q3(e)
# fit q with SVI observing player A winning 10 games
games = two_player_toy_games(10,0);
params_1 = fit_toy_variational_dist(toy_params_init, games);
# savefig("Q3_e.pgn")

# Q3(f)
# fit q with SVI observing player A winning 10 games and player B winning 10 games
games = two_player_toy_games(10,10);
params_1 = fit_toy_variational_dist(toy_params_init, games);
# savefig("Q3_f.pgn")

## Question 4
# Load the Data
using MAT
vars = matread("tennis_data.mat");
player_names = vars["W"];
tennis_games = Int.(vars["G"]);
num_players = length(player_names);
print("Loaded data for $num_players players")

# Q4(a)
# Yes

# Q4(b)
function fit_variational_dist(init_params, tennis_games; num_itrs=500, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params
  for i in 1:num_itrs
    loss(params) = neg_toy_elbo(params; games=tennis_games, num_samples=num_q_samples)
    grad_params = gradient(loss, params_cur)[1]
    mu_grad = grad_params[1]
    logσ_grad = grad_params[2]
    mu = params_cur[1] - lr .* mu_grad
    logσ = params_cur[2] - lr .* logσ_grad
    params_cur = (mu, logσ)
    println(loss(params_cur))
  end
  return params_cur
end;


# Initialize variational family
init_mu = rand(-2:3, size(player_names)[1]);
init_log_sigma = rand(107);
init_params = (init_mu, init_log_sigma);

# Train variational distribution
trained_params = fit_variational_dist(init_params, tennis_games);

# Q4(c)
skill_μ = trained_params[1];
skill_σ = trained_params[2];
perm = sortperm(skill_μ);
plot(title="All player skill",
    xlabel = "Player No.",
    ylabel = "Skill Mean",
    legend = false,
   )
plot!(skill_μ[perm], yerror=exp.(skill_σ[perm]))

# Q4(d)
top_10 = reverse(player_names[perm])[1:10];
print(top_10)

# Q4(e)
sorted_μ = reverse(skill_μ[perm]);
sorted_σ = reverse(skill_σ[perm]);
μ1 = sorted_μ[2];
μ2 = sorted_μ[3];
logσ1 = sorted_σ[2];
logσ2 = sorted_σ[3];
plot_μ = [μ1, μ2];
plot_logσ = [logσ1, logσ2];

# find index of Roger-Federer and Rafael-Nadal
f1(x) = x=="Roger-Federer";
car1 = findall(f1, player_names);
f2(x) = x=="Rafael-Nadal";
car2 = findall(f2, player_names);
L = LinearIndices(player_names);
ind1 = L[car1][1];
ind2 = L[car2][1];

f3(x) = (x[1] == ind1) & (x[2] == ind2);
f3_ind = f3.(eachrow(tennis_games));
f3_result = tennis_games[f3_ind, :];

f4(x) = (x[1] == ind2) & (x[2] == ind1);
f4_ind = f4.(eachrow(tennis_games));
f4_result = tennis_games[f4_ind, :];

games = vcat(f3_result, f4_result);
plot_joint(zs) = exp.(factorized_gaussian_log_density(plot_μ, plot_logσ, zs));
plot(title="Roger vs Rafael",
    xlabel = "Roger skill",
    ylabel = "Rafael skill",
)
skillcontour!(plot_joint)
plot_line_equal_skill!()


# Q4(g)
mu_g = μ1 - μ2;
sigma_g = exp(logσ1);
dist_g = Normal(mu_g, sigma_g);
prob_exact_g = 1- cdf(dist_g, 0)

sample_g = randn(10000) .* sigma_g .+ mu_g;
f5(x) = x .> 0;
prob_mc_g = mean(f5(sample_g))

# Q4(h)
skill_μ = trained_params[1];
skill_σ = trained_params[2];
perm = sortperm(skill_μ);
sorted_μ = reverse(skill_μ[perm]);
sorted_σ = reverse(skill_σ[perm]);

μ107 = sorted_μ[107];
mu_h = μ1 - μ107;
sigma_h = exp(logσ1);
dist_h = Normal(mu_h, sigma_h);
prob_exact_h = 1- cdf(dist_h, 0)

sample_h = randn(10000) .* sigma_h .+ mu_h;
prob_mc_h = mean(f5(sample_h))

# Q4(i)
