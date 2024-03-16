

pip install numpy matplotlib scipy arviz

# load the libraries
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

# Data
x=np.array([-0.86,-0.30,-0.05,0.73])
n=np.array([5,5,5,5])
y=np.array([0,1,3,5])

# Logit and logistic functions
def logistic(x):
    return 1/(1+np.exp(-x))

# Log-likelihood function
def log_likelihood(alpha_beta,x,n,y):
    alpha, beta=alpha_beta
    theta=logistic(alpha+beta*x)
    return np.sum(y*np.log(theta)+(n-y)*np.log(1-theta))

# Proposal function
def propose(current):
    proposal=stats.norm(current,0.5).rvs()  #sigma_2=0.5
    return proposal

#Define metropolis algorithm function
def metropolis_function(iterations, burn_in, x, n, y, starting_alpha, starting_beta):
    alpha,beta=starting_alpha,starting_beta  # Use starting values
    chain=np.zeros((iterations, 2))
    accepted=0  # Initialize accepted proposals counter

    for i in range(iterations):
        current_params=np.array([alpha,beta])
        proposed_params=propose(current_params)
        current_ll=log_likelihood(current_params,x,n,y)
        proposed_ll=log_likelihood(proposed_params,x,n,y)

        # Calculate acceptance probability
        A=np.exp(proposed_ll-current_ll)

        # Decide whether to accept the new proposal
        if np.random.rand()<A:
            alpha,beta=proposed_params  # Update parameters to proposed values
            if i>=burn_in:  # Only count acceptance after burn-in
                accepted += 1

        chain[i,:]=[alpha,beta]  # Store current parameters in the chain

    print(f"Acceptance rate (post burn-in): {accepted /(iterations-burn_in):.3f}")
    return chain[burn_in:,:]  # Discard burn-in samples and return the rest

# Define a function to store chains for the multiple random runs
def run_multiple_chains(n_chains,iterations,burn_in,x,n,y):
    chains=[]
    for _ in range(n_chains):
        np.random.seed()  # Reset the seed for each chain for diversity
        starting_alpha=np.random.rand()*2-1  # starting point for alpha
        starting_beta=np.random.rand()*2-1  # starting point for beta
        chain=metropolis_function(iterations + burn_in,burn_in,x,n,y,starting_alpha,starting_beta)
        chains.append(chain)
    return np.array(chains)  # Shape: (n_chains, iterations, 2 for alpha and beta)

# Acceptance Rates for each chain:
n_chains=4
iterations=10000
burn_in=5000
chains=run_multiple_chains(n_chains,iterations,burn_in,x,n,y)

chains.shape

# Extraxt the samples from the posterior samples
posterior = {
    "alpha":chains[:,:,0],  # Extracting alpha samples
    "beta":chains[:,:,1]    # Extracting beta samples
}

# Creating coordinates and dimensions for the InferenceData object
coords={"chain":np.arange(chains.shape[0]),
          "draw":np.arange(chains.shape[1])}
dims={"alpha":["chain","draw"],
        "beta":["chain","draw"]}

# Creating InferenceData object
idata=az.from_dict(posterior=posterior,coords=coords,dims=dims)

#Compute R-hats:
rhat=az.rhat(idata)
print("R-hat values:",rhat)

coords

dims

# Visulaize the chains
fig,axs=plt.subplots(2,1,figsize=(18,10))
for i in range(n_chains):
    axs[0].plot(chains[i,:,0],label=f'Chain {i+1}')
    axs[1].plot(chains[i, :, 1],label=f'Chain {i+1}')

axs[0].set_title('Trace Plot for Alpha')
axs[1].set_title('Trace Plot for Beta')

for ax in axs:
    ax.legend(loc='upper left',bbox_to_anchor=(1, 1))

plt.tight_layout(pad=2.0)
plt.subplots_adjust(right=0.75)
plt.show()

#Scatter Plot of MCMC samples for alpha and beta
combined_chains = chains.reshape(-1, 2) # Combine four chains

combined_chains.shape

# Create scatter plot
plt.figure(figsize=(10, 7))
plt.scatter(combined_chains[:, 0], combined_chains[:, 1], alpha=0.8, s=10)
plt.xlabel('α', fontsize=20)
plt.ylabel('β', fontsize=20)
plt.grid(True)

# Increase tick marks label size
plt.tick_params(axis='both', which='major', labelsize=25)

plt.show()
