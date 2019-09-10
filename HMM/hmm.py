from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(X_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    # TODO
    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array delta[i, t] = P(X_t = s_i, Z_1:Z_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        
        # Base Case
        output = self.obs_dict[Osequence[0]]
        for s in range(S):
            alpha[s][0] = self.pi[s] * self.B[s][output]

        # Recursion
        for t in range(1, L):
            output = self.obs_dict[Osequence[t]]
            for s in range(S):
                sum = 0
                for sp in range(S):
                    sum = sum + self.A[sp][s] * alpha[sp][t-1]
                alpha[s][t] = self.B[s][output] * sum
        
        return alpha

    # TODO:
    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array gamma[i, t] = P(Z_t+1:Z_T | X_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        
        # Base Case
        for s in range(S):
            beta[s][L-1] = 1

        # Recursion
        for tr in range(0, L-1):
            t = L-2-tr
            output = self.obs_dict[Osequence[t+1]]
            for s in range(S):
                sum = 0
                for sp in range(S):
                    sum = sum + self.A[s][sp] * self.B[sp][output] * beta[sp][t+1]
                beta[s][t] = sum

        return beta

    # TODO:
    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(Z_1:Z_T | λ)
        """
        prob = 0
        
        alpha = self.forward(Osequence)
        S = len(self.pi)
        T = len(Osequence)
        for s in range(S):
            prob = prob + alpha[s][T-1]
            
        return prob

    # TODO:
    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(X_t = i | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])

        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        deno = 0
        for s in range(S):
            deno = deno + alpha[s][L-1]

        for t in range(L):
            for s in range(S):
                prob[s][t] = (alpha[s][t] * beta[s][t])/deno

        return prob

    # TODO:
    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        
        S = len(self.pi)
        L = len(Osequence)
        path = [0]*L
        delta = np.zeros([S, L])
        big_delta = np.zeros([S, L])

        # Base case
        output = self.obs_dict[Osequence[0]]
        for s in range(S):
            delta[s][0] = self.pi[s] * self.B[s][output]

        # Recursion
        for t in range(1, L):
            output = self.obs_dict[Osequence[t]]
            for s in range(S):
                max = 0
                argmax = 0
                for sp in range(S):
                    ad = self.A[sp][s] * delta[sp][t-1]
                    if max < ad:
                        max = ad
                        argmax = sp
                delta[s][t] = self.B[s][output] * max
                big_delta[s][t] = argmax

        # Backtracking
        # Base Case
        max = 0
        argmax = 0
        for s in range(S):
            if max < delta[s][L-1]:
                max = delta[s][L-1]
                argmax = s
        path[L-1] = argmax

        # Recursion
        for tr in range(0, L-1):
            t = L-2-tr
            path[t] = int(big_delta[int(path[t+1])][t+1])

        # States from index
        key_list = list(self.state_dict.keys())
        val_list = list(self.state_dict.values())
        for t in range(L):
            val = path[t]
            path[t] = key_list[val_list.index(val)]
            
        return path
