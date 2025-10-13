import numpy as np
import random
import warnings
import pandas as pd
import ast

#---------------------------------TASK FUNCTIONS ---------------------------------------
#GEOMETRIC DISTRIBUTION
def generate_geometric_block_duration(x_type, mean_x, N_blocks):
    """"
    Generates a vector of length N_blocks indicating the number of trials in each block.
    - If x_type = "fixed": all blocks have length = mean_x.
    - If x_type = "exp": block durations are drawn from a uniform distribution [20, 55] 
      (instead of geometric).
    - Output: 1D numpy array of integers (length N_blocks)
    """
    # if (N_blocks % 2 == 0):
    # Warning('The number of blocks must be an even number')
    x = np.ndarray(shape=(N_blocks, 1), dtype=int)
    if x_type == "fixed":
        x[0:] = mean_x
    elif x_type == "exp":
        x = np.random.geometric(1 / mean_x, (N_blocks, 1))
        x = np.clip(x, 25, 35)  #amplitude of the blocks
    else:
        Warning('Blocked type not supported')
    return x.flatten()



# RANDOM UNIFORM DISTRIBUTION
def generate_uniform_block_duration(x_type, mean_x, N_blocks):
    """"
    Generates a vector of length N_blocks indicating the number of trials in each block.
    - If x_type = "fixed": all blocks have length = mean_x.
    - If x_type = "exp": block durations are drawn from a uniform distribution [20, 55] (instead of geometric).
    - Output: 1D numpy array of integers (length N_blocks)
    """
    x = np.ndarray(shape=(int(N_blocks), 1), dtype=int)
    #x = np.ndarray(shape=(N_blocks, 1), dtype=int)
    if x_type == "fixed":
        x[0:] = mean_x
    elif x_type == "exp":
        mean_x = None 
        x = np.random.uniform(20, 50, (int(N_blocks), 1)).astype(int)
    else:
        Warning('Blocked type not supported')
    return x.flatten()


def generate_block_probs_vec(N_blocks, prob_block_type, p_list, prob_Left_Right_blocks):
    """
    Generates the reward probabilities (pR) for each block.
    - Divides blocks into half "Right" and half "Left".
    - If `prob_block_type` is 'rdm_values':
         - Randomly samples pR from p_list for Right blocks.
         - Left block pR is 1-pR (if 'balanced') or independently sampled (if 'indep').
    - If `prob_block_type` is 'permutation_prob_list':
         - Uses permuted versions of p_list across blocks.
    - Returns a 1D numpy array of length N_blocks with pR values.
    """
    N_blocks = int(N_blocks)
    if (N_blocks % 2) == 1:
        N_blocks += 1
        warnings.warn('We increased the number of blocks by one to have an even number')
        print('N_blocks = ', N_blocks)

    N_probs = len(p_list)
    print('N_probs = ', N_probs)
    
    # Allocate memory for the probs_vec array x
    output_probs_vec = np.ndarray(int(N_blocks), dtype=float)
    N_blocks_by_half = int(N_blocks / 2)
    print('N_blocks_by_half = ', N_blocks_by_half)

    prob_list_Right_blocks = np.zeros(N_blocks_by_half, dtype=float)
    prob_list_Left_blocks = np.zeros(N_blocks_by_half, dtype=float)

    if prob_block_type == 'rdm_values':
        # Generate Right blocks 
        prob_list_Right_blocks = np.random.choice(p_list, N_blocks_by_half) 
        # Generate Left blocks 
        if prob_Left_Right_blocks == 'indep':
            prob_list_Left_blocks = 1. - np.squeeze(np.random.choice(p_list, N_blocks_by_half))
        elif prob_Left_Right_blocks == 'balanced':
            prob_list_Left_blocks = 1. - np.random.permutation(prob_list_Right_blocks)
        else:
            warnings.warn('Specify the relation between Left and Right probs as balanced or indep')
            return None  # Exit the function if input is invalid

    elif prob_block_type == 'permutation_prob_list':
        times_rep_per = N_blocks_by_half // N_probs
        print('times_rep_per = ', times_rep_per)
        for i in range(times_rep_per):
            per_Right_probs = np.random.permutation(p_list)
            per_Left_probs = 1. - np.random.permutation(p_list)
            prob_list_Right_blocks[i * N_probs:(i + 1) * N_probs] = per_Right_probs
            prob_list_Left_blocks[i * N_probs:(i + 1) * N_probs] = per_Left_probs
        
        remainder = N_blocks_by_half % N_probs
        if remainder > 0:
            per_Right_probs = np.random.permutation(p_list)
            per_Left_probs = 1. - np.random.permutation(p_list)
            prob_list_Right_blocks[-remainder:] = per_Right_probs[:remainder]
            prob_list_Left_blocks[-remainder:] = per_Left_probs[:remainder]

    else:
        warnings.warn('Specify the way to take prob values from prob_list: rdm_values or permutation_prob_list')
        return None  # Exit the function if input is invalid

    # Assign Right and Left probs to the output vector depending on the order chosen by rdm_right_block_order
    rdm_right_block_order = np.random.permutation([0, 1])
    output_probs_vec[rdm_right_block_order[0]::2] = prob_list_Right_blocks
    output_probs_vec[rdm_right_block_order[1]::2] = prob_list_Left_blocks

    return output_probs_vec        



def generate_block_reward_side_vec(N_blocks, block_duration_vec, probs_vec):
    """
    Generates a 2D array of size (sum(block_duration_vec), 3):
    - Column 0: reward probability used in that trial
    - Column 1: binary reward side (1 = right, 0 = left), sampled using Binomial(pR)
    - Column 2: block index
    Each block has consistent pR value and duration, and trial values are sampled per block.
    """

    N_blocks = int(N_blocks)

    # N_x = block_duration_vec.sum()
    # x = np.ndarray(shape= (N_x, 1), dtype = int)
    x = np.zeros([1, 3])

    #print(x)

    for i_block in range(N_blocks):
        # we convert the array entry  block_duration_vec[i_block] into a scalar int or it will complain: "TypeError: only integer scalar arrays can be converted to a scalar index"
        bloc_dur = np.take(block_duration_vec[i_block], 0)

        # column vector with the p_value of the block
        p = probs_vec[i_block] * np.ones((bloc_dur, 1), dtype=float)

        # column vector with the rewarded sides of the block block_duration_vec[i_block]
        y = np.random.binomial(1, probs_vec[i_block], (bloc_dur, 1))

        # paste the two columns together
        blocks = np.repeat(i_block, bloc_dur).reshape(-1,1)

        Z = np.concatenate((p, y, blocks), axis=1)

        # conncateate the two-column vector of the block with previous blocks


        x = np.concatenate((x, Z), axis=0)

        # remove the first row of x which contains zeros
    x = np.delete(x, 0, axis=0)
    #print("x: " + str(x))
    return x

        
# funtion to generate the truncated exponential distribution for the ITIs
def generate_trial_values(lambda_param, max_value, num_values):
    """
    Generates one truncated exponential value (mean 1/λ) up to max_value.
    Repeats until the value is ≤ max_value.
    Returns: a list of valid ITI durations for one trial.
    """
    trial_values = []
    for _ in range(num_values):
        while True:
            value = random.expovariate(lambda_param)
            if value <= max_value:
                trial_values.append(value)
                break
    return trial_values


# function to obtain the values
def custom_random_iti(num_trials, num_values_per_trial, lambda_param):
    """
    Generates a full list of ITI durations for an entire session.
    - For each trial, calls `generate_trial_values()` with lambda=0.5 and max=30s.
    Returns: a flat list of ITI durations (one per trial).
    """
    num_trials = int(num_trials)
    max_value = 30  # max value
    all_values = []
    for _ in range(num_trials):
        trial_values = generate_trial_values(lambda_param, max_value, num_values_per_trial)
        all_values.extend(trial_values)
    return all_values


    
# ----------------------------------PLOTTING FUNCTIONS----------------------------------------

def assign_ports(df: pd.DataFrame) -> pd.DataFrame:
    """Assign left/right poke ports based on system_name."""
    system_name = df['system_name'].iloc[0]

    if system_name == 9:
        df['left_poke_in'] = df['Port2In']
        df['left_poke_out'] = df['Port2Out']
        df['right_poke_in'] = df['Port5In']
        df['right_poke_out'] = df['Port5Out']
    elif system_name == 11:
        df['left_poke_in'] = df['Port2In']
        df['left_poke_out'] = df['Port2Out']
        df['right_poke_in'] = df['Port5In']
        df['right_poke_out'] = df['Port5Out']
    elif system_name == 12:
        df['left_poke_in'] = df['Port7In']
        df['left_poke_out'] = df['Port7Out']
        df['right_poke_in'] = df['Port1In']
        df['right_poke_out'] = df['Port1Out']
    elif system_name == 12:
        df['left_poke_in'] = df['Port3In']
        df['left_poke_out'] = df['Port3Out']
        df['right_poke_in'] = df['Port1In']
        df['right_poke_out'] = df['Port1Out']
    else:
        raise ValueError(f"Unsupported system_name: {system_name}")
    
    return df

def extract_first_float(val):
            if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                try:
                    return float(ast.literal_eval(val)[0])
                except Exception:
                    return None
            try:
                return float(val)  # fallback se il valore è già un numero in stringa
            except:
                return None
    
