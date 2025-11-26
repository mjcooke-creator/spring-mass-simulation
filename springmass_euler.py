# Author: MJ Cooke
# Date: 10/17/2025
# 
# Following FastLabTutorial's tutorial on getting started with 6DOF simulators
# starting with 2 degrees, a spring mass damper example
# hoping to use this as a jumping off to get into flight simulations
# and more complex physical sims


import numpy as np
import matplotlib.pyplot as plt

def derivatives(state_inp, cont):
    """
    Take the derivative of the initialized states
    using a mathematical model for a spring damper

    Parameters
    ---------
    state_input : array
        input of the current states of the simulation
    cont : float
        the control value for the current state

    Returns
    -------
    state_deriv : array
        The array of calculated derivatives of the 
        current states
    """
    m = 1.0 #mass
    c = 2.0 #damping coefficient
    k = 3.0 #spring constant
    A = np.asarray([[0.0, 1.0], [-k/m, -c/m]]) # 2 x 2 array
    B = np.asarray([0.0, 1.0/m]) # 2 x 1
    #the model equations given in the tutorial
    state_deriv = np.dot(A, state_inp) + B*cont 
    return state_deriv

def control(state_inp, t):
    """
    Calculate the control value for the current states
    at the current time, based off of a PD controller

    Parameters
    ----------
    state_inp : array
        The current states of the model
    t : flloat
        The current time in the simulation

    Returns
    -------
    ucontrol : float
        returns the value of the control factor
        for the current state and time
    """
    kp = 30.0 # factor to adjust control based off current error
    # larger value, larger correction - too large can overshoot
    kd = 10.0 # factor based on the rate of change of the error (the damping effect)
    xcommand = 0.5
    xderivcommand = 0.0
    ucontrol = 0.0
    if t > 5:
        ucontrol = -kp * (state_inp[0]-xcommand) - kd * (state_inp[1]-xderivcommand)
    return ucontrol

def main():
    print("\n== Spring Mass Damper Program ==\n")
    state = np.asarray([1.0, -2.0]) 

    t_initial = 0.0
    t_final = 20.0
    t_step = 0.01
    time = np.arange(t_initial, t_final+t_step, t_step)

    state_out = np.zeros((2,len(time))) 

    for idx in range(len(time)):
        #print(f"Simulation {time[idx]/t_final * 100:.2f} Percent Complete")
        
        # implementing Euler's method 
        state_out[:,idx] = state
        ucontrol = control(state, time[idx])
        state_derivatives = derivatives(state, ucontrol)
        state += state_derivatives * t_step


    # plotting
    position = state_out[0,:]
    velocity = state_out[1,:]
    plt.figure()
    # position vs. time
    plt.subplot(121)
    plt.title("Euler's Method Version")
    plt.plot(time, position)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.grid()
    # velocity vs. time
    plt.subplot(122)
    plt.plot(time, velocity)
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.grid()
    plt.show()

main()