
# Author: MJ Cooke
# Date: 10/18/2025
# 
# Building off of FastLabTutorial's guidelines for a spring mass damper simulation
# but implementing Runge-Kutta fourth order integration instead of Euler's method
# for greater accuracy, and to compare the use of both methods


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
    #the model given in the tutorial
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
    kd = 10.0 # factor based on the rate of change of the error (damping effect)
    xcommand = 0.5
    xderivcommand = 0.0
    ucontrol = 0.0
    if t > 5:
        ucontrol = -kp * (state_inp[0]-xcommand) - kd * (state_inp[1]-xderivcommand)
    return ucontrol

def main():

    print("\n== Spring Mass Damper Pogram ==\n")
    state = np.asarray([1.0, -2.0]) # initial position, initial velocity

    t_initial, t_final, t_step = 0.0, 20.0, 0.01
    time = np.arange(t_initial, t_final+t_step, t_step)

    state_out = np.zeros((2,len(time))) #2 = system dimension

    # Rnge-Kutta method
    for idx in range(len(time)):
        #print(f"Simulation {time[idx]/t_final * 100:.2f} Percent Complete")
        t = time[idx]
        state_out[:,idx] = state
        
        #RK4 coefficients
        u1 = control(state, t)
        k1 = derivatives(state, u1)

        u2 = control(state + 0.5 * t_step * k1, t + 0.5 * t_step)
        k2 = derivatives(state + 0.5 * t_step * k1, u2)

        u3 = control(state + 0.5 * t_step * k2, t + 0.5 * t_step)
        k3 = derivatives(state + 0.5 * t_step * k2, u3)

        u4 = control(state + t_step * k3, t + t_step)
        k4 = derivatives(state + t_step * k3, u4)

        #calculating the next state via RK4
        state += (t_step / 6) * (k1 + (2*k2) + (2*k3) + k4)


    # plotting everything
    position = state_out[0,:]
    velocity = state_out[1,:]
    plt.figure()
    # position vs. time
    plt.subplot(121)
    plt.title("RK4 Integration Version")
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