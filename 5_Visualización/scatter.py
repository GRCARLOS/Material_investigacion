
import numpy as np
import matplotlib.pyplot as plt
import sys
#import pandas as pd
import time

# **Simulación sistema Controlador-Motor** Practica V
## Este codigo esta encamido a poder ejecutar el algoritmo de RKDP al tiempo que se grafican las respuestas del 
## sistema bajo estudio, para tales efectos se hace uso de plt.scatter() o plt.plot()
## Observaciones: Ambos metodos, resultaron en tiempos menores al de amimation, 15 y 5 seg aprox.

start=time.perf_counter()

def model_motor(x, u, Tl):
                       # u -> control input (voltaje)
                       #Tl -> external torque (Definir unidades ?)
    R=0.343            # Impedance
    L = 0.00018        # Inductance
    kb=0.0167          # contraelectromotriz constant
    Jm=2.42*10**-6     # Motor inertia
    kt=0.0167          # Torque constant
    B=5.589458*10**-6

    current=x[0]       # Current
    theta=x[1]         # Angular postion
    theta_dot=x[2]     # Angular velocity

    current_dot=(1/L)*u -(R/L)*current -(kb/L)*theta_dot       # Derivative of current
    theta_ddot= (kt/Jm)*current -(B/Jm)*theta_dot - (1/Jm)*Tl  # Angular aceleration
    return np.array([ current_dot, theta_dot, theta_ddot])  # ret

def PI_control(Kp,Ki,error,E_integral):
    Pout=Kp*error
    Iout=Ki*E_integral
    Salida_PI= Pout+Iout
    return Salida_PI


# def solver(h, x, u, Tl, q):
#     k1=h*model_motor(x, u, Tl)
#     k2=h*model_motor(x + (k1/5), u ,Tl )
#     k3=h*model_motor(x + (3/40)*k1 + (9/40)*k2, u ,Tl )
#     k4=h*model_motor(x + (44/45)*k1 - (56/15)*k2 + (32/9)*k3, u ,Tl )
#     k5=h*model_motor(x + (19372/6561)*k1 - (25360/2187)*k2 +(64448/6561)*k3 - (212/729)*k4, u ,Tl )
#     k6=h*model_motor(x + (9017/3168)*k1 -(355/33)*k2 +(46732/5247)*k3 + (49/176)*k4 - (5103/18656)*k5, u ,Tl )
#     k7=h*model_motor(x + (35/384)*k1 +(500/1113)*k3 +(125/192)*k4 -(2187/6784)*k5 + (11/84)*k6, u ,Tl )
#     #print(k1, k2, k3,  k4, k5, k6, k7)
#     return k1, k2, k3,  k4, k5, k6, k7
#     #q.put(np.array((k1, k2, k3,  k4, k5, k6, k7)))


#plt.figure()
fig, ax = plt.subplots()
fig.suptitle('Estimated Current in the motor')
ax.set_xlim(0, 5010)
ax.set_ylim(0, 0.22)
ax.set_ylabel('Currnet (A)')
ax.set_xlabel('Time (ms)')
# plt.xlabel('Time')
# plt.ylabel('Current/ Angle / Angular velocity')
xdata, ydata = [], []
N=5000
t_simulation=6.000  # Define here the simulation time  <<<----------------------------------------------
Step_size=0.001     # Define here the time step
dt=h=Step_size      # Asing identifiers to the time step
t_sim=np.arange(0, t_simulation, dt)    #Create a vector simulation time
    #print(len(t_sim))

    #Initial conditions for the system equations
In_current= 0          # Define Initial current 
In_position= 0         # Define Initial angular position 
In_velocity= 0         # Define Initial angular velocity
x0=np.array([ In_current, In_position, In_velocity ])  

    ### Definimos entradas externas 
    #External torque
Tl=np.zeros_like(t_sim)

    ### Creamos el vector solución y le asignamos las condiciones iniciales.
x_sol=np.zeros((len(t_sim), len(x0)))
x_sol[0]=x0

Corriente_referencia=0.2
I_reference= Corriente_referencia*np.ones(len(t_sim))
    #print(I_reference)

    # ---------------------------Definimos ganancias del controlador PI-----------------------------------
Kp=0.0
#Ki=200
Ki= 103.632275098676
    #---------------------------------------------------------------------------------------------------
    ### Definimos la condición inicial para la integral del error del controlador PI
E_integral_initial=0     #Integral initial condition
E_integral= E_integral_initial  # Assig
for i in range (1,N):
    I_error=  I_reference[i-1]- x_sol[i-1,0]         # Error 
    E_integral=E_integral+ (I_error*dt)
    u=PI_control(Kp,Ki,I_error, E_integral)     # Control signal
    #print(i)
        # Calculating the RK DP terms  for the actual step time
    k1=h*model_motor(x_sol[i-1], u, Tl[i-1])
    k2=h*model_motor(x_sol[i-1]+ (k1/5), u,Tl[i-1])
    k3=h*model_motor(x_sol[i-1]+ (3/40)*k1 + (9/40)*k2, u,Tl[i-1])
    k4=h*model_motor(x_sol[i-1]+ (44/45)*k1 - (56/15)*k2 + (32/9)*k3, u,Tl[i-1])
    k5=h*model_motor(x_sol[i-1]+ (19372/6561)*k1 - (25360/2187)*k2 +(64448/6561)*k3 - (212/729)*k4, u,Tl[i-1])
    k6=h*model_motor(x_sol[i-1]+ (9017/3168)*k1 -(355/33)*k2 +(46732/5247)*k3 + (49/176)*k4 - (5103/18656)*k5, u,Tl[i-1])
    k7=h*model_motor(x_sol[i-1]+ (35/384)*k1 +(500/1113)*k3 +(125/192)*k4 -(2187/6784)*k5 + (11/84)*k6,u,Tl[i-1])
        
        # Using the RK DP terms for solve the equations system in the actual step time and add the answer to the vector solution.
    x_sol[i]=x_sol[i-1]+ (35/384)*k1 + (500/1113)*k3 +(125/192)*k4 -(2187/6784)*k5 +(11/84)*k6
   
    xdata.append(i)
    ydata.append(x_sol[i,0])
    if i%70==0:
        #plt.plot(xdata,ydata,color = "green")
        plt.scatter(xdata,ydata,color = "red")
        plt.pause(0.001)
    
plt.close()    
end=time.perf_counter()
print(f"time take is {end-start}")
print("Proceso finalizado")


   

