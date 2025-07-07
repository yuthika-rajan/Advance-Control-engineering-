"""
Preset: a 3-wheel robot (kinematic model a. k. a. non-holonomic integrator).

"""
  
import pathlib  
import os  
import warnings
import csv
from datetime import datetime
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import systems
import simulator
import controllers
import loggers
import visuals
from utilities import on_key_press

import argparse

#----------------------------------------Set up dimensions
dim_state = 3
dim_input = 2
dim_output = dim_state
dim_disturb = 0

dim_R1 = dim_output + dim_input
dim_R2 = dim_R1

description = "Agent-environment preset: a 3-wheel robot (kinematic model a.k.a. non-holonomic integrator)."

parser = argparse.ArgumentParser(description=description)

parser.add_argument('--ctrl_mode', metavar='ctrl_mode', type=str,
                    choices=['MPC',
                             "Nominal",
                             "lqr"],
                    default="MPC",
                    help='Control mode. Currently available: ' +
                    '----manual: manual constant control specified by action_manual; ' +
                    '----nominal: nominal controller, usually used to benchmark optimal controllers;' +                     
                    '----MPC:model-predictive control; ' +
                    '----RQL: Q-learning actor-critic with Nactor-1 roll-outs of running objective; ' +
                    '----SQL: stacked Q-learning; ' + 
                    '----RLStabLyap: (experimental!) learning agent with Lyapunov-like stabilizing contraints.')
parser.add_argument('--dt', type=float, metavar='dt',
                    default=0.1,
                    help='Controller sampling time.' )
parser.add_argument('--t1', type=float, metavar='t1',
                    default=20,
                    help='Final time of episode.' )
parser.add_argument('--Nruns', type=int,
                    default=1,
                    help='Number of episodes. Learned parameters are not reset after an episode.')
parser.add_argument('--is_log_data', type=int,
                    default=1,
                    help='Flag to log data into a data file. Data are stored in simdata folder.')
parser.add_argument('--is_visualization', type=int,
                    default=1
                    ,
                    help='Flag to produce graphical output.')
parser.add_argument('--is_print_sim_step', type=int,
                    default=1,
                    help='Flag to print simulation data into terminal.')
parser.add_argument('--action_manual', type=float,
                    default=[-5, -3], nargs='+',
                    help='Manual control action to be fed constant, system-specific!')
parser.add_argument('--Nactor', type=int,
                    default=15,
                    help='Horizon length (in steps) for predictive controllers.')
parser.add_argument('--pred_step_size_multiplier', type=float,
                    default=5.0,
                    help='Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time dt.')
parser.add_argument('--buffer_size', type=int,
                    default=25,
                    help='Size of the buffer (experience replay) for model estimation, agent learning etc.')
parser.add_argument('--run_obj_struct', type=str,
                    default='quadratic',
                    choices=['quadratic',
                             'biquadratic'],
                    help='Structure of running objective function.')
parser.add_argument('--Q', type=float, nargs='+',
                    default=[60,60,60])
parser.add_argument('--R', type=float,nargs='+',
                    default=[1, 1])
parser.add_argument('--R1_diag', type=float, nargs='+',
                    default=[105, 105, 10, 10, 5],
                    help='Parameter of running objective function. Must have proper dimension. ' +
                    'Say, if chi = [observation, action], then a quadratic running objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.')
parser.add_argument('--Qf', type=float, nargs='+',
                    default=[10,10,10])

parser.add_argument('--R2_diag', type=float, nargs='+',
                    default=[1, 10, 1, 0, 0],
                    help='Parameter of running objective function . Must have proper dimension. ' + 
                    'Say, if chi = [observation, action], then a bi-quadratic running objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, ' +
                    'where diag() is transformation of a vector to a diagonal matrix.')
parser.add_argument('--Ncritic', type=int,
                    default=25,
                    help='Critic stack size (number of temporal difference terms in critic cost).')
parser.add_argument('--gamma', type=float,
                    default=0.9,
                    help='Discount factor.')
parser.add_argument('--critic_period_multiplier', type=float,
                    default=1.0,
                    help='Critic is updated every critic_period_multiplier times dt seconds.')
parser.add_argument('--critic_struct', type=str,
                    default='quadratic', choices=['quad-lin',
                                                   'quadratic',
                                                   'quad-nomix',
                                                   'quad-mix',
                                                   'poly3',
                                                   'poly4'],
                    help='Feature structure (critic). Currently available: ' +
                    '----quad-lin: quadratic-linear; ' +
                    '----quadratic: quadratic; ' +
                    '----quad-nomix: quadratic, no mixed terms; ' +
                    '----quad-mix: quadratic, mixed observation-action terms (for, say, Q or advantage function approximations); ' +
                    '----poly3: 3-order model, see the code for the exact structure; ' +
                    '----poly4: 4-order model, see the code for the exact structure. '
                    )
parser.add_argument('--actor_struct', type=str,
                    default='quad-nomix', choices=['quad-lin',
                                                   'quadratic',
                                                   'quad-nomix'],
                    help='Feature structure (actor). Currently available: ' +
                    '----quad-lin: quadratic-linear; ' +
                    '----quadratic: quadratic; ' +
                    '----quad-nomix: quadratic, no mixed terms.')
parser.add_argument('--init_robot_pose_x', type=float,
                    default=-3.0,
                    help='Initial x-coordinate of the robot pose.')
parser.add_argument('--init_robot_pose_y', type=float,
                    default=-3.0,
                    help='Initial y-coordinate of the robot pose.')
parser.add_argument('--init_robot_pose_theta', type=float,
                    default=1.57,
                    help='Initial orientation angle (in radians) of the robot pose.')
parser.add_argument('--distortion_x', type=float,
                    default=-0.6,
                    help='X-coordinate of the center of distortion.')
parser.add_argument('--distortion_y', type=float,
                    default=-0.5,
                    help='Y-coordinate of the center of distortion.')
parser.add_argument('--distortion_sigma', type=float,
                    default=0.1,
                    help='Standard deviation (sigma) of distortion.')
parser.add_argument('--seed', type=int,
                    default=1,
                    help='Seed for random number generation.')

args = parser.parse_args()
Nactor = int(os.getenv("NACTOR", args.Nactor))
r1_str = os.getenv("R1_DIAG")
qf_str = os.getenv("QF")
q_str = os.getenv("Q")
r_str = os.getenv("R")

# MPC: R1 (running objective)
if r1_str:
    R1 = np.diag([float(x) for x in r1_str.split()])
else:
    R1 = np.diag(args.R1_diag)

# MPC: Qf (terminal cost)
if qf_str:
    Qf = np.diag([float(x) for x in qf_str.split()])
else:
    Qf = np.diag(args.Qf)

# LQR: Q matrix (state error)
if q_str:
    Q = np.diag([float(x) for x in q_str.split()])
else:
    Q = np.diag(args.Q)

# LQR: R matrix (control effort)
if r_str:
    R = np.diag([float(x) for x in r_str.split()])
else:
    R = np.diag(args.R)

seed=args.seed
print(seed)

xdistortion_x = args.distortion_x
ydistortion_y = args.distortion_y
distortion_sigma = args.distortion_sigma

x = args.init_robot_pose_x
y = args.init_robot_pose_y
theta = args.init_robot_pose_theta

while theta > np.pi:
        theta -= 2 * np.pi
while theta < -np.pi:
        theta += 2 * np.pi

state_init = np.array([x, y, theta])

args.action_manual = np.array(args.action_manual)

pred_step_size = args.dt * args.pred_step_size_multiplier
critic_period = args.dt * args.critic_period_multiplier

Nactor = int(os.getenv("NACTOR", args.Nactor))

r1_str = os.getenv("R1_DIAG")
qf_str = os.getenv("QF")

if r1_str:
    R1 = np.diag([float(x) for x in r1_str.split()])
else:
    R1 = np.diag(np.array(args.R1_diag))

if qf_str:
    Qf = np.diag([float(x) for x in qf_str.split()])
else:
    Qf = np.diag(np.array(args.Qf))

assert args.t1 > args.dt > 0.0
assert state_init.size == dim_state

globals().update(vars(args))

#----------------------------------------Fixed settings
is_disturb = 0
is_dyn_ctrl = 0

t0 = 0

action_init = 0 * np.ones(dim_input)

# Solver
atol = 1e-3
rtol = 1e-2

# xy-plane
xMin = -4#-1.2
xMax = 2
yMin = -4#-1.2
yMax = 2

# Control constraints
v_min = -0.22 *10
v_max = 0.22 *10
omega_min = -2.84
omega_max = 2.84

ctrl_bnds=np.array([[v_min, v_max], [omega_min, omega_max]])
# ctrl_bnds=np.zeros((2,2))

#----------------------------------------Initialization : : system
my_sys = systems.Sys3WRobotNI(sys_type="diff_eqn", 
                                     dim_state=dim_state,
                                     dim_input=dim_input,
                                     dim_output=dim_output,
                                     dim_disturb=dim_disturb,
                                     pars=[],
                                     ctrl_bnds=ctrl_bnds,
                                     is_dyn_ctrl=is_dyn_ctrl,
                                     is_disturb=is_disturb,
                                     pars_disturb=[])

observation_init = my_sys.out(state_init)

xCoord0 = state_init[0]
yCoord0 = state_init[1]
alpha0 = state_init[2]
alpha_deg_0 = alpha0/2*np.pi

#----------------------------------------Initialization : : model

#----------------------------------------Initialization : : controller

target_x=0.0
target_y=0.0
target_theta=0.0
dt=0.1
# my_ctrl_nominal = controllers.N_CTRL(k_rho=1.0, k_alpha=4.0, k_beta=-1.5, # Example gains, vary for Experiment A
#         target_x=target_x, target_y=target_y, target_theta=target_theta,
#         sampling_time=dt,
#         dim_input=dim_input, dim_output=dim_output
#     )


#  Nominal forward velocity for linearization
v_nom =0.001

# Linearized discrete system (unit time step)
A = np.array([
    [1, 0, -v_nom * dt*np.sin(0)],
    [0, 1,  v_nom *dt* np.cos(0)],
    [0, 0, 1]
])

# print(A)

B = np.array([
    [dt*np.cos(0), 0],
    [dt*np.sin(0), 0],
    [0, dt]
])

# ===============================
# 2. Define cost matrices
# ===============================

# Q penalizes state error (x, y, theta)
Q = np.diag(Q)

# R penalizes control effort (v, omega)
R = np.diag(R)

print(Q)
print(R)
my_ctrl_lqr=controllers.LQR(A=A,B=B,Q=Q,R=R,obs_target=np.array([0.0,0.0,0.0]),sampling_time=dt)


# Predictive optimal controller
my_ctrl_opt_pred = controllers.ControllerOptimalPredictive(dim_input,
                                           dim_output,
                                           ctrl_mode,
                                           ctrl_bnds = ctrl_bnds,
                                           action_init = [],
                                           t0 = t0,
                                           sampling_time = dt,
                                           Nactor = Nactor,
                                           pred_step_size = pred_step_size,
                                           sys_rhs = my_sys._state_dyn,
                                           sys_out = my_sys.out,
                                           state_sys = state_init,
                                           buffer_size = buffer_size,
                                           gamma = gamma,
                                           Ncritic = Ncritic,
                                           critic_period = critic_period,
                                           critic_struct = critic_struct,
                                           run_obj_struct = run_obj_struct,
                                           run_obj_pars = [R1,Qf],
                                           observation_target = [0.0,0.0,0.0],
                                           state_init=state_init,
                                           obstacle=[xdistortion_x, ydistortion_y,distortion_sigma],
                                           seed=seed)


my_ctrl_benchm = my_ctrl_opt_pred
    
#----------------------------------------Initialization : : simulator
my_simulator = simulator.Simulator(sys_type = "diff_eqn",
                                   closed_loop_rhs = my_sys.closed_loop_rhs,
                                   sys_out = my_sys.out,
                                   state_init = state_init,
                                   disturb_init = [],
                                   action_init = action_init,
                                   t0 = t0,
                                   t1 = t1,
                                   dt = dt,
                                   max_step = dt,
                                   first_step = 1e-4,
                                   atol = atol,
                                   rtol = rtol,
                                   is_disturb = is_disturb,
                                   is_dyn_ctrl = is_dyn_ctrl)

#----------------------------------------Initialization : : logger
date = datetime.now().strftime("%Y-%m-%d")
time = datetime.now().strftime("%Hh%Mm%Ss")
datafiles = [None] * Nruns

data_folder = 'simdata/' + ctrl_mode + "/Init_angle_{}_seed_{}_Nactor_{}".format(str(state_init[2]), seed, Nactor)

if is_log_data:
    pathlib.Path(data_folder).mkdir(parents=True, exist_ok=True) 

for k in range(0, Nruns):
    datafiles[k] = data_folder + '/' + my_sys.name + '_' + ctrl_mode + '_' + date + '_' + time + '__run{run:02d}.csv'.format(run=k+1)
    
    if is_log_data:
        print('Logging data to:    ' + datafiles[k])
            
        with open(datafiles[k], 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['System', my_sys.name ] )
            writer.writerow(['Controller', ctrl_mode ] )
            writer.writerow(['dt', str(dt) ] )
            writer.writerow(['state_init', str(state_init) ] )
            writer.writerow(['Nactor', str(Nactor) ] )
            writer.writerow(['pred_step_size_multiplier', str(pred_step_size_multiplier) ] )
            writer.writerow(['buffer_size', str(buffer_size) ] )
            writer.writerow(['run_obj_struct', str(run_obj_struct) ] )
            writer.writerow(['R1_diag', str(R1_diag) ] )
            writer.writerow(['R2_diag', str(R2_diag) ] )
            writer.writerow(['Ncritic', str(Ncritic) ] )
            writer.writerow(['gamma', str(gamma) ] )
            writer.writerow(['critic_period_multiplier', str(critic_period_multiplier) ] )
            writer.writerow(['critic_struct', str(critic_struct) ] )
            writer.writerow(['actor_struct', str(actor_struct) ] )   
            writer.writerow(['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'run_obj', 'accum_obj', 'v [m/s]', 'omega [rad/s]'] )

# Do not display annoying warnings when print is on
if is_print_sim_step:
    warnings.filterwarnings('ignore')

k_rho = 2.0
k_alpha = 5.0
k_beta = -1.5

    
my_logger = loggers.Logger3WRobotNI()
my_ctrl_nominal=controllers.N_CTRL(k_rho=k_rho, k_alpha=k_alpha, k_beta=k_beta, # Example gains, vary for Experiment A
        target_x=0.0, target_y=0.0, target_theta=0.0,
        sampling_time=dt,
        dim_input=dim_input, dim_output=dim_output)

# my_ctrl_lqr=controllers.LQR(A=np.diag([2.0,3.0,4.0]),B=np.diag([2.0,3.0]),Q=np.diag([1.0,2.0,3.0]),R=np.diag([3.0,4.0]),obs_target=np.array([0.0,0.0,0.0]))

#----------------------------------------Main loop
state_full_init = my_simulator.state_full

if is_visualization:
    my_animator = visuals.Animator3WRobotNI(objects=(my_simulator,
                                                     my_sys,
                                                     my_ctrl_nominal,
                                                     my_ctrl_benchm,
                                                     my_ctrl_lqr,
                                                     datafiles,
                                                     controllers.ctrl_selector,
                                                     my_logger),
                                            pars=(state_init,
                                                  action_init,
                                                  t0,
                                                  t1,
                                                  state_full_init,
                                                  xMin,
                                                  xMax,
                                                  yMin,
                                                  yMax,
                                                  ctrl_mode,
                                                  action_manual,
                                                  v_min,
                                                  omega_min,
                                                  v_max,
                                                  omega_max,
                                                  Nruns,
                                                  is_print_sim_step, is_log_data, 0, [], [xdistortion_x, ydistortion_y,distortion_sigma]))

    anm = animation.FuncAnimation(my_animator.fig_sim,
                                  my_animator.animate,
                                  init_func=my_animator.init_anim,
                                  blit=False, interval=dt/1e6, repeat=True)
    print("ALSO GOOD")
    my_animator.get_anm(anm)
    
    cId = my_animator.fig_sim.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event, anm))
    
    anm.running = True
    
    my_animator.fig_sim.tight_layout()
    
    plt.show()
    
else:   
    run_curr = 1
    datafile = datafiles[0]
    
    while True:
        
        my_simulator.sim_step()
        
        t, state, observation, state_full = my_simulator.get_sim_step_data()
        
        action = controllers.ctrl_selector(t, observation, action_manual, my_ctrl_nominal, my_ctrl_benchm,my_ctrl_lqr, ctrl_mode)
        print("action: ", action,ctrl_mode)
        
        my_sys.receive_action(action)
        my_ctrl_benchm.receive_sys_state(my_sys._state)
        my_ctrl_benchm.upd_accum_obj(observation, action)
        
        xCoord = state_full[0]
        yCoord = state_full[1]
        alpha = state_full[2]
        print("sample")
        
        run_obj = my_ctrl_benchm.run_obj(observation, action)
        accum_obj = my_ctrl_benchm.accum_obj_val
        
        # count_CALF = my_ctrl_benchm.D_count()
        # count_N_CTRL = my_ctrl_benchm.get_N_CTRL_count()

        if is_print_sim_step:
            my_logger.print_sim_step(t, xCoord, yCoord, alpha, run_obj, accum_obj, action)
            
        if is_log_data:
            my_logger.log_data_row(datafile, t, xCoord, yCoord, alpha, run_obj, accum_obj, action)
        

        if t >= t1 or np.linalg.norm(observation[:2]) < 0.2:

            # Reset simulator
            my_simulator.reset()
            
            if ctrl_mode != 'MPC':
                my_ctrl_benchm.reset(t0)
            elif ctrl_mode=='lqr':
                my_ctrl_lqr.reset(t0)
            else:
                my_ctrl_nominal.reset(t0)
            
            accum_obj = 0 

            if is_print_sim_step:
                print('.....................................Run {run:2d} done.....................................'.format(run = run_curr))
                
            run_curr += 1
            
            if run_curr > Nruns:
                plt.close('all')
                break
                
            if is_log_data:
                datafile = datafiles[run_curr-1]
                 