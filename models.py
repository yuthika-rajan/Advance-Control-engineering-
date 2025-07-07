"""
Contains classes to be used in fitting system models.
Includes both State-Space and placeholder Neural Network models.
"""

import numpy as np # Importing numpy for matrix operations

class ModelSS:
    """
    Discrete-Time State-Space Model

    \[
    \begin{array}{ll}
        \hat{x}_{k+1} &= A \hat{x}_k + B u_k \\
        y_k &= C \hat{x}_k + D u_k
    \end{array}
    \]

    Attributes:
    -----------
    A, B, C, D : ndarray
        State-space matrices defining system dynamics.
    x0set : ndarray
        Initial condition (state estimate) for simulation.
    """

    def __init__(self, A, B, C, D, x0est):  # Consistent naming for initial state
        self.A = A  # State transition matrix
        self.B = B  # Input matrix
        self.C = C  # Output matrix
        self.D = D  # Feedforward matrix
        self.x0set = x0est  # Consistent naming for initial state

    def upd_pars(self, Anew, Bnew, Cnew, Dnew): 
        """Update system matrices with new values."""
        self.A = Anew   # Update state transition matrix
        self.B = Bnew   # Update input matrix
        self.C = Cnew   # Update output matrix
        self.D = Dnew   # Update feedforward matrix

    def updateIC(self, x0setNew):   
        """Update initial condition/state estimate."""
        self.x0set = x0setNew   # Consistent naming for initial state

    def predict(self, x, u):
        """
        Perform one-step forward simulation using current model.

        Parameters:
        -----------
        x : ndarray
            Current state.
        u : ndarray
            Current control input.

        Returns:
        --------
        x_next : ndarray
            Predicted next state.
        y : ndarray
            Output signal.
        """
        x_next = self.A @ x + self.B @ u    # State update equation
        y = self.C @ x + self.D @ u   # Output equation
        return x_next, y    


class ModelNN:  
    """
    Placeholder for Neural Network Model.
    Intended to be implemented in the future for learning nonlinear dynamics.
    """

    def __init__(self, *args, **kwargs):    
        raise NotImplementedError(f"Class {self.__class__.__name__} is not yet implemented.")  
        # Placeholder for future implementation
        # This will raise an error if instantiated, indicating that the class is not yet ready for use.
    def predict(self, x, u):
        """
        Placeholder for prediction method.
        Intended to be implemented in the future for learning nonlinear dynamics.

        Parameters:
        -----------
        x : ndarray
            Current state.
        u : ndarray
            Current control input.

        Returns:
        --------
        NotImplementedError
            Indicates that this method is not yet implemented.
        """
        raise NotImplementedError(f"Method {self.predict.__name__} in class {self.__class__.__name__} is not yet implemented.")
        # Placeholder for future implementation
        # This will raise an error if called, indicating that the method is not yet ready for use.
