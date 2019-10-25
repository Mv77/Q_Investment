## Qmod: A Python class implementing the Abel-Hayashi "Marginal Q" model of investment.

This implementation follows Professor Christopher D. Carroll's [lecture notes](http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/Investment/qModel/).

### 1. Qmod

The Qmod folder includes a file defining the Qmod Python class, which represents a Q-model of capital investment. The Qmod class' current functions include:
- Solution of the model, obtaining its policy rule.
- Drawing of the model's phase diagram.
- Simulation of the model's dynamics starting from a given level of capital.

### 2. Dolo

The Dolo folder implements the model using [Dolo](https://dolo.readthedocs.io/en/latest/#).

### 3. Examples

Current examples include:
- [Qmod_basic_features](https://github.com/Mv77/Q_Investment/blob/master/Examples/Qmod_basic_features.ipynb): illustrates how to use Qmod and its main functions.
- [Dolo_simulations](https://github.com/Mv77/Q_Investment/blob/master/Examples/Dolo_simulations.ipynb): uses Dolo to conduct more complicated simulation exercises that could not be easily achieved using Qmod.
- [Structural_changes_Qmod_Dolo](https://github.com/Mv77/Q_Investment/blob/master/Examples/Structural_changes_Qmod_Dolo.ipynb): solves the dynamic exercises in Professor Christopher D. Carroll's [lecture notes](http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/Investment/qModel/) using both Qmod and Dolo.
