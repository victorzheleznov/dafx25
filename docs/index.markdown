---
layout: splash
classes:
  - wide
---

<h2 style="font-size: 1.5em" align="center">
  Learning Nonlinear Dynamics in Physical Modelling Synthesis using Neural Ordinary Differential Equations
</h2>

<p style="font-size: 1.0em" align="center">
  Victor Zheleznov<sup>1</sup>, Stefan Bilbao<sup>1</sup>, Alec Wright<sup>1</sup> and Simon King<sup>2</sup>
</p>

<p style="text-align: center; font-size: 0.75em">
  <i>
    <sup>1</sup><a href="https://www.acoustics.ed.ac.uk/" target="_blank" rel="noopener noreferrer">Acoustics and Audio Group</a>, University of Edinburgh, Edinburgh, UK<br>
    <sup>2</sup><a href="https://www.cstr.ed.ac.uk/" target="_blank" rel="noopener noreferrer">Centre for Speech Technology Research</a>, University of Edinburgh, Edinburgh, UK<br>
  </i>
</p>

<p style="font-size: 1.0em; text-align: center">
  Accompanying web-page for the DAFx25 submission
</p>

<div style="text-align: center; align-items: center">
  <a href="https://github.com/victorzheleznov/dafx25" class="btn btn--primary btn--small" target="_blank" rel="noopener noreferrer">
    Code
  </a>
</div>



## Abstract

Modal synthesis methods are a long-standing approach for modelling distributed musical systems. In some cases extensions are possible in order to handle geometric nonlinearities. One such case is the high-amplitude vibration of a string, where geometric nonlinear effects lead to perceptually important effects including pitch glides and a dependence of brightness on striking amplitude. A modal decomposition leads to a coupled nonlinear system of ordinary differential equations. Recent work in applied machine learning approaches (in particular neural ordinary differential equations) has been used to model lumped dynamic systems such as electronic circuits automatically from data. In this work, we examine how modal decomposition can be combined with neural ordinary differential equations for modelling distributed musical systems. The proposed model leverages the analytical solution for linear vibration of system's modes and employs a neural network to account for nonlinear dynamic behaviour. Physical parameters of a system remain easily accessible after the training without the need for a parameter encoder in the network architecture. As an initial proof of concept, we generate synthetic data for a nonlinear transverse string and show that the model can be trained to reproduce the nonlinear dynamics of the system. Sound examples are presented.



## Sound Examples

Below are some selected sound examples along with string and excitation parameters for the datasets used in the submission. All sound examples can be downloaded from [the accompanying repository](https://github.com/victorzheleznov/dafx25/tree/master/audio).



### Test Dataset

<table>
  <thead>
    <tr>
      <th style="text-align: center">Linear</th>
      <th style="text-align: center">Target</th>
      <th style="text-align: center">Predicted</th>
      <th style="text-align: center">$\gamma$</th>
      <th style="text-align: center">$\kappa$</th>
      <th style="text-align: center">$x_e$</th>
      <th style="text-align: center">$x_o$</th>
      <th style="text-align: center">$f_{\mathrm{amp}}$</th>
      <th style="text-align: center">$T_e$</th>
      <th style="text-align: center">Note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/9_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/9_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/9_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        $155.4$
      </td>
      <td style="text-align: center">
        $1.07$
      </td>
      <td style="text-align: center">
        $0.79$
      </td>
      <td style="text-align: center">
        $0.87$
      </td>
      <td style="text-align: center">
        $2.3 \times 10^4$
      </td>
      <td style="text-align: center">
        $1.1\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Largest relative MSE for audio output (illustrated example in the manuscript)
      </td>
    </tr>
    <tr>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/34_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/34_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/34_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        $233.2$
      </td>
      <td style="text-align: center">
        $1.03$
      </td>
      <td style="text-align: center">
        $0.31$
      </td>
      <td style="text-align: center">
        $0.66$
      </td>
      <td style="text-align: center">
        $2.0 \times 10^4$
      </td>
      <td style="text-align: center">
        $1.3\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Lowest relative MSE for audio output
      </td>
    </tr>
    <tr>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/37_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/37_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/37_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        $161.8$
      </td>
      <td style="text-align: center">
        $1.02$
      </td>
      <td style="text-align: center">
        $0.89$
      </td>
      <td style="text-align: center">
        $0.76$
      </td>
      <td style="text-align: center">
        $2.7 \times 10^4$
      </td>
      <td style="text-align: center">
        $0.5\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #1
      </td>
    </tr>
    <tr>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/56_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/56_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/56_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        $203.5$
      </td>
      <td style="text-align: center">
        $1.04$
      </td>
      <td style="text-align: center">
        $0.74$
      </td>
      <td style="text-align: center">
        $0.71$
      </td>
      <td style="text-align: center">
        $2.9 \times 10^4$
      </td>
      <td style="text-align: center">
        $1.5\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #2
      </td>
    </tr>
    <tr>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/3_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/3_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/3_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        $168.3$
      </td>
      <td style="text-align: center">
        $1.01$
      </td>
      <td style="text-align: center">
        $0.46$
      </td>
      <td style="text-align: center">
        $0.34$
      </td>
      <td style="text-align: center">
        $2.8 \times 10^4$
      </td>
      <td style="text-align: center">
        $1.1\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #3
      </td>
    </tr>
    <tr>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/39_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/39_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/39_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        $242.0$
      </td>
      <td style="text-align: center">
        $1.03$
      </td>
      <td style="text-align: center">
        $0.15$
      </td>
      <td style="text-align: center">
        $0.20$
      </td>
      <td style="text-align: center">
        $2.2 \times 10^4$
      </td>
      <td style="text-align: center">
        $1.3\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #4
      </td>
    </tr>
    <tr>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/8_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/8_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/8_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        $139.4$
      </td>
      <td style="text-align: center">
        $1.09$
      </td>
      <td style="text-align: center">
        $0.57$
      </td>
      <td style="text-align: center">
        $0.78$
      </td>
      <td style="text-align: center">
        $2.8 \times 10^4$
      </td>
      <td style="text-align: center">
        $1.3\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #5
      </td>
    </tr>
    <tr>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/28_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/28_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/28_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        $182.7$
      </td>
      <td style="text-align: center">
        $1.02$
      </td>
      <td style="text-align: center">
        $0.79$
      </td>
      <td style="text-align: center">
        $0.14$
      </td>
      <td style="text-align: center">
        $2.7 \times 10^4$
      </td>
      <td style="text-align: center">
        $1.1\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #6
      </td>
    </tr>
    <tr>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/24_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/24_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/24_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        $221.8$
      </td>
      <td style="text-align: center">
        $1.01$
      </td>
      <td style="text-align: center">
        $0.41$
      </td>
      <td style="text-align: center">
        $0.78$
      </td>
      <td style="text-align: center">
        $2.9 \times 10^4$
      </td>
      <td style="text-align: center">
        $0.8\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #7
      </td>
    </tr>
    <tr>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/44_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/44_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/44_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        $191.3$
      </td>
      <td style="text-align: center">
        $1.08$
      </td>
      <td style="text-align: center">
        $0.34$
      </td>
      <td style="text-align: center">
        $0.88$
      </td>
      <td style="text-align: center">
        $2.9 \times 10^4$
      </td>
      <td style="text-align: center">
        $0.6\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #8
      </td>
    </tr>
    <tr>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/22_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/22_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/22_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        $235.7$
      </td>
      <td style="text-align: center">
        $1.06$
      </td>
      <td style="text-align: center">
        $0.26$
      </td>
      <td style="text-align: center">
        $0.23$
      </td>
      <td style="text-align: center">
        $2.7 \times 10^4$
      </td>
      <td style="text-align: center">
        $0.8\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #9
      </td>
    </tr>
    <tr>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/1_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/1_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_96000Hz_3sec_100modes_f10f6bff8da73917e3f7a795f58b2043/1_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        $149.3$
      </td>
      <td style="text-align: center">
        $1.05$
      </td>
      <td style="text-align: center">
        $0.18$
      </td>
      <td style="text-align: center">
        $0.89$
      </td>
      <td style="text-align: center">
        $2.1 \times 10^4$
      </td>
      <td style="text-align: center">
        $0.8\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #10
      </td>
    </tr>
  </tbody>
</table>



### Training and Validation Dataset

<table>
  <thead>
    <tr>
      <th style="text-align: center">Linear</th>
      <th style="text-align: center">Target</th>
      <th style="text-align: center">Predicted</th>
      <th style="text-align: center">$x_e$</th>
      <th style="text-align: center">$x_o$</th>
      <th style="text-align: center">$f_{\mathrm{amp}}$</th>
      <th style="text-align: center">$T_e$</th>
      <th style="text-align: center">Note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/10_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/10_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/10_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        $0.80$
      </td>
      <td style="text-align: center">
        $0.14$
      </td>
      <td style="text-align: center">
        $2.7 \times 10^4$
      </td>
      <td style="text-align: center">
        $0.9\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Largest relative MSE for audio output
      </td>
    </tr>
    <tr>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/33_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/33_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/33_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center">
        $0.58$
      </td>
      <td style="text-align: center">
        $0.31$
      </td>
      <td style="text-align: center">
        $2.8 \times 10^4$
      </td>
      <td style="text-align: center">
        $1.5\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Lowest relative MSE for audio output
      </td>
    </tr>
    <tr>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/52_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/52_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/52_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        $0.84$
      </td>
      <td style="text-align: center" >
        $0.87$
      </td>
      <td style="text-align: center" >
        $3.0 \times 10^4$
      </td>
      <td style="text-align: center" >
        $1.0\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #1
      </td>
    </tr>
    <tr>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/4_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/4_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/4_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        $0.19$
      </td>
      <td style="text-align: center" >
        $0.13$
      </td>
      <td style="text-align: center" >
        $2.1 \times 10^4$
      </td>
      <td style="text-align: center" >
        $0.8\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #2
      </td>
    </tr>
    <tr>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/5_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/5_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/5_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        $0.60$
      </td>
      <td style="text-align: center" >
        $0.49$
      </td>
      <td style="text-align: center" >
        $2.8 \times 10^4$
      </td>
      <td style="text-align: center" >
        $1.4\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #3
      </td>
    </tr>
    <tr>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/40_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/40_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/40_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        $0.59$
      </td>
      <td style="text-align: center" >
        $0.26$
      </td>
      <td style="text-align: center" >
        $2.4 \times 10^4$
      </td>
      <td style="text-align: center" >
        $0.9\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #4
      </td>
    </tr>
    <tr>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/13_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/13_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/13_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        $0.24$
      </td>
      <td style="text-align: center" >
        $0.78$
      </td>
      <td style="text-align: center" >
        $2.9 \times 10^4$
      </td>
      <td style="text-align: center" >
        $1.3\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #5
      </td>
    </tr>
    <tr>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/7_lin_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/7_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/7_pred_PCM_24_0.1dBFS.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        $0.46$
      </td>
      <td style="text-align: center" >
        $0.39$
      </td>
      <td style="text-align: center" >
        $2.7 \times 10^4$
      </td>
      <td style="text-align: center" >
        $0.7\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Selected example #6
      </td>
    </tr>
  </tbody>
</table>
