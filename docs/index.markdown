---
layout: splash
classes:
  - wide
---

<style>
        /* Flexbox container to align images side by side */
        .image-container {
            display: flex;
            justify-content: space-between; /* Adjust spacing between images */
        }

        /* Style for each figure element */
        figure {
            text-align: center;
            margin: 0 50px; /* Add some space between the images */
        }

        /* Ensure images are responsive */
        img {
            max-width: 100%; /* Makes sure the image doesn't overflow */
            height: auto;
        }

        /* Optional: Add caption styling */
        figcaption {
            font-style: italic;
            font-size: 0.75em;
            margin-top: 5px;
            text-align: center;
        }
</style>

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

### Training Dataset

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
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/10_lin.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/10.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/10_pred.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        $0.80$
      </td>
      <td style="text-align: center" >
        $0.14$
      </td>
      <td style="text-align: center" >
        $2.7 \times 10^4$
      </td>
      <td style="text-align: center" >
        $0.9\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Largest relative MSE for audio output
      </td>
    </tr>
    <tr>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/9_lin.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/9.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/9_pred.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        $0.25$
      </td>
      <td style="text-align: center" >
        $0.64$
      </td>
      <td style="text-align: center" >
        $2.3 \times 10^4$
      </td>
      <td style="text-align: center" >
        $1.4\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Largest relative MSE for audio output (validation)
      </td>
    </tr>
    <tr>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/33_lin.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/33.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/33_pred.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        $0.58$
      </td>
      <td style="text-align: center" >
        $0.31$
      </td>
      <td style="text-align: center" >
        $2.8 \times 10^4$
      </td>
      <td style="text-align: center" >
        $1.5\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Lowest relative MSE for audio output
      </td>
    </tr>
    <tr>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/14_lin.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/14.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/verlet_88200Hz_2sec_100modes_b560b2ccf3355481139c27275c460555/14_pred.wav" type="audio/wav">
        </audio>
      </td>
      <td style="text-align: center" >
        $0.78$
      </td>
      <td style="text-align: center" >
        $0.70$
      </td>
      <td style="text-align: center" >
        $2.9 \times 10^4$
      </td>
      <td style="text-align: center" >
        $0.9\;\mathrm{ms}$
      </td>
      <td style="text-align: left">
        Lowest relative MSE for audio output (validation)
      </td>
    </tr>
  </tbody>
</table>
