# NN Playground

Tiny neural nets trained live: decision boundary, per-neuron maps and loss curve,
redrawn every frame.

## Install and run (Windows 11)

    pip install torch numpy pyqtgraph PyQt6
    python nn_playground.py

`--opengl` switches the viewport to OpenGL; try it if frames feel heavy.

## Controls

- **Hidden layers**: `4,4` = two layers of 4. `8:sin,4:tanh` sets per-layer activations. Empty = linear model.
- **Activation**: tanh, relu, leaky_relu, sigmoid, gelu, silu, mish, elu, softplus, sin, gauss, abs, square, linear.
- **Inputs**: extra features (x², y², x·y, sin, r) like TF Playground. Checking x² and y² solves Circles with zero hidden neurons.
- **Residual**: adds skip connections between same-width layers.
- **Steps / frame**: training speed. Frames render as fast as the display allows.
- **Grid res**: boundary resolution (128 default; 256 is sharper, costs ~4x render).
- **Neuron view**: which hidden layer's neurons to show (blue = positive, orange = negative, up to 32).
- Keys: Space play/pause, S step, R reset weights, N new data.

## Challenges

- XOR gate (4 points): 2 tanh neurons solve it (most seeds). 1 neuron works too: `1:square` every seed I tried, `1:abs` only on some. Why?
- Spiral (2 arms) with x, y only: find the smallest net that hits 100%. Try `sin`.
- Same spiral, residual on, `16,16,16,16` relu vs gelu.
