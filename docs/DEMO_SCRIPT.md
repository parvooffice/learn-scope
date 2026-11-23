# 2-minute demo script

Start the app (`python -m learnscope`), open <http://localhost:8080> and record with any screen recorder (1440×900, browser zoom 100%).

| Time      | Do                                                                                                                                                                              | Say                                                                                                                                          |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 0:00–0:15 | Open the landing page. The spiral hero is training live. Click **Open the Playground**.                                                                                          | "This is LearnScope: you can watch a neural network learn, live. The engine behind it is written from scratch in Python and NumPy."          |
| 0:15–0:40 | The playground trains the spiral. Point at the decision boundary, the network graph, the loss curve.                                                                             | "Weights are red or blue lines, the heat map is what the network believes, and these charts show gradients and activations for every layer." |
| 0:40–1:05 | Drag the learning-rate slider to ~1.5. Wait a few seconds for the **Model Doctor** cards (it reports _Learning rate looks too high_ plus its consequences).                      | "Normally you only see a loss number. The Doctor reads the telemetry and tells me the learning rate is too high, with the evidence."         |
| 1:05–1:20 | Click **Show me** on the learning-rate card, then the fix button. Training restarts with a lower rate and the Doctor says "No problems detected" once it has recovered.          | "It jumps to the step where it went wrong, and one click applies the fix."                                                                   |
| 1:20–1:35 | Drag the **timeline** back through training; boundary, weights and histograms rewind. Click **Back to live**.                                                                    | "Every step is recorded, so I can time-travel and inspect the whole network at any moment."                                                  |
| 1:35–1:50 | Open **Backprop**; press Next through the steps to the update.                                                                                                                   | "This explains backprop with real numbers from my from-scratch autograd engine, checked against a hand-written chain rule."                  |
| 1:50–2:00 | Open **MNIST**, press Train for ~10 seconds, draw a digit, show the prediction and feature maps.                                                                                 | "And it trains a convolutional network, written from scratch in NumPy, on real handwritten digits; draw one and watch it classify it."       |

Optional extra: the **Compare** page (Adam vs SGD on the same spiral) makes a good 15-second cutaway.

Suggested GIF: the 0:15–1:35 stretch, 12 fps, ~900 px wide (a tool like ScreenToGif works well on Windows).
