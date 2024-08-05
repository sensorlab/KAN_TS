results_conv_kan: layer_sizes=[40,40], 
                  l1_activation_penalty=0.1, 
                  droput=0.5
                  optimizer = optim.AdamW(model.parameters(), lr=0.001)
                  NO SCHEDULER! (for changing learning rate)

results_conv_kan2: layer_sizes = [8 * 4, 16 * 4, 32 * 4, 64 * 4], 
                   l1_activation_penalty = 0.0, 
                   dropout = 0.25
                   optimizer = optim.AdamW(model.parameters(), lr=0.001)
                   NO SCHEDULER! (for changing learning rate)

for both: epochs=500
           