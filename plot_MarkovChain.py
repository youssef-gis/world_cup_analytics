import numpy as np

#transition matrix
A = np.matrix([[0.25, 0.2, 0.1], 
              [0.1, 0.25, 0.2], 
              [0.1, 0.1, 0.25]])
#goal vector
goal = np.transpose(np.matrix([[0.05, 0.15, 0.05]]))

# #Linear Algebra
# xT1 = np.linalg.solve(np.identity(3)-A, goal)

# print("EXpected Threats: ")
# print("Central, Box, Wing")
# print(np.transpose(xT1))

#Iterative method
xT2= np.zeros((3,1))
for i in range(10):
    xT2 = np.matmul(A, xT2) + goal

# print("EXpected Threats: ")
# print("Central, Box, Wing")
# print(np.transpose(xT2))

#simulation method
n_simulation=5
xT3 = np.zeros(3)
description = {0:"Central", 1:"Box", 2:"Wing"}
for i in range(3): 
    num_goals =  0
    print('---------------------------------')
    print('Starting position: ', description[i])
    print('---------------------------------')

    for n in range(n_simulation):
        ball_in_play = True
        s=i
        describe_position=''
        while ball_in_play:
            r = np.random.rand()

            # Make commentary
            describe_position = describe_position + ' - ' + description[s]

            c_sum = np.cumsum(A[s,:])

            new_s = np.sum(c_sum < r)

            if new_s > 2 :
                #Ball is either goal or out of play
                ball_in_play = False
                if r < goal[s]+c_sum[0, 2]:
                    #Goal
                    num_goals= num_goals + 1
                    describe_position= describe_position + ' - Goal'
                else:
                    #Out of play
                    describe_position= describe_position + ' - Out of play'
            s= new_s
        print(describe_position)
    xT3[i] = num_goals/n_simulation

print('\n\n---------------------------------')
print('Expected Threats: ')
print('Central, Box, Wing')
print(xT3)
