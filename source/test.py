'''
This is an example using 3 input ensembles Adenylate Kinase pdb_chainID: 1AKE_A, 4AKE_A, 1ANK_A
'''
import numpy as np
from gibbs import load_coordinates, GibbsSampler
from matplotlib import pyplot as plt

input_coordinate = load_coordinates('1AKE_A', '4AKE_A', '1ANK_A')

#Run Gibb sampler using 2 priors
gibb = GibbsSampler(input_coordinate, prior=2)
gibb.run(500)

print 'Number of Domain = ', np.unique(gibb.membership).shape[0]
print 'Membership', gibb.membership
print 'Log likelihood = ', gibb.log_likelihood

plt.plot(gibb.membership)
plt.title('Membership')
plt.show()





