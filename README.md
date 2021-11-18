#**A probabilistic model for detecting rigid domains in protein structures
**========================================
This is a python program to detect rigid domain in protein structure using Expectation Maximization and Gibbs sampler.

##**Code**
----------------------------------------

The Python code is to be found in the folder *./source*. 

##**Compatibility**
-------------

In short: Motion Detection program requires python 2.7 or higher.

##**Requirement**:
------------

   numpy==1.10.3, scipy==0.17.0, csb==1.2.5, matplotlib 
   
  
##**Usage**:
- Our main program script is ./source/gibbs.py 
- M Input structures are loaded by function gibbs.load_coordinates(codes).
- Our Gibbsampler class detects the rigid domain and categorize them into K* different cluster.

For example:

To run our program for analyzing 3 Adenylate kinase entries(1AKE_A, 4AKE_A, 1ANK_A)

- First we load the 3D coordinate using load_coordinate function:

    input_coordinate = load_coordinate(1AKE_A, 4AKE_A, 1ANK_A)
    
- Using input_coordinate value to construct Gibbsampler class and run:

    gibb = GibbsSampler(input_coordinate) # Used default value (K = 10, estimate_sigma=True, prior=1)
    gibb.run(niter = 200) 
    
- Membership for each position 
    membership = gibb.membership
More information, see  test script at ./source/test.py  

## **Reference**: 
Nguyen, Thach, and Michael Habeck. "A probabilistic model for detecting rigid domains in protein structures." Bioinformatics 32.17 (2016): i710-i717.


Motion Detection is open source and distributed under OSI-approved MIT license.
::

   Copyright (c) 2016 Michael Habeck
   
   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:
   
   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
