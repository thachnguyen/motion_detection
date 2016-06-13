from gibbs import load_coordinates, GibbsSampler
input_coordinate = load_coordinates('1AKE_A', '4AKE_A', '1ANK_A')
gibb = GibbsSampler(input_coordinate, prior=2)
gibb.run(500)
