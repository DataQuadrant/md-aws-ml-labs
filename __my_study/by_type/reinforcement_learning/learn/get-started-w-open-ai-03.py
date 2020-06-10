from gym import spaces
space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
print(space.to_jsonable(2))
x = space.sample()
assert space.contains(x)
assert space.n == 8