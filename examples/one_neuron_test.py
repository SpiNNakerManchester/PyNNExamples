import spynnaker8 as p


timestep=1
p.setup(timestep)


pop_ex1 = p.Population(2, p.IF_curr_exp, {}, label="ex1")

# pop_ex1.record('all')

p.run(100)

# res = pop_ex1.get_data('all')

p.end()
print("\n job done")