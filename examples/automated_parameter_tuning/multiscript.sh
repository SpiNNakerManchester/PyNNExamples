 #!/bin/bash
 trap 'trap - SIGINT; kill -SIGINT $$' SIGINT;
 for i in {0..10}
 do
	sleep 2
	 python GA_population.py "logbooks/small_tests/checkpoint_population_$i.pkl" & 
 done
