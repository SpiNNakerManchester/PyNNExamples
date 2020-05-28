import examples.multicomp_tests.multicomp_rate_plasticity_test as multicomp_rate_plasticity_test
import dendrite_exc_inh_stimuli as t1
import soma_exc_inh_stimuli as t2
import mixed_static_dend_soma_stimuli as t3

from termcolor import colored

def run_test(obj, to_print):

    if obj.test():

        to_print.append(obj.success_desc())
        return True

    to_print.append(obj.failure_desc())
    return False

if __name__ == "__main__":

    to_print = []
    test_list = [t1, t2, t3]
    failed = False

    for test in test_list:

        if not run_test(test, to_print):
            failed = True
            break

    print "\n\n\n\n"
    print colored("Urbanczik-Senn model Base Test", "green")
    print colored("Author Luca Peres, The Univerity of Manchester", "green")
    print "----------------------------------------------------------"

    for val in to_print[:-1]:

        print colored(val, "green")

    if failed:
        print colored(to_print[-1], "red")
    else:
        print colored(to_print[-1], "green")
        print colored("All tests PASSED!", "green")
