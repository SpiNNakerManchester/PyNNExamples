import dendrite_exc_inh_stimuli as t1
import soma_exc_inh_stimuli as t2
import mixed_static_dend_soma_stimuli as t3
import static_rate_test as t4
import dendritic_plasticity_test as t5
import dendritic_plasticity_teaching_current_test as t6
import dendritic_prediction_sin_soma as t7
import pyramidal_static_stimuli as t8
import pyramidal_basal_plasticity as t9
import pyramidal_full_plasticity as t10
import two_sources_simple_sin as t11
import Urbanczik_Senn_pt2 as t12

from termcolor import colored

def run_test(obj, to_print, graphic):

    if obj == t11 or obj == t12:
        if obj.test(graphic=graphic):
            to_print.append(obj.success_desc())
            return True

        to_print.append(obj.failure_desc())
        return False

    if obj.test():

        to_print.append(obj.success_desc())
        return True

    to_print.append(obj.failure_desc())
    return False

if __name__ == "__main__":

    to_print = []
    failed = False

    options = {
        1: [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12],
        2: [t1, t2, t3, t4],
        3: [t8],
        4: [t1, t2, t3, t4, t5, t6, t7, t11, t12],
        5: [t8, t9, t10],
        6: [t11, t12]
    }

    print "\n\n\n\n"
    print colored("Multicompartment models Base Test", "green")
    print colored("Author Luca Peres, The Univerity of Manchester, 2020", "green")
    print "----------------------------------------------------------"
    print colored("Select one option:", "green")
    print colored("    1 - Full test suite", "green")
    print colored("    2 - Static US model tests", "green")
    print colored("    3 - Static pyramidal model test", "green")
    print colored("    4 - Static + plastic US model tests", "green")
    print colored("    5 - Static + plastic pyramidal model tests", "green")
    print colored("    6 - Advanced tests only", "green")

    val = int(raw_input())

    if val not in options.keys():
        print colored("Not a valid choice, terminating...", "red")
    else:

        test_list = options[val]

        graphic = False

        if t12 in test_list:
            print colored(
                "Some of the tests in your selection have graphic options, do you want to enable these? [Y/N]", "green")
            print colored(
                "    If you decide to enable them the output will take longer to be generated", "green")
            graphics = raw_input()
            if graphics == "Y" or graphics == "y":
                print colored(
                    "Graphic features enabled",
                    "green")
                graphic = True
            elif graphics == "N" or graphics == "n":
                print colored(
                    "Graphic features disabled",
                    "green")
                graphic = False
            else:
                print colored(
                    "Not a valid choice, graphic features disabled by default",
                    "green")
                graphic = False

        for test in test_list:

            if not run_test(test, to_print, graphic):
                failed = True
                break

        for v in to_print[:-1]:

            print colored(v, "green")

        if failed:
            print colored(to_print[-1], "red")
        else:
            print colored(to_print[-1], "green")
            print colored("All tests PASSED!", "green")
