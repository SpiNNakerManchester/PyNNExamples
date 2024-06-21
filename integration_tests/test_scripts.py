# Copyright (c) 2019 The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from spinnaker_testbase import ScriptChecker


class TestScripts(ScriptChecker):
    """
    This file tests the scripts as configured in script_builder.py

    Please do not manually edit this file.
    It is rebuilt each time SpiNNakerManchester/IntegrationTests is run

    If it is out of date please edit and run script_builder.py
    Then the new file can be added to github for reference only.
    """
# flake8: noqa

    def test_examples_synfire_if_curr_exp_large_array(self):
        self.check_script("examples/synfire_if_curr_exp_large_array.py")

    def test_examples_synfire_izk_curr_exp(self):
        self.check_script("examples/synfire_izk_curr_exp.py")

    def test_examples_hbp_neuroguidebook_example(self):
        self.check_script("examples/hbp_neuroguidebook_example.py")

    def test_examples_partitioner_examples_splitter_usage(self):
        self.check_script("examples/partitioner_examples/splitter_usage.py")

    def test_examples_split_examples_structural_plasticity_with_stdp_neuromodulated(self):
        self.check_script("examples/split_examples/structural_plasticity_with_stdp_neuromodulated.py")

    def test_examples_split_examples_stdp_neuromodulated_example_split(self):
        self.check_script("examples/split_examples/stdp_neuromodulated_example_split.py")

    def test_examples_split_examples_pynnBrunnelSplit(self):
        self.check_script("examples/split_examples/pynnBrunnelSplit.py")

    def test_examples_split_examples_va_benchmark_split(self):
        self.check_script("examples/split_examples/va_benchmark_split.py")

    def test_examples_balanced_random_balanced_random(self):
        self.check_script("examples/balanced_random/balanced_random.py")

    def test_examples_balanced_random_split_balanced_random_split(self):
        self.check_script("examples/balanced_random/split/balanced_random_split.py")

    def test_examples_synfire_if_curr_exp_random(self):
        self.check_script("examples/synfire_if_curr_exp_random.py")

    def test_examples_va_benchmark(self):
        self.check_script("examples/va_benchmark.py")

    def test_examples_structural_plasticity_with_stdp_2d(self):
        self.check_script("examples/structural_plasticity_with_stdp_2d.py")

    def test_examples_if_curr_delta(self):
        self.check_script("examples/if_curr_delta.py")

    def test_examples_structural_plasticity_without_stdp_2d(self):
        self.check_script("examples/structural_plasticity_without_stdp_2d.py")

    def test_examples_structural_plasticity_with_stdp_neuromodulated_separate_pops(self):
        self.check_script("examples/structural_plasticity_with_stdp_neuromodulated_separate_pops.py")

    def test_examples_external_devices_examples_retina_example(self):
        self.check_script("examples/external_devices_examples/retina_example.py")

    def test_examples_external_devices_examples_motor_example(self):
        self.check_script("examples/external_devices_examples/motor_example.py")

    def test_examples_external_devices_examples_live_examples_spike_io(self):
        self.check_script("examples/external_devices_examples/live_examples/spike_io.py")

    def test_examples_external_devices_examples_live_examples_synfire_if_curr_exp_live(self):
        self.check_script("examples/external_devices_examples/live_examples/synfire_if_curr_exp_live.py")

    def test_examples_external_devices_examples_live_examples_spike_io_interactive_demo_with_c_vis(self):
        self.check_script("examples/external_devices_examples/live_examples/spike_io_interactive_demo_with_c_vis.py")

    def test_examples_external_devices_examples_live_examples_balanced_random_live_rate(self):
        self.check_script("examples/external_devices_examples/live_examples/balanced_random_live_rate.py")

    # Not testing file due to: Runs forever
    # examples/external_devices_examples/pushbot_light_follower.py

    # Not testing file due to: Needs a physical pushbot
    # examples/external_devices_examples/pushbot_ethernet_example.py

    def test_examples_external_devices_examples_pushbot_spinnaker_link_example(self):
        self.check_script("examples/external_devices_examples/pushbot_spinnaker_link_example.py")

    def test_examples_if_curr_alpha(self):
        self.check_script("examples/if_curr_alpha.py")

    def test_examples_stdp_example_izk(self):
        self.check_script("examples/stdp_example_izk.py")

    def test_examples_synfire_if_curr_exp(self):
        self.check_script("examples/synfire_if_curr_exp.py")

    def test_examples_spike_time_compare(self):
        self.check_script("examples/spike_time_compare.py")

    def test_examples_stdp_example_cond(self):
        self.check_script("examples/stdp_example_cond.py")

    def test_examples_stdp_pairing(self):
        self.check_script("examples/stdp_pairing.py")

    def test_examples_simple_STDP(self):
        self.check_script("examples/simple_STDP.py")

    def test_examples_stdp_neuromodulation_test(self):
        self.check_script("examples/stdp_neuromodulation_test.py")

    def test_examples_pynnBrunnel(self):
        self.check_script("examples/pynnBrunnel.py")

    def test_examples_synfire_if_curr_exp_get_weights(self):
        self.check_script("examples/synfire_if_curr_exp_get_weights.py")

    def test_examples_synfire_if_cond_exp(self):
        self.check_script("examples/synfire_if_cond_exp.py")

    def test_examples_stdp_neuromodulated_example(self):
        self.check_script("examples/stdp_neuromodulated_example.py")

    def test_examples_stdp_curve_cond(self):
        self.check_script("examples/stdp_curve_cond.py")

    def test_examples_stdp_curve(self):
        self.check_script("examples/stdp_curve.py")

    def test_examples_current_injection_example(self):
        self.check_script("examples/current_injection_example.py")

    def test_examples_stdp_example(self):
        self.check_script("examples/stdp_example.py")

    def test_examples_stdp_example_get_plastic_params(self):
        self.check_script("examples/stdp_example_get_plastic_params.py")

    def test_examples_extra_models_examples_LGN_Izhikevich(self):
        self.check_script("examples/extra_models_examples/LGN_Izhikevich.py")

    def test_examples_extra_models_examples_vogel_2011_vogels_2011_live(self):
        self.check_script("examples/extra_models_examples/vogel_2011/vogels_2011_live.py")

    def test_examples_extra_models_examples_vogel_2011_vogels_2011(self):
        self.check_script("examples/extra_models_examples/vogel_2011/vogels_2011.py")

    def test_examples_extra_models_examples_stdp_associative_memory(self):
        self.check_script("examples/extra_models_examples/stdp_associative_memory.py")

    def test_examples_extra_models_examples_stdp_triplet(self):
        self.check_script("examples/extra_models_examples/stdp_triplet.py")

    def test_examples_extra_models_examples_synfire_if_curr_dual_exp(self):
        self.check_script("examples/extra_models_examples/synfire_if_curr_dual_exp.py")

    def test_examples_extra_models_examples_IF_curr_exp_sEMD(self):
        self.check_script("examples/extra_models_examples/IF_curr_exp_sEMD.py")

    def test_examples_extra_models_examples_IF_curr_delta(self):
        self.check_script("examples/extra_models_examples/IF_curr_delta.py")

    def test_examples_extra_models_examples_stdp_example_izk_cond(self):
        self.check_script("examples/extra_models_examples/stdp_example_izk_cond.py")

    def test_examples_extra_models_examples_IF_curr_exp_ca2_adaptive(self):
        self.check_script("examples/extra_models_examples/IF_curr_exp_ca2_adaptive.py")

    def test_examples_extra_models_examples_IF_cond_exp_stoc(self):
        self.check_script("examples/extra_models_examples/IF_cond_exp_stoc.py")

    def test_balanced_random_balanced_random(self):
        self.check_script("balanced_random/balanced_random.py")

    def test_balanced_random_split_balanced_random_split(self):
        self.check_script("balanced_random/split/balanced_random_split.py")

    def test_learning_random_dist(self):
        self.check_script("learning/random_dist.py")

    def test_learning_stdp(self):
        self.check_script("learning/stdp.py")

    def test_learning_struct_pl(self):
        self.check_script("learning/struct_pl.py")

    def test_learning_split_struct_pl_stdp_split(self):
        self.check_script("learning/split/struct_pl_stdp_split.py")

    def test_learning_split_struct_pl_split(self):
        self.check_script("learning/split/struct_pl_split.py")

    def test_learning_split_stdp_split(self):
        self.check_script("learning/split/stdp_split.py")

    def test_learning_simple(self):
        self.check_script("learning/simple.py")

    def test_learning_struct_pl_stdp(self):
        self.check_script("learning/struct_pl_stdp.py")

    def test_synfire_synfire(self):
        self.check_script("synfire/synfire.py")

    def test_synfire_synfire_collab(self):
        self.check_script("synfire/synfire_collab.py")
