# coding=utf-8
import spynnaker8 as p

class Layer23:

    def __init__(self, L23_en, L23_in, L23_ew, L23_iw, L23_ed, L23_id):

        self.L23_exc_n = L23_en
        self.L23_inh_n = L23_in
        self.w_exc = L23_ew
        self.w_inh = L23_iw
        self.d_exc = L23_ed
        self.d_inh = L23_id

        # === L2/3 Neuron parameters ====

        self.L23_cell_params = {
            'tau_m': 30.0,
            'cm': 0.16,
            'v_rest': -65.0,
            'v_reset': -72.0,
            'v_thresh': -40.0,
            'tau_syn_E': 5.0,
            'tau_syn_I': 15.0,
            'tau_refrac': 10,
            'i_offset': 0}

        # === L4 Excitatory and Inhibitory populations ======

        self.L23_exc_cells = p.Population(
            self.L23_exc_n, p.IF_curr_exp(**self.L23_cell_params), label="L2/3 Excitatory_Cells")
        self.L23_inh_cells = p.Population(
            self.L23_exc_n, p.IF_curr_exp(**self.L23_cell_params), label="L2/3 Inhibitory_Cells")

    def Intralayerconnector(self, exconn, inconn):

        self.exc_conn = exconn
        self.inh_conn = inconn

        self.connections = {
            # === Layer 2/3 intra ======
            'L2/3e-e': p.Projection(
                self.L23_exc_cells, self.L23_exc_cells, self.exc_conn, receptor_type='excitatory',
                synapse_type=p.StaticSynapse(weight=self.w_exc, delay=self.d_exc)),
            'L2/3e-i': p.Projection(
                self.L23_exc_cells, self.L23_inh_cells, self.exc_conn, receptor_type='excitatory',
                synapse_type=p.StaticSynapse(weight=self.w_exc, delay=self.d_exc)),
            'L2/3i-e': p.Projection(
                self.L23_inh_cells, self.L23_exc_cells, self.inh_conn, receptor_type='inhibitory',
                synapse_type=p.StaticSynapse(weight=self.w_inh, delay=self.d_inh)),
            'L2/3i-i': p.Projection(
                self.L23_inh_cells, self.L23_inh_cells, self.inh_conn, receptor_type='inhibitory',
                synapse_type=p.StaticSynapse(weight=self.w_inh, delay=self.d_inh)), }

    def Interbarrelconnector(self, econn_target, iconn_target, e_weight, i_weight, e_del, connect_prob):

        self.xL4_exc_cells = econn_target
        self.xL4_inh_cells = iconn_target
        self.exc_weight = e_weight
        self.inh_weight = i_weight
        self.d_exc = e_del
        self.conn_prob = connect_prob

        self.connections = {
        # Layer 2/3 - Adjacent barrel ========
        'L4e-x-L4e': p.Projection(
            self.L23_exc_cells, self.xL4_exc_cells, p.FixedProbabilityConnector(self.conn_prob),
            synapse_type=p.StaticSynapse(weight=self.exc_weight, delay=self.d_exc), receptor_type='excitatory'),
        'L4e-L23i': p.Projection(
            self.L23_exc_cells, self.xL4_inh_cells, p.FixedProbabilityConnector(self.conn_prob),
            synapse_type=p.StaticSynapse(weight=self.inh_weight, delay=self.d_exc), receptor_type='excitatory'),
        }