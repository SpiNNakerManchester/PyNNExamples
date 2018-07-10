# coding=utf-8
import spynnaker8 as p

class Layer4:

    def __init__(self, L4_en, L4_in, L4_ew, L4_iw, L4_ed, L4_id):

        self.L4_exc_n = L4_en
        self.L4_inh_n = L4_in
        self.w_exc = L4_ew
        self.w_inh = L4_iw
        self.d_exc = L4_ed
        self.d_inh = L4_id

        # === L4 Neuron parameters ====

        self.L4_cell_params = {
            'tau_m': 35.0,
            'cm': 0.12,
            'v_rest': -65.0,
            'v_reset': -66.0,
            'v_thresh': -40.0,
            'tau_syn_E': 5.0,
            'tau_syn_I': 15.0,
            'tau_refrac': 10,
            'i_offset': 0}

        # === L4 Excitatory and Inhibitory populations ======

        self.L4_exc_cells = p.Population(
            self.L4_exc_n, p.IF_curr_exp(**self.L4_cell_params), label="L4 Excitatory_Cells")
        self.L4_inh_cells = p.Population(
            self.L4_inh_n, p.IF_curr_exp(**self.L4_cell_params), label="L4 Inhibitory_Cells")

    def Thalconnector(self, src_Thal, W_Thal, D_Thal, Thal_Conn_Prob):

        self.src_Thal = src_Thal
        self.Thal_w = W_Thal
        self.Thal_delay = D_Thal
        self.Thal_prob = Thal_Conn_Prob

        self.connections = {
        # === Thalamic - Layer 4 projections =====
        'Thal-L4e': p.Projection(
            self.src_Thal, self.L4_exc_cells, p.FixedProbabilityConnector(self.Thal_prob),
            p.StaticSynapse(weight=self.Thal_w, delay=self.Thal_delay), receptor_type="excitatory"),
        'Thal-L4i': p.Projection(
            self.src_Thal, self.L4_inh_cells, p.FixedProbabilityConnector(self.Thal_prob),
            p.StaticSynapse(weight=self.Thal_w, delay=self.Thal_delay), receptor_type="excitatory"),
        }

    def Intralayerconnector(self, exconn, inconn, dexc, dinh):

        self.exc_conn = exconn
        self.inh_conn = inconn
        self.d_exc = dexc
        self.d_inh = dinh

        self.connections = {
        # === Layer 4 intra ======
        'L4e-e': p.Projection(
            self.L4_exc_cells, self.L4_exc_cells, self.exc_conn, receptor_type='excitatory',
            synapse_type=p.StaticSynapse(weight=self.w_exc, delay=self.d_exc)),
        'L4e-i': p.Projection(
            self.L4_exc_cells, self.L4_inh_cells, self.exc_conn, receptor_type='excitatory',
            synapse_type=p.StaticSynapse(weight=self.w_exc, delay=self.d_exc)),
        'L4i-e': p.Projection(
            self.L4_inh_cells, self.L4_exc_cells, self.inh_conn, receptor_type='inhibitory',
            synapse_type=p.StaticSynapse(weight=self.w_inh, delay=self.d_inh)),
        'L4i-i': p.Projection(
            self.L4_inh_cells, self.L4_inh_cells, self.inh_conn, receptor_type='inhibitory',
            synapse_type=p.StaticSynapse(weight=self.w_inh, delay=self.d_inh)),
        }

    def Interlayerconnector(self, L23_econn_target, L23_iconn_target, e_weight, e_del):

        self.L23_exc_cells = L23_econn_target
        self.L23_inh_cells = L23_iconn_target
        self.exc_weight = e_weight
        self.d_exc = e_del

        self.connections = {
        # Layer 4-2/3 inter ========
        'L4e-L23e': p.Projection(
            self.L4_exc_cells, self.L23_exc_cells, p.FixedProbabilityConnector(0.1),
            synapse_type=p.StaticSynapse(weight=self.exc_weight, delay=self.d_exc), receptor_type='excitatory'),
        'L4e-L23i': p.Projection(
            self.L4_exc_cells, self.L23_inh_cells, p.FixedProbabilityConnector(0.1),
            synapse_type=p.StaticSynapse(weight=self.exc_weight, delay=self.d_exc), receptor_type='excitatory'),
        }